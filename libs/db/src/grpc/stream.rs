use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use impeller2::types::{ComponentId, Timestamp, msg_id};
use tokio::sync::{mpsc, watch};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use super::{
    common,
    v1::{
        self, StreamComponentsRequest, StreamComponentsResponse, StreamControl,
        StreamMessagesRequest, StreamMessagesResponse, WatchDbRequest, WatchDbResponse,
        stream_components_request, stream_components_response, stream_messages_request,
        stream_service_server::StreamService, watch_db_response,
    },
};
use crate::{Component, DB, msg_log::MsgLog};

#[derive(Clone)]
pub(super) struct StreamServiceImpl {
    db: Arc<DB>,
    playbacks: Arc<Mutex<HashMap<u64, watch::Sender<Playback>>>>,
}

#[derive(Clone, Copy)]
struct Playback {
    playing: bool,
    timestamp: Timestamp,
    seek_generation: u64,
    timestep: Duration,
    frequency: u64,
}

impl StreamServiceImpl {
    pub(super) fn new(db: Arc<DB>) -> Self {
        Self {
            db,
            playbacks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn components(&self, names: &[String]) -> Result<Vec<(String, Component)>, Status> {
        self.db.with_state(|state| {
            if names.is_empty() {
                let mut values = state
                    .components
                    .values()
                    .map(|component| {
                        let name = state
                            .get_component_metadata(component.component_id)
                            .map_or_else(
                                || component.component_id.to_string(),
                                |metadata| metadata.name.clone(),
                            );
                        (name, component.clone())
                    })
                    .collect::<Vec<_>>();
                values.sort_by(|a, b| a.0.cmp(&b.0));
                return Ok(values);
            }
            names
                .iter()
                .map(|name| {
                    state
                        .get_component(ComponentId::new(name))
                        .cloned()
                        .map(|component| (name.clone(), component))
                        .ok_or_else(|| Status::not_found(format!("component {name} not found")))
                })
                .collect()
        })
    }

    fn messages(&self, names: &[String]) -> Result<Vec<(String, MsgLog)>, Status> {
        self.db.with_state(|state| {
            if names.is_empty() {
                let mut values = state
                    .msg_logs
                    .values()
                    .filter_map(|log| {
                        let name = log.metadata()?.name.clone();
                        Some((name, log.clone()))
                    })
                    .collect::<Vec<_>>();
                values.sort_by(|a, b| a.0.cmp(&b.0));
                return Ok(values);
            }
            names
                .iter()
                .map(|name| {
                    state
                        .get_msg_log(msg_id(name))
                        .cloned()
                        .map(|log| (name.clone(), log))
                        .ok_or_else(|| Status::not_found(format!("message {name} not found")))
                })
                .collect()
        })
    }

    fn playback(&self, fixed: &v1::FixedRate) -> Result<Playback, Status> {
        let initial = v1::InitialTimestamp::try_from(fixed.initial)
            .unwrap_or(v1::InitialTimestamp::Unspecified);
        let timestamp = match initial {
            v1::InitialTimestamp::Earliest => self.db.earliest_timestamp.latest(),
            v1::InitialTimestamp::Latest | v1::InitialTimestamp::Unspecified => {
                self.db.last_updated.latest()
            }
            v1::InitialTimestamp::Manual => Timestamp(fixed.initial_timestamp_ns / 1000),
        };
        if fixed.timestep_ns == 0 || fixed.frequency == 0 {
            return Err(Status::invalid_argument(
                "fixed-rate timestep_ns and frequency must be non-zero",
            ));
        }
        Ok(Playback {
            playing: true,
            timestamp,
            seek_generation: 0,
            timestep: Duration::from_nanos(fixed.timestep_ns),
            frequency: fixed.frequency,
        })
    }

    fn register_playback(
        &self,
        playback: Playback,
    ) -> (u64, watch::Sender<Playback>, watch::Receiver<Playback>) {
        loop {
            let id = fastrand::u64(1..);
            let mut playbacks = self.playbacks.lock().unwrap();
            if playbacks.contains_key(&id) {
                continue;
            }
            let (tx, rx) = watch::channel(playback);
            playbacks.insert(id, tx.clone());
            return (id, tx, rx);
        }
    }

    fn playback_sender(&self, id: u64) -> Option<watch::Sender<Playback>> {
        self.playbacks.lock().unwrap().get(&id).cloned()
    }
}

#[tonic::async_trait]
impl StreamService for StreamServiceImpl {
    type StreamComponentsStream = ReceiverStream<Result<StreamComponentsResponse, Status>>;
    type StreamMessagesStream = ReceiverStream<Result<StreamMessagesResponse, Status>>;
    type WatchDbStream = ReceiverStream<Result<WatchDbResponse, Status>>;

    async fn stream_components(
        &self,
        request: Request<tonic::Streaming<StreamComponentsRequest>>,
    ) -> Result<Response<Self::StreamComponentsStream>, Status> {
        let mut incoming = request.into_inner();
        let first = incoming
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("stream requires StreamOpen"))?;
        let Some(stream_components_request::Request::Open(open)) = first.request else {
            return Err(Status::invalid_argument(
                "first stream request must be StreamOpen",
            ));
        };
        let components = self.components(&open.components)?;
        let (tx, rx) = mpsc::channel(32);
        for (name, component) in &components {
            let element_names = self.db.with_state(|state| {
                state
                    .get_component_metadata(component.component_id)
                    .map(|metadata| common::element_names(metadata.element_names()))
                    .unwrap_or_default()
            });
            tx.send(Ok(header(name, component, element_names)))
                .await
                .map_err(|_| Status::cancelled("client closed response stream"))?;
        }

        match open.behavior {
            Some(v1::stream_open::Behavior::FixedRate(fixed)) => {
                let playback = self.playback(&fixed)?;
                let (stream_id, control_tx, control_rx) = self.register_playback(playback);
                tx.send(Ok(StreamComponentsResponse {
                    response: Some(stream_components_response::Response::Opened(
                        v1::StreamOpened { stream_id },
                    )),
                }))
                .await
                .map_err(|_| Status::cancelled("client closed response stream"))?;
                tokio::spawn(read_component_controls(incoming, control_tx));
                tokio::spawn(run_fixed_components(components, tx, control_rx));
            }
            Some(v1::stream_open::Behavior::RealTime(real_time)) if real_time.immediate => {
                tokio::spawn(reject_component_controls(incoming));
                for component in components {
                    tokio::spawn(run_immediate_component(component, tx.clone()));
                }
                drop(tx);
            }
            _ => {
                tokio::spawn(reject_component_controls(incoming));
                let db = self.db.clone();
                tokio::spawn(run_batched_components(db, components, tx));
            }
        }
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn stream_messages(
        &self,
        request: Request<tonic::Streaming<StreamMessagesRequest>>,
    ) -> Result<Response<Self::StreamMessagesStream>, Status> {
        let mut incoming = request.into_inner();
        let first = incoming
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("stream requires MessageStreamOpen"))?;
        let Some(stream_messages_request::Request::Open(open)) = first.request else {
            return Err(Status::invalid_argument(
                "first stream request must be MessageStreamOpen",
            ));
        };
        let messages = self.messages(&open.messages)?;
        let playback_stream_id = open.playback_stream_id;
        let (tx, rx) = mpsc::channel(32);
        match open.behavior {
            Some(v1::message_stream_open::Behavior::FixedRate(fixed)) => {
                let (control_tx, control_rx) = if playback_stream_id == 0 {
                    watch::channel(self.playback(&fixed)?)
                } else {
                    let control_tx = self
                        .playback_sender(playback_stream_id)
                        .ok_or_else(|| Status::not_found("playback_stream_id not found"))?;
                    let control_rx = control_tx.subscribe();
                    (control_tx, control_rx)
                };
                tokio::spawn(read_message_controls(incoming, control_tx));
                tokio::spawn(run_fixed_messages(messages, tx, control_rx));
            }
            _ if playback_stream_id != 0 => {
                return Err(Status::invalid_argument(
                    "playback_stream_id requires fixed-rate behavior",
                ));
            }
            _ => {
                tokio::spawn(reject_message_controls(incoming));
                for message in messages {
                    tokio::spawn(run_live_message(message, tx.clone()));
                }
                drop(tx);
            }
        }
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn watch_db(
        &self,
        _request: Request<WatchDbRequest>,
    ) -> Result<Response<Self::WatchDbStream>, Status> {
        let db = self.db.clone();
        let (tx, rx) = mpsc::channel(8);
        tokio::spawn(async move {
            let mut last_timestamp = db.last_updated.latest();
            let mut config_generation = db.db_config_gen.latest();
            if send_db_events(&db, &tx, last_timestamp).await.is_err() {
                return;
            }
            loop {
                let seen_timestamp = last_timestamp;
                let seen_generation = config_generation;
                futures_lite::future::race(
                    db.last_updated
                        .wait_for(move |value| value != seen_timestamp),
                    db.db_config_gen
                        .wait_for(move |value| value != seen_generation),
                )
                .await;
                let timestamp = db.last_updated.latest();
                let generation = db.db_config_gen.latest();
                if timestamp != last_timestamp
                    && tx
                        .send(Ok(WatchDbResponse {
                            event: Some(watch_db_response::Event::LastUpdatedNs(
                                timestamp.0.saturating_mul(1000),
                            )),
                        }))
                        .await
                        .is_err()
                {
                    return;
                }
                if generation != config_generation && send_db_config(&db, &tx).await.is_err() {
                    return;
                }
                last_timestamp = timestamp;
                config_generation = generation;
            }
        });
        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

fn header(
    name: &str,
    component: &Component,
    element_names: Vec<String>,
) -> StreamComponentsResponse {
    StreamComponentsResponse {
        response: Some(stream_components_response::Response::Header(
            v1::TimeSeriesHeader {
                component: name.to_string(),
                prim_type: common::prim_type(component.schema.prim_type) as i32,
                dims: component.schema.shape().into_vec(),
                element_names,
            },
        )),
    }
}

fn update(name: &str, timestamp: Timestamp, value: &[u8]) -> StreamComponentsResponse {
    StreamComponentsResponse {
        response: Some(stream_components_response::Response::Update(
            v1::ComponentUpdate {
                component: name.to_string(),
                timestamp_ns: timestamp.0.saturating_mul(1000),
                packed_value: value.to_vec(),
            },
        )),
    }
}

async fn run_batched_components(
    db: Arc<DB>,
    components: Vec<(String, Component)>,
    tx: mpsc::Sender<Result<StreamComponentsResponse, Status>>,
) {
    let mut sent = HashMap::<ComponentId, (Timestamp, Vec<u8>)>::new();
    loop {
        for (name, component) in &components {
            let Some((timestamp, value)) = component.time_series.latest() else {
                continue;
            };
            let current = (*timestamp, value.to_vec());
            if sent.get(&component.component_id) == Some(&current) {
                continue;
            }
            if tx.send(Ok(update(name, *timestamp, value))).await.is_err() {
                return;
            }
            sent.insert(component.component_id, current);
        }
        db.last_updated.wait().await;
    }
}

async fn run_immediate_component(
    (name, component): (String, Component),
    tx: mpsc::Sender<Result<StreamComponentsResponse, Status>>,
) {
    let waiter = component.time_series.waiter();
    let mut sent = None;
    loop {
        if let Some((timestamp, value)) = component.time_series.latest() {
            let current = (*timestamp, value.to_vec());
            if sent.as_ref() != Some(&current) {
                if tx.send(Ok(update(&name, *timestamp, value))).await.is_err() {
                    return;
                }
                sent = Some(current);
            }
        }
        let _ = waiter.wait().await;
    }
}

async fn run_fixed_components(
    components: Vec<(String, Component)>,
    tx: mpsc::Sender<Result<StreamComponentsResponse, Status>>,
    mut control: watch::Receiver<Playback>,
) {
    let mut cursor = control.borrow().timestamp;
    let mut seek_generation = control.borrow().seek_generation;
    let mut emitted = None;
    loop {
        let state = *control.borrow();
        if !state.playing {
            if emitted != Some(cursor) {
                if !send_component_frame(&components, &tx, cursor).await {
                    return;
                }
                emitted = Some(cursor);
            }
            if control.changed().await.is_err() {
                return;
            }
            apply_seek(&control, &mut cursor, &mut seek_generation);
            continue;
        }
        if !send_component_frame(&components, &tx, cursor).await {
            return;
        }
        emitted = Some(cursor);
        let sleep = tokio::time::sleep(Duration::from_secs_f64(1.0 / state.frequency as f64));
        tokio::pin!(sleep);
        tokio::select! {
            _ = &mut sleep => {
                cursor = Timestamp(
                    cursor.0.saturating_add((state.timestep.as_nanos() / 1000) as i64)
                );
            }
            result = control.changed() => {
                if result.is_err() {
                    return;
                }
                apply_seek(&control, &mut cursor, &mut seek_generation);
            }
        }
    }
}

async fn send_component_frame(
    components: &[(String, Component)],
    tx: &mpsc::Sender<Result<StreamComponentsResponse, Status>>,
    cursor: Timestamp,
) -> bool {
    for (name, component) in components {
        if let Some((timestamp, value)) = component.time_series.get_nearest(cursor)
            && tx.send(Ok(update(name, timestamp, value))).await.is_err()
        {
            return false;
        }
    }
    tx.send(Ok(StreamComponentsResponse {
        response: Some(stream_components_response::Response::Timestamp(
            v1::StreamTimestamp {
                timestamp_ns: cursor.0.saturating_mul(1000),
            },
        )),
    }))
    .await
    .is_ok()
}

fn apply_control(state: &mut Playback, control: StreamControl) -> Result<(), Status> {
    if let Some(playing) = control.playing {
        state.playing = playing;
    }
    if let Some(timestamp) = control.seek_ns {
        state.timestamp = Timestamp(timestamp / 1000);
        state.seek_generation = state.seek_generation.wrapping_add(1);
    }
    if let Some(timestep) = control.timestep_ns {
        state.timestep = common::duration(timestep)?;
    }
    if let Some(frequency) = control.frequency {
        if frequency == 0 {
            return Err(Status::invalid_argument("frequency must be non-zero"));
        }
        state.frequency = frequency;
    }
    Ok(())
}

async fn read_component_controls(
    mut incoming: tonic::Streaming<StreamComponentsRequest>,
    control: watch::Sender<Playback>,
) {
    while let Ok(Some(request)) = incoming.message().await {
        let Some(stream_components_request::Request::Control(update)) = request.request else {
            continue;
        };
        let mut next = *control.borrow();
        if apply_control(&mut next, update).is_err() || control.send(next).is_err() {
            return;
        }
    }
}

async fn read_message_controls(
    mut incoming: tonic::Streaming<StreamMessagesRequest>,
    control: watch::Sender<Playback>,
) {
    while let Ok(Some(request)) = incoming.message().await {
        let Some(stream_messages_request::Request::Control(update)) = request.request else {
            continue;
        };
        let mut next = *control.borrow();
        if apply_control(&mut next, update).is_err() || control.send(next).is_err() {
            return;
        }
    }
}

async fn reject_component_controls(mut incoming: tonic::Streaming<StreamComponentsRequest>) {
    while let Ok(Some(_)) = incoming.message().await {}
}

async fn reject_message_controls(mut incoming: tonic::Streaming<StreamMessagesRequest>) {
    while let Ok(Some(_)) = incoming.message().await {}
}

async fn run_live_message(
    (name, log): (String, MsgLog),
    tx: mpsc::Sender<Result<StreamMessagesResponse, Status>>,
) {
    let waiter = log.waiter();
    let mut sent = None;
    loop {
        if let Some((timestamp, payload)) = log.latest()
            && sent != Some(timestamp)
        {
            if tx
                .send(Ok(StreamMessagesResponse {
                    name: name.clone(),
                    timestamp_ns: timestamp.0.saturating_mul(1000),
                    payload: payload.to_vec(),
                }))
                .await
                .is_err()
            {
                return;
            }
            sent = Some(timestamp);
        }
        let _ = waiter.wait().await;
    }
}

async fn run_fixed_messages(
    messages: Vec<(String, MsgLog)>,
    tx: mpsc::Sender<Result<StreamMessagesResponse, Status>>,
    mut control: watch::Receiver<Playback>,
) {
    let mut sent = HashMap::<String, Timestamp>::new();
    let mut cursor = control.borrow().timestamp;
    let mut seek_generation = control.borrow().seek_generation;
    loop {
        let state = *control.borrow();
        if !state.playing {
            if !send_message_frame(&messages, &tx, cursor, &mut sent).await {
                return;
            }
            if control.changed().await.is_err() {
                return;
            }
            if apply_seek(&control, &mut cursor, &mut seek_generation) {
                sent.clear();
            }
            continue;
        }
        if !send_message_frame(&messages, &tx, cursor, &mut sent).await {
            return;
        }
        let sleep = tokio::time::sleep(Duration::from_secs_f64(1.0 / state.frequency as f64));
        tokio::pin!(sleep);
        tokio::select! {
            _ = &mut sleep => {
                cursor = Timestamp(
                    cursor.0.saturating_add((state.timestep.as_nanos() / 1000) as i64)
                );
            }
            result = control.changed() => {
                if result.is_err() {
                    return;
                }
                if apply_seek(&control, &mut cursor, &mut seek_generation) {
                    sent.clear();
                }
            }
        }
    }
}

fn apply_seek(
    control: &watch::Receiver<Playback>,
    cursor: &mut Timestamp,
    generation: &mut u64,
) -> bool {
    let state = *control.borrow();
    if state.seek_generation == *generation {
        return false;
    }
    *cursor = state.timestamp;
    *generation = state.seek_generation;
    true
}

async fn send_message_frame(
    messages: &[(String, MsgLog)],
    tx: &mpsc::Sender<Result<StreamMessagesResponse, Status>>,
    cursor: Timestamp,
    sent: &mut HashMap<String, Timestamp>,
) -> bool {
    for (name, log) in messages {
        let Some((timestamp, payload)) = log.get_nearest(cursor) else {
            continue;
        };
        if sent.get(name) == Some(&timestamp) {
            continue;
        }
        if tx
            .send(Ok(StreamMessagesResponse {
                name: name.clone(),
                timestamp_ns: timestamp.0.saturating_mul(1000),
                payload: payload.to_vec(),
            }))
            .await
            .is_err()
        {
            return false;
        }
        sent.insert(name.clone(), timestamp);
    }
    true
}

async fn send_db_events(
    db: &DB,
    tx: &mpsc::Sender<Result<WatchDbResponse, Status>>,
    timestamp: Timestamp,
) -> Result<(), ()> {
    tx.send(Ok(WatchDbResponse {
        event: Some(watch_db_response::Event::LastUpdatedNs(
            timestamp.0.saturating_mul(1000),
        )),
    }))
    .await
    .map_err(|_| ())?;
    send_db_config(db, tx).await
}

async fn send_db_config(
    db: &DB,
    tx: &mpsc::Sender<Result<WatchDbResponse, Status>>,
) -> Result<(), ()> {
    let value = db.with_state(|state| common::db_config(&state.db_config));
    tx.send(Ok(WatchDbResponse {
        event: Some(watch_db_response::Event::Config(value)),
    }))
    .await
    .map_err(|_| ())
}

#[cfg(test)]
mod tests {
    use impeller2::types::PrimType;
    use impeller2_wkt::SetDbConfig;
    use tempfile::TempDir;

    use super::*;
    use crate::ComponentSchema;

    fn component(directory: &TempDir) -> Component {
        let component = Component::create(
            directory.path(),
            ComponentId::new("demo.signal"),
            "demo.signal".into(),
            ComponentSchema::new(PrimType::F64, &[]),
            Timestamp(100),
        )
        .unwrap();
        for timestamp in 100..=102 {
            component
                .time_series
                .push_buf(Timestamp(timestamp), &(timestamp as f64).to_le_bytes())
                .unwrap();
        }
        component
    }

    #[tokio::test]
    async fn fixed_stream_renders_paused_seek() {
        let directory = TempDir::new().unwrap();
        let component = component(&directory);
        let (tx, mut rx) = mpsc::channel(8);
        let initial = Playback {
            playing: false,
            timestamp: Timestamp(100),
            seek_generation: 0,
            timestep: Duration::from_micros(1),
            frequency: 100,
        };
        let (control_tx, control_rx) = watch::channel(initial);
        let task = tokio::spawn(run_fixed_components(
            vec![("demo.signal".into(), component)],
            tx,
            control_rx,
        ));

        let first = rx.recv().await.unwrap().unwrap();
        assert!(matches!(
            first.response,
            Some(stream_components_response::Response::Update(_))
        ));
        let first_clock = rx.recv().await.unwrap().unwrap();
        assert!(matches!(
            first_clock.response,
            Some(stream_components_response::Response::Timestamp(
                v1::StreamTimestamp {
                    timestamp_ns: 100_000
                }
            ))
        ));

        let mut seek = initial;
        seek.timestamp = Timestamp(102);
        seek.seek_generation = 1;
        control_tx.send(seek).unwrap();
        let update = rx.recv().await.unwrap().unwrap();
        let Some(stream_components_response::Response::Update(update)) = update.response else {
            panic!("expected component update");
        };
        assert_eq!(update.timestamp_ns, 102_000);
        let clock = rx.recv().await.unwrap().unwrap();
        assert!(matches!(
            clock.response,
            Some(stream_components_response::Response::Timestamp(
                v1::StreamTimestamp {
                    timestamp_ns: 102_000
                }
            ))
        ));
        task.abort();
    }

    #[test]
    fn control_updates_playback_policy() {
        let mut state = Playback {
            playing: true,
            timestamp: Timestamp(1),
            seek_generation: 0,
            timestep: Duration::from_millis(1),
            frequency: 60,
        };
        apply_control(
            &mut state,
            StreamControl {
                playing: Some(false),
                seek_ns: Some(5_000),
                timestep_ns: Some(2_000),
                frequency: Some(120),
            },
        )
        .unwrap();
        assert!(!state.playing);
        assert_eq!(state.timestamp, Timestamp(5));
        assert_eq!(state.seek_generation, 1);
        assert_eq!(state.timestep, Duration::from_nanos(2_000));
        assert_eq!(state.frequency, 120);
    }

    #[tokio::test]
    async fn message_stream_can_share_component_playback() {
        let directory = TempDir::new().unwrap();
        let db = Arc::new(DB::create(directory.path().join("db")).unwrap());
        let service = StreamServiceImpl::new(db);
        let playback = Playback {
            playing: true,
            timestamp: Timestamp(10),
            seek_generation: 0,
            timestep: Duration::from_millis(1),
            frequency: 60,
        };
        let (stream_id, _, mut component_clock) = service.register_playback(playback);
        let message_clock = service.playback_sender(stream_id).unwrap();
        let mut updated = playback;
        updated.playing = false;
        message_clock.send(updated).unwrap();
        component_clock.changed().await.unwrap();
        assert!(!component_clock.borrow().playing);
    }

    #[tokio::test]
    async fn live_message_stream_tails_updates() {
        let directory = TempDir::new().unwrap();
        let log = MsgLog::create(directory.path()).unwrap();
        log.push(Timestamp(10), b"first").unwrap();
        let (tx, mut rx) = mpsc::channel(4);
        let task = tokio::spawn(run_live_message(("demo.log".into(), log.clone()), tx));
        assert_eq!(rx.recv().await.unwrap().unwrap().payload, b"first");
        log.push(Timestamp(11), b"second").unwrap();
        assert_eq!(rx.recv().await.unwrap().unwrap().payload, b"second");
        task.abort();
    }

    #[tokio::test]
    async fn watch_db_pushes_config_changes() {
        let directory = TempDir::new().unwrap();
        let db = Arc::new(DB::create(directory.path().join("db")).unwrap());
        let service = StreamServiceImpl::new(db.clone());
        let mut stream = service
            .watch_db(Request::new(WatchDbRequest {}))
            .await
            .unwrap()
            .into_inner();
        let _ = futures_lite::StreamExt::next(&mut stream).await.unwrap();
        let _ = futures_lite::StreamExt::next(&mut stream).await.unwrap();
        db.apply_set_db_config(SetDbConfig {
            recording: None,
            metadata: [("demo".into(), "value".into())].into_iter().collect(),
        })
        .unwrap();
        let response = tokio::time::timeout(
            Duration::from_secs(1),
            futures_lite::StreamExt::next(&mut stream),
        )
        .await
        .unwrap()
        .unwrap()
        .unwrap();
        let Some(watch_db_response::Event::Config(config)) = response.event else {
            panic!("expected config event");
        };
        assert_eq!(config.metadata["demo"], "value");
    }
}
