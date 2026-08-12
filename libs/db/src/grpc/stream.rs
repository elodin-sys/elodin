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

struct PlaybackRegistration {
    id: u64,
    playbacks: Arc<Mutex<HashMap<u64, watch::Sender<Playback>>>>,
}

impl Drop for PlaybackRegistration {
    fn drop(&mut self) {
        self.playbacks.lock().unwrap().remove(&self.id);
    }
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
            v1::InitialTimestamp::Manual => common::record_timestamp(fixed.initial_timestamp_ns),
        };
        validate_timestep(fixed.timestep_ns)?;
        validate_frequency(fixed.frequency)?;
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
    ) -> (
        u64,
        watch::Sender<Playback>,
        watch::Receiver<Playback>,
        PlaybackRegistration,
    ) {
        loop {
            let id = fastrand::u64(1..);
            let mut playbacks = self.playbacks.lock().unwrap();
            if playbacks.contains_key(&id) {
                continue;
            }
            let (tx, rx) = watch::channel(playback);
            playbacks.insert(id, tx.clone());
            let registration = PlaybackRegistration {
                id,
                playbacks: self.playbacks.clone(),
            };
            return (id, tx, rx, registration);
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
                let (stream_id, control_tx, _control_rx, registration) =
                    self.register_playback(playback);
                tx.send(Ok(StreamComponentsResponse {
                    response: Some(stream_components_response::Response::Opened(
                        v1::StreamOpened { stream_id },
                    )),
                }))
                .await
                .map_err(|_| Status::cancelled("client closed response stream"))?;
                tokio::spawn(read_component_controls(
                    incoming,
                    control_tx.clone(),
                    tx.clone(),
                ));
                tokio::spawn(run_fixed_components(
                    components,
                    tx,
                    control_tx,
                    registration,
                ));
            }
            Some(v1::stream_open::Behavior::RealTime(real_time)) if real_time.immediate => {
                tokio::spawn(reject_component_controls(incoming, tx.clone()));
                for component in components {
                    tokio::spawn(run_immediate_component(component, tx.clone()));
                }
                drop(tx);
            }
            _ => {
                tokio::spawn(reject_component_controls(incoming, tx.clone()));
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
        let (tx, rx) = mpsc::channel(32);
        match open.behavior {
            Some(v1::message_stream_open::Behavior::FixedRate(fixed)) => {
                match open.playback_stream_id {
                    None => {
                        let (control_tx, _control_rx) = watch::channel(self.playback(&fixed)?);
                        tokio::spawn(read_message_controls(
                            incoming,
                            control_tx.clone(),
                            tx.clone(),
                        ));
                        tokio::spawn(run_fixed_messages(messages, tx, control_tx));
                    }
                    Some(0) => {
                        return Err(Status::invalid_argument(
                            "playback_stream_id must be non-zero",
                        ));
                    }
                    Some(id) => {
                        let control_tx = self
                            .playback_sender(id)
                            .ok_or_else(|| Status::not_found("playback_stream_id not found"))?;
                        let control_rx = control_tx.subscribe();
                        drop(control_tx);
                        tokio::spawn(reject_message_controls(incoming, tx.clone()));
                        tokio::spawn(follow_shared_messages(messages, tx, control_rx));
                    }
                }
            }
            _ if open.playback_stream_id.is_some() => {
                return Err(Status::invalid_argument(
                    "playback_stream_id requires fixed-rate behavior",
                ));
            }
            _ => {
                tokio::spawn(reject_message_controls(incoming, tx.clone()));
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
                tokio::select! {
                    _ = futures_lite::future::race(
                        db.last_updated.wait_for(move |value| value != seen_timestamp),
                        db.db_config_gen.wait_for(move |value| value != seen_generation),
                    ) => {}
                    _ = tx.closed() => return,
                }
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
        tokio::select! {
            _ = db.last_updated.wait() => {}
            _ = tx.closed() => return,
        }
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
        tokio::select! {
            _ = waiter.wait() => {}
            _ = tx.closed() => return,
        }
    }
}

// Owns the playback clock: the advancing cursor is published through the
// watch channel so late-attaching shared message streams start at, and stay
// on, the current position.
async fn run_fixed_components(
    components: Vec<(String, Component)>,
    tx: mpsc::Sender<Result<StreamComponentsResponse, Status>>,
    control_tx: watch::Sender<Playback>,
    _registration: PlaybackRegistration,
) {
    let mut control = control_tx.subscribe();
    let mut emitted = None;
    let mut seek_generation = control.borrow().seek_generation;
    loop {
        let state = *control.borrow_and_update();
        // A seek always resamples, even to the current position.
        if state.seek_generation != seek_generation {
            seek_generation = state.seek_generation;
            emitted = None;
        }
        if emitted != Some(state.timestamp)
            && !send_component_frame(&components, &tx, state.timestamp).await
        {
            return;
        }
        emitted = Some(state.timestamp);
        if !state.playing {
            tokio::select! {
                result = control.changed() => {
                    if result.is_err() {
                        return;
                    }
                }
                _ = tx.closed() => return,
            }
            continue;
        }
        let sleep = tokio::time::sleep(Duration::from_secs_f64(1.0 / state.frequency as f64));
        tokio::pin!(sleep);
        tokio::select! {
            _ = &mut sleep => {
                advance_cursor(&control_tx, &state);
            }
            result = control.changed() => {
                if result.is_err() {
                    return;
                }
            }
            _ = tx.closed() => return,
        }
    }
}

// `state` is the playback snapshot the elapsed frame was emitted with; a seek
// that landed since then wins the race and the advance is skipped so the seek
// target is always emitted.
fn advance_cursor(control_tx: &watch::Sender<Playback>, state: &Playback) {
    let step = (state.timestep.as_nanos() / 1000) as i64;
    control_tx.send_modify(|playback| {
        if playback.seek_generation == state.seek_generation {
            playback.timestamp = Timestamp(playback.timestamp.0.saturating_add(step));
        }
    });
}

async fn send_component_frame(
    components: &[(String, Component)],
    tx: &mpsc::Sender<Result<StreamComponentsResponse, Status>>,
    cursor: Timestamp,
) -> bool {
    for (name, component) in components {
        // get_nearest snaps to the first sample when the cursor precedes all
        // data; playback must never emit samples ahead of its clock.
        if let Some((timestamp, value)) = component.time_series.get_nearest(cursor)
            && timestamp <= cursor
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

fn validate_timestep(timestep_ns: u64) -> Result<(), Status> {
    if timestep_ns < 1_000 {
        return Err(Status::invalid_argument(
            "timestep_ns must be at least 1000: the database records at 1 µs resolution",
        ));
    }
    Ok(())
}

// An unbounded frequency yields a zero-duration frame sleep and a busy loop.
const MAX_FRAME_FREQUENCY: u64 = 1_000;

fn validate_frequency(frequency: u64) -> Result<(), Status> {
    if frequency == 0 || frequency > MAX_FRAME_FREQUENCY {
        return Err(Status::invalid_argument(format!(
            "frequency must be 1..={MAX_FRAME_FREQUENCY}"
        )));
    }
    Ok(())
}

// Validates every field before mutating so a rejected control leaves the
// shared playback state untouched.
fn apply_control(state: &mut Playback, control: StreamControl) -> Result<(), Status> {
    if let Some(timestep) = control.timestep_ns {
        validate_timestep(timestep)?;
    }
    if let Some(frequency) = control.frequency {
        validate_frequency(frequency)?;
    }
    if let Some(playing) = control.playing {
        state.playing = playing;
    }
    if let Some(timestamp) = control.seek_ns {
        state.timestamp = common::record_timestamp(timestamp);
        state.seek_generation = state.seek_generation.wrapping_add(1);
    }
    if let Some(timestep) = control.timestep_ns {
        state.timestep = Duration::from_nanos(timestep);
    }
    if let Some(frequency) = control.frequency {
        state.frequency = frequency;
    }
    Ok(())
}

// send_if_modified applies the control atomically against concurrent cursor
// advances; an invalid control terminates the stream instead of being
// silently dropped.
fn try_control(control: &watch::Sender<Playback>, update: StreamControl) -> Result<(), Status> {
    let mut result = Ok(());
    control.send_if_modified(|state| {
        result = apply_control(state, update);
        result.is_ok()
    });
    result
}

async fn read_component_controls(
    mut incoming: tonic::Streaming<StreamComponentsRequest>,
    control: watch::Sender<Playback>,
    tx: mpsc::Sender<Result<StreamComponentsResponse, Status>>,
) {
    while let Ok(Some(request)) = incoming.message().await {
        let status = match request.request {
            Some(stream_components_request::Request::Control(update)) => {
                match try_control(&control, update) {
                    Ok(()) => continue,
                    Err(status) => status,
                }
            }
            _ => Status::invalid_argument("only StreamControl may follow StreamOpen"),
        };
        let _ = tx.send(Err(status)).await;
        return;
    }
}

async fn read_message_controls(
    mut incoming: tonic::Streaming<StreamMessagesRequest>,
    control: watch::Sender<Playback>,
    tx: mpsc::Sender<Result<StreamMessagesResponse, Status>>,
) {
    while let Ok(Some(request)) = incoming.message().await {
        let status = match request.request {
            Some(stream_messages_request::Request::Control(update)) => {
                match try_control(&control, update) {
                    Ok(()) => continue,
                    Err(status) => status,
                }
            }
            _ => Status::invalid_argument("only StreamControl may follow MessageStreamOpen"),
        };
        let _ = tx.send(Err(status)).await;
        return;
    }
}

async fn reject_component_controls(
    mut incoming: tonic::Streaming<StreamComponentsRequest>,
    tx: mpsc::Sender<Result<StreamComponentsResponse, Status>>,
) {
    if let Ok(Some(_)) = incoming.message().await {
        let _ = tx
            .send(Err(Status::failed_precondition(
                "stream controls require fixed-rate behavior",
            )))
            .await;
    }
}

async fn reject_message_controls(
    mut incoming: tonic::Streaming<StreamMessagesRequest>,
    tx: mpsc::Sender<Result<StreamMessagesResponse, Status>>,
) {
    if let Ok(Some(_)) = incoming.message().await {
        let _ = tx
            .send(Err(Status::failed_precondition(
                "stream controls require an independent fixed-rate clock",
            )))
            .await;
    }
}

// Delivers the latest existing message on subscribe, then drains every
// subsequent append by index so bursts and equal timestamps are not skipped.
async fn run_live_message(
    (name, log): (String, MsgLog),
    tx: mpsc::Sender<Result<StreamMessagesResponse, Status>>,
) {
    let waiter = log.waiter();
    let mut next = log.timestamps().len().saturating_sub(1);
    loop {
        // A truncated log restarts indexing from its new tail.
        next = next.min(log.timestamps().len());
        while let Some((timestamp, payload)) = log.get_index(next) {
            let response = StreamMessagesResponse {
                name: name.clone(),
                timestamp_ns: timestamp.0.saturating_mul(1000),
                payload: payload.to_vec(),
            };
            if tx.send(Ok(response)).await.is_err() {
                return;
            }
            next += 1;
        }
        tokio::select! {
            _ = waiter.wait() => {}
            _ = tx.closed() => return,
        }
    }
}

// Owns an independent message playback clock; the same publish-through-watch
// structure as run_fixed_components.
async fn run_fixed_messages(
    messages: Vec<(String, MsgLog)>,
    tx: mpsc::Sender<Result<StreamMessagesResponse, Status>>,
    control_tx: watch::Sender<Playback>,
) {
    let mut control = control_tx.subscribe();
    let mut sent = HashMap::<String, Timestamp>::new();
    let mut seek_generation = control.borrow().seek_generation;
    loop {
        let state = *control.borrow_and_update();
        if state.seek_generation != seek_generation {
            seek_generation = state.seek_generation;
            sent.clear();
        }
        if !send_message_frame(&messages, &tx, state.timestamp, &mut sent).await {
            return;
        }
        if !state.playing {
            tokio::select! {
                result = control.changed() => {
                    if result.is_err() {
                        return;
                    }
                }
                _ = tx.closed() => return,
            }
            continue;
        }
        let sleep = tokio::time::sleep(Duration::from_secs_f64(1.0 / state.frequency as f64));
        tokio::pin!(sleep);
        tokio::select! {
            _ = &mut sleep => {
                advance_cursor(&control_tx, &state);
            }
            result = control.changed() => {
                if result.is_err() {
                    return;
                }
            }
            _ = tx.closed() => return,
        }
    }
}

// Mirrors a clock owned by a component stream: emits whenever the owner
// publishes a new position and ends when the owning stream ends.
async fn follow_shared_messages(
    messages: Vec<(String, MsgLog)>,
    tx: mpsc::Sender<Result<StreamMessagesResponse, Status>>,
    mut control: watch::Receiver<Playback>,
) {
    let mut sent = HashMap::<String, Timestamp>::new();
    let mut seek_generation = control.borrow().seek_generation;
    loop {
        let state = *control.borrow_and_update();
        if state.seek_generation != seek_generation {
            seek_generation = state.seek_generation;
            sent.clear();
        }
        if !send_message_frame(&messages, &tx, state.timestamp, &mut sent).await {
            return;
        }
        tokio::select! {
            result = control.changed() => {
                if result.is_err() {
                    return;
                }
            }
            _ = tx.closed() => return,
        }
    }
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
        if timestamp > cursor || sent.get(name) == Some(&timestamp) {
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
    use impeller2_wkt::{ComponentMetadata, SetDbConfig};
    use tempfile::TempDir;
    use v1::stream_service_server::StreamServiceServer;

    use super::*;
    use crate::ComponentSchema;

    async fn transport_service() -> (
        TempDir,
        StreamServiceImpl,
        v1::stream_service_client::StreamServiceClient<tonic::transport::Channel>,
        tokio::task::JoinHandle<Result<(), tonic::transport::Error>>,
    ) {
        let directory = TempDir::new().unwrap();
        let db = Arc::new(DB::create(directory.path().join("db")).unwrap());
        let id = ComponentId::new("demo.signal");
        db.with_state_mut(|state| {
            state
                .insert_component(id, ComponentSchema::new(PrimType::F64, &[]), &db.path)
                .unwrap();
            state
                .set_component_metadata(
                    ComponentMetadata {
                        component_id: id,
                        name: "demo.signal".into(),
                        metadata: Default::default(),
                    },
                    &db.path,
                )
                .unwrap();
        });
        db.apply_component_row(
            Timestamp(100),
            &[(id, 1.0f64.to_le_bytes().to_vec())],
            false,
        )
        .unwrap();
        let service = StreamServiceImpl::new(db);
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let server_service = service.clone();
        let server = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(StreamServiceServer::new(server_service))
                .serve(addr)
                .await
        });
        let endpoint = format!("http://{addr}");
        let client = loop {
            match v1::stream_service_client::StreamServiceClient::connect(endpoint.clone()).await {
                Ok(client) => break client,
                Err(_) => tokio::time::sleep(Duration::from_millis(10)).await,
            }
        };
        (directory, service, client, server)
    }

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
        let (control_tx, _control_rx) = watch::channel(initial);
        let playbacks = Arc::new(Mutex::new(HashMap::from([(1, control_tx.clone())])));
        let task = tokio::spawn(run_fixed_components(
            vec![("demo.signal".into(), component)],
            tx,
            control_tx.clone(),
            PlaybackRegistration { id: 1, playbacks },
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

    #[tokio::test]
    async fn fixed_stream_holds_updates_until_cursor_reaches_data() {
        let directory = TempDir::new().unwrap();
        let component = component(&directory);
        let (tx, mut rx) = mpsc::channel(8);
        let initial = Playback {
            playing: false,
            timestamp: Timestamp(50),
            seek_generation: 0,
            timestep: Duration::from_micros(1),
            frequency: 100,
        };
        let (control_tx, _control_rx) = watch::channel(initial);
        let playbacks = Arc::new(Mutex::new(HashMap::from([(1, control_tx.clone())])));
        let task = tokio::spawn(run_fixed_components(
            vec![("demo.signal".into(), component)],
            tx,
            control_tx.clone(),
            PlaybackRegistration { id: 1, playbacks },
        ));

        // The cursor precedes every sample (data starts at 100 µs), so the
        // frame carries only the clock, never a future sample.
        let first = rx.recv().await.unwrap().unwrap();
        assert!(matches!(
            first.response,
            Some(stream_components_response::Response::Timestamp(
                v1::StreamTimestamp {
                    timestamp_ns: 50_000
                }
            ))
        ));

        let mut seek = initial;
        seek.timestamp = Timestamp(100);
        seek.seek_generation = 1;
        control_tx.send(seek).unwrap();
        let update = rx.recv().await.unwrap().unwrap();
        let Some(stream_components_response::Response::Update(update)) = update.response else {
            panic!("expected component update");
        };
        assert_eq!(update.timestamp_ns, 100_000);
        task.abort();
    }

    #[test]
    fn advance_yields_to_intervening_seek() {
        let initial = Playback {
            playing: true,
            timestamp: Timestamp(100),
            seek_generation: 0,
            timestep: Duration::from_micros(1),
            frequency: 100,
        };
        let (control_tx, _control_rx) = watch::channel(initial);

        // A seek that lands between frame emission and the elapsed sleep must
        // not be stepped over.
        let mut seek = initial;
        seek.timestamp = Timestamp(500);
        seek.seek_generation = 1;
        control_tx.send(seek).unwrap();
        advance_cursor(&control_tx, &initial);
        assert_eq!(control_tx.borrow().timestamp, Timestamp(500));

        // Without an intervening seek the cursor advances by one timestep.
        let current = *control_tx.borrow();
        advance_cursor(&control_tx, &current);
        assert_eq!(control_tx.borrow().timestamp, Timestamp(501));
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

        // Sub-microsecond timesteps would freeze the cursor at the database's
        // 1 µs resolution.
        let error = apply_control(
            &mut state,
            StreamControl {
                timestep_ns: Some(999),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);

        // Unbounded frequencies would turn the frame sleep into a busy loop,
        // and a rejected control must leave the state untouched even when it
        // combines valid fields with the invalid one.
        let snapshot = state;
        let error = apply_control(
            &mut state,
            StreamControl {
                playing: Some(true),
                seek_ns: Some(9_000),
                frequency: Some(MAX_FRAME_FREQUENCY + 1),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
        assert_eq!(state.playing, snapshot.playing);
        assert_eq!(state.timestamp, snapshot.timestamp);
        assert_eq!(state.seek_generation, snapshot.seek_generation);
        assert_eq!(state.frequency, snapshot.frequency);
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
        let (stream_id, _, mut component_clock, _registration) =
            service.register_playback(playback);
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
    async fn live_message_stream_drains_bursts_and_equal_timestamps() {
        let directory = TempDir::new().unwrap();
        let log = MsgLog::create(directory.path()).unwrap();
        log.push(Timestamp(10), b"first").unwrap();
        let (tx, mut rx) = mpsc::channel(8);
        let task = tokio::spawn(run_live_message(("demo.log".into(), log.clone()), tx));
        assert_eq!(rx.recv().await.unwrap().unwrap().payload, b"first");
        // Burst appended before the task wakes, including a duplicate timestamp.
        log.push(Timestamp(11), b"second").unwrap();
        log.push(Timestamp(11), b"third").unwrap();
        log.push(Timestamp(12), b"fourth").unwrap();
        for expected in [b"second".as_slice(), b"third", b"fourth"] {
            assert_eq!(rx.recv().await.unwrap().unwrap().payload, expected);
        }
        task.abort();
    }

    #[tokio::test]
    async fn shared_playback_attaches_at_current_cursor() {
        let directory = TempDir::new().unwrap();
        let component = component(&directory);
        let (frame_tx, frame_rx) = mpsc::channel(64);
        let initial = Playback {
            playing: true,
            timestamp: Timestamp(100),
            seek_generation: 0,
            timestep: Duration::from_micros(1),
            frequency: 500,
        };
        let (control_tx, mut watcher) = watch::channel(initial);
        let playbacks = Arc::new(Mutex::new(HashMap::from([(1, control_tx.clone())])));
        let owner = tokio::spawn(run_fixed_components(
            vec![("demo.signal".into(), component)],
            frame_tx,
            control_tx.clone(),
            PlaybackRegistration { id: 1, playbacks },
        ));
        // The owner publishes its advancing cursor into the shared clock.
        watcher
            .wait_for(|state| state.timestamp > Timestamp(100))
            .await
            .unwrap();

        let log = MsgLog::create(directory.path().join("log")).unwrap();
        log.push(Timestamp(101), b"attached").unwrap();
        let (message_tx, mut message_rx) = mpsc::channel(8);
        let follower = tokio::spawn(follow_shared_messages(
            vec![("demo.log".into(), log)],
            message_tx,
            control_tx.subscribe(),
        ));
        let first = message_rx.recv().await.unwrap().unwrap();
        // A late attacher starts at the owner's current position, not the
        // original open timestamp.
        assert!(first.timestamp_ns >= 101_000);
        drop(frame_rx);
        owner.await.unwrap();
        follower.abort();
    }

    #[tokio::test]
    async fn transport_releases_fixed_playback_on_disconnect() {
        let (_directory, service, mut client, server) = transport_service().await;
        for _ in 0..4 {
            let mut responses = client
                .stream_components(tokio_stream::iter([StreamComponentsRequest {
                    request: Some(stream_components_request::Request::Open(v1::StreamOpen {
                        components: vec!["demo.signal".into()],
                        behavior: Some(v1::stream_open::Behavior::FixedRate(v1::FixedRate {
                            initial: v1::InitialTimestamp::Manual as i32,
                            initial_timestamp_ns: 100_000,
                            timestep_ns: 1_000_000,
                            frequency: 100,
                        })),
                    })),
                }]))
                .await
                .unwrap()
                .into_inner();
            while !matches!(
                futures_lite::StreamExt::next(&mut responses)
                    .await
                    .unwrap()
                    .unwrap()
                    .response,
                Some(stream_components_response::Response::Opened(_))
            ) {}
            drop(responses);
            tokio::time::timeout(Duration::from_secs(1), async {
                loop {
                    if service.playbacks.lock().unwrap().is_empty() {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .unwrap();
        }
        server.abort();
    }

    #[tokio::test]
    async fn transport_requires_stream_open_first() {
        let (_directory, _service, mut client, server) = transport_service().await;
        let error = client
            .stream_components(tokio_stream::iter([StreamComponentsRequest {
                request: Some(stream_components_request::Request::Control(StreamControl {
                    playing: Some(false),
                    ..Default::default()
                })),
            }]))
            .await
            .unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
        server.abort();
    }

    #[tokio::test]
    async fn transport_rejects_controls_on_realtime_stream() {
        let (_directory, _service, mut client, server) = transport_service().await;
        let (tx, rx) = mpsc::channel(4);
        tx.send(StreamComponentsRequest {
            request: Some(stream_components_request::Request::Open(v1::StreamOpen {
                components: vec!["demo.signal".into()],
                behavior: Some(v1::stream_open::Behavior::RealTime(v1::RealTime {
                    immediate: false,
                })),
            })),
        })
        .await
        .unwrap();
        let mut responses = client
            .stream_components(ReceiverStream::new(rx))
            .await
            .unwrap()
            .into_inner();
        tx.send(StreamComponentsRequest {
            request: Some(stream_components_request::Request::Control(StreamControl {
                playing: Some(false),
                ..Default::default()
            })),
        })
        .await
        .unwrap();
        let error = tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if let Err(error) = responses.message().await {
                    break error;
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(error.code(), tonic::Code::FailedPrecondition);
        server.abort();
    }

    #[tokio::test]
    async fn shared_follower_emits_once_seek_reaches_message() {
        let directory = TempDir::new().unwrap();
        let log = MsgLog::create(directory.path()).unwrap();
        log.push(Timestamp(200), b"later").unwrap();
        let initial = Playback {
            playing: false,
            timestamp: Timestamp(100),
            seek_generation: 0,
            timestep: Duration::from_micros(1),
            frequency: 100,
        };
        let (control_tx, control_rx) = watch::channel(initial);
        let (tx, mut rx) = mpsc::channel(4);
        let task = tokio::spawn(follow_shared_messages(
            vec![("demo.log".into(), log)],
            tx,
            control_rx,
        ));
        // No message exists at or before the cursor, so nothing is emitted.
        assert!(
            tokio::time::timeout(Duration::from_millis(100), rx.recv())
                .await
                .is_err()
        );
        let mut seek = initial;
        seek.timestamp = Timestamp(200);
        seek.seek_generation = 1;
        control_tx.send(seek).unwrap();
        assert_eq!(rx.recv().await.unwrap().unwrap().timestamp_ns, 200_000);
        task.abort();
    }

    #[tokio::test]
    async fn transport_invalid_control_terminates_stream() {
        let (_directory, _service, mut client, server) = transport_service().await;
        let (tx, rx) = mpsc::channel(4);
        tx.send(StreamComponentsRequest {
            request: Some(stream_components_request::Request::Open(v1::StreamOpen {
                components: vec!["demo.signal".into()],
                behavior: Some(v1::stream_open::Behavior::FixedRate(v1::FixedRate {
                    initial: v1::InitialTimestamp::Manual as i32,
                    initial_timestamp_ns: 100_000,
                    timestep_ns: 1_000_000,
                    frequency: 100,
                })),
            })),
        })
        .await
        .unwrap();
        let mut responses = client
            .stream_components(ReceiverStream::new(rx))
            .await
            .unwrap()
            .into_inner();
        tx.send(StreamComponentsRequest {
            request: Some(stream_components_request::Request::Control(StreamControl {
                frequency: Some(0),
                ..Default::default()
            })),
        })
        .await
        .unwrap();
        let error = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if let Err(error) = responses.message().await {
                    break error;
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
        server.abort();
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
