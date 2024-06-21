use crate::{Compiled, Error, WorldExec};
use conduit::{
    client::{Msg, MsgPair},
    query::MetadataStore,
    Connection, ControlMsg, Packet, Payload, StreamId, SubscriptionManager,
};

pub struct ConduitExec {
    sub_manager: SubscriptionManager,
    connections: Vec<Connection>,
    rx: flume::Receiver<MsgPair>,
    exec: WorldExec<Compiled>,
    playing: bool,
    state: State,
}

impl ConduitExec {
    pub fn new(exec: WorldExec<Compiled>, rx: flume::Receiver<MsgPair>) -> Self {
        let mut metadata_store = MetadataStore::default();
        for (_, metadata) in exec.world.component_map.values() {
            metadata_store.push(metadata.clone());
        }
        Self {
            sub_manager: SubscriptionManager::new(metadata_store),
            connections: Vec::new(),
            rx,
            exec,
            playing: true,
            state: State::default(),
        }
    }

    pub fn time_step(&self) -> std::time::Duration {
        self.exec.world.time_step.0
    }

    pub fn run(&mut self) -> Result<(), Error> {
        if self.playing {
            match &mut self.state {
                State::Running => {
                    self.exec.run()?;
                }
                State::Replaying { index } => {
                    *index += 1;
                    if *index >= self.exec.tick() {
                        self.state = State::Running;
                    }
                }
            }
        }
        self.send();
        self.recv();
        Ok(())
    }

    pub fn send(&mut self) {
        let tick = match self.state {
            State::Running => self.exec.tick(),
            State::Replaying { index } => index,
        };
        // drop connections and subscriptions if the connection is closed
        self.connections.retain_mut(|con| {
            con.send(Packet {
                stream_id: StreamId::CONTROL,
                payload: Payload::ControlMsg(ControlMsg::Tick {
                    tick,
                    max_tick: self.exec.tick(),
                }),
            })
            .inspect_err(|err| {
                tracing::debug!(?err, "send tick error, dropping connection");
            })
            .is_ok()
        });
        let tick = match self.state {
            State::Running => self.exec.world.tick,
            State::Replaying { index } => index,
        };
        self.sub_manager.send(&self.exec.world, tick);
    }

    pub fn recv(&mut self) {
        while let Ok(pair) = self.rx.try_recv() {
            if let Err(err) = self.process_msg_pair(pair) {
                match err {
                    Error::ComponentNotFound => tracing::debug!("component not found"),
                    err => {
                        tracing::warn!(?err, "error processing msg pair");
                    }
                }
            }
        }
    }

    pub fn connections(&self) -> &[Connection] {
        &self.connections
    }

    pub fn add_connection(&mut self, conn: Connection) -> Result<(), Error> {
        let already_exits = self.connections.iter().any(|c| c.same_channel(&conn));
        if already_exits {
            tracing::debug!("connection already exists");
            return Ok(());
        }
        tracing::debug!("received connect, sending metadata");
        conn.send(Packet {
            stream_id: StreamId::CONTROL,
            payload: Payload::ControlMsg(ControlMsg::StartSim {
                metadata_store: self.sub_manager.metadata_store.clone(),
                time_step: self.time_step(),
                entity_ids: self.exec.world.entity_ids(),
            }),
        })?;
        self.connections.push(conn);
        Ok(())
    }

    fn process_msg_pair(&mut self, MsgPair { msg, tx }: MsgPair) -> Result<(), Error> {
        let Some(tx) = tx.and_then(|tx| tx.upgrade()) else {
            tracing::debug!("channel closed");
            return Ok(());
        };
        match msg {
            Msg::Control(ControlMsg::Connect) => self.add_connection(tx)?,
            Msg::Control(ControlMsg::Subscribe { query }) => {
                self.sub_manager.subscribe(query, tx)?;
            }
            Msg::Control(ControlMsg::SetPlaying(playing)) => self.playing = playing,
            Msg::Control(ControlMsg::Rewind(index)) => self.state = State::Replaying { index },
            Msg::Control(ControlMsg::Query { time_range, query }) => {
                self.sub_manager
                    .query(time_range, query, &self.exec.world, tx)?;
            }
            Msg::Control(_) => {}
            Msg::Column(new_col) => {
                // NOTE: the entity ids in `new_col` can be a subset of the ones in `col`,
                // but the order must be the same
                let mut col_ref = self
                    .exec
                    .world
                    .column_by_id_mut(new_col.metadata.component_id())
                    .ok_or(Error::ComponentNotFound)?;
                let mut col = col_ref.iter();
                let updates = new_col
                    .iter()
                    .filter_map(|res| {
                        let value = res
                            .inspect_err(|err| {
                                tracing::warn!(?err, "error processing column value")
                            })
                            .ok()?;
                        // `col` is only ever scanned once because the iterator state is preserved across calls to `position`
                        let offset = col.position(|(entity_id, _)| entity_id == value.entity_id)?;
                        Some((offset, value.value))
                    })
                    .collect::<Vec<_>>();
                drop(col);
                for (offset, value) in updates {
                    if let Err(err) = col_ref.update(offset, value) {
                        tracing::warn!(?err, "error processing column value");
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Default, Copy, Clone)]
enum State {
    #[default]
    Running,
    Replaying {
        index: u64,
    },
}

#[cfg(feature = "tokio")]
pub fn spawn_tcp_server(
    socket_addr: std::net::SocketAddr,
    exec: WorldExec,
    client: nox::Client,
    check_canceled: impl Fn() -> bool,
) -> Result<(), Error> {
    use std::time::{Duration, Instant};

    use conduit::server::TcpServer;

    let (tx, rx) = flume::unbounded();
    let exec = exec.compile(client)?;
    let mut conduit_exec = ConduitExec::new(exec, rx);
    let time_step = conduit_exec.time_step();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            let server = TcpServer::bind(tx, socket_addr).await.unwrap();
            server.run().await
        })
        .unwrap();
    });
    let mut start = Instant::now();
    loop {
        conduit_exec.run()?;
        if check_canceled() {
            break Ok(());
        }
        if time_step > Duration::ZERO {
            let sleep_time = time_step.saturating_sub(start.elapsed());
            std::thread::sleep(sleep_time);
            start += time_step;
        }
    }
}
