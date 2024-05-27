use conduit::{ser_de::Frozen, ColumnPayload, ComponentId, Metadata};

pub use conduit;

pub trait Decomponentize {
    fn apply_column<B: AsRef<[u8]>>(&mut self, metadata: &Metadata, payload: &ColumnPayload<B>);
}

pub trait Componentize {
    fn sink_columns<Buf: Frozen>(&self, output: &mut impl ColumnSink<Buf>);
    fn get_metadata(&self, component_id: ComponentId) -> Option<&Metadata>;
}

pub trait ColumnSink<Buf: Frozen> {
    fn sink_column(&mut self, component_id: ComponentId, payload: ColumnPayload<Buf>);
}

pub trait Handler {
    type World: Default + Decomponentize + Componentize;
    fn tick(&mut self, world: &mut Self::World);
}

pub mod flume {
    const MAX_CONNECTIONS: usize = 16;
    use crate::{ColumnSink, Componentize, Decomponentize, Handler};
    use conduit::{
        bytes::Bytes,
        client::{Msg, MsgPair},
        ColumnPayload, ComponentId, ControlMsg, Packet, Payload, Query, StreamId,
    };
    use std::{
        collections::HashMap,
        time::{Duration, Instant},
    };
    use tracing::warn;

    impl ColumnSink<Bytes> for HashMap<ComponentId, ColumnPayload<Bytes>> {
        fn sink_column(&mut self, component_id: ComponentId, payload: ColumnPayload<Bytes>) {
            self.insert(component_id, payload);
        }
    }

    pub fn run<H: Handler>(mut handler: H, tick_time: Duration, rx: flume::Receiver<MsgPair>) {
        let mut world = H::World::default();
        let mut subscriptions = heapless::Vec::<
            (Query, StreamId, flume::Sender<Packet<Payload<Bytes>>>),
            { MAX_CONNECTIONS },
        >::new();
        let mut output = HashMap::with_capacity(8);
        loop {
            let instant = Instant::now();
            handler.tick(&mut world);
            world.sink_columns(&mut output);
            for (query, stream_id, tx) in &subscriptions {
                if let Some(column) = output.get(&query.component_id) {
                    if let Err(err) = tx.try_send(Packet::column(*stream_id, column.clone())) {
                        warn!(?err, "error sending packet");
                    }
                }
            }
            while let Ok(MsgPair { msg, tx }) = rx.try_recv() {
                match msg {
                    Msg::Control(ControlMsg::Subscribe { query }) => {
                        let tx = tx.upgrade().expect("sender is dead");
                        let stream_id = StreamId::rand();
                        if let Some(metadata) = world.get_metadata(query.component_id) {
                            if let Err(err) =
                                tx.try_send(Packet::start_stream(stream_id, metadata.clone()))
                            {
                                warn!(?err, "error sending packet");
                            }
                            if subscriptions.push((query, stream_id, tx)).is_err() {
                                warn!("ignoring subscription, too many subscriptions");
                            }
                        } else {
                            warn!("ignoring subscription, component not found");
                        }
                    }
                    Msg::Column(column) => {
                        world.apply_column(&column.metadata, &column.payload);
                    }
                    _ => {}
                }
            }
            let elapsed = instant.elapsed();
            let delay = tick_time.saturating_sub(elapsed);
            if delay > Duration::ZERO {
                std::thread::sleep(delay); // TODO: replace this with spin sleep
            } else {
                warn!("tick took too long: {:?}", elapsed);
            }
        }
    }
}

pub mod tcp {
    use std::{net::SocketAddr, time::Duration};

    use conduit::{
        client::Msg,
        server::{handle_socket, TcpServer},
        ControlMsg, Packet,
    };
    use tokio::{net::TcpStream, task::JoinSet};
    use tracing::warn;

    use crate::Handler;

    pub struct TcpBuilder<H: Handler> {
        addr: SocketAddr,
        tick_time: Duration,
        subscriptions: Vec<(conduit::Query, SocketAddr)>,
        outputs: Vec<(conduit::Query, SocketAddr)>,
        handler: H,
    }

    pub fn builder<H: Handler>(handler: H, tick_time: Duration, addr: SocketAddr) -> TcpBuilder<H> {
        TcpBuilder {
            addr,
            tick_time,
            handler,
            subscriptions: vec![],
            outputs: vec![],
        }
    }

    impl<H: Handler> TcpBuilder<H> {
        pub fn subscribe(mut self, query: conduit::Query, addr: SocketAddr) -> Self {
            self.subscriptions.push((query, addr));
            self
        }
        pub fn output(mut self, query: conduit::Query, addr: SocketAddr) -> Self {
            self.outputs.push((query, addr));
            self
        }

        pub fn run(self) {
            let Self {
                tick_time,
                handler,
                addr,
                subscriptions,
                outputs,
            } = self;
            let (tx, rx) = flume::unbounded();
            std::thread::spawn(move || {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async move {
                    let server_tx = tx.clone();
                    let mut set = JoinSet::new();
                    set.spawn(async move {
                        let server = TcpServer::bind(server_tx, addr).await.unwrap();
                        server.run().await
                    });
                    for (query, addr) in outputs {
                        let tx = tx.clone();
                        set.spawn(async move {
                            loop {
                                let Ok(socket) = TcpStream::connect(addr).await else {
                                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                                    continue;
                                };
                                let (rx_socket, tx_socket) = socket.into_split();

                                if let Err(err) = handle_socket(
                                    tx.clone(),
                                    tx_socket,
                                    rx_socket,
                                    std::iter::empty(),
                                    std::iter::once(Msg::Control(ControlMsg::Subscribe {
                                        query: query.clone(),
                                    })),
                                )
                                .await
                                {
                                    warn!(?err, "socket error");
                                }
                            }
                        });
                    }
                    for (query, addr) in subscriptions {
                        let tx = tx.clone();
                        set.spawn(async move {
                            loop {
                                let Ok(socket) = TcpStream::connect(addr).await else {
                                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                                    continue;
                                };
                                let (rx_socket, tx_socket) = socket.into_split();

                                if let Err(err) = handle_socket(
                                    tx.clone(),
                                    tx_socket,
                                    rx_socket,
                                    std::iter::once(Packet::subscribe(query.clone())),
                                    std::iter::empty(),
                                )
                                .await
                                {
                                    warn!(?err, "socket error");
                                }
                            }
                        });
                    }
                    set.join_next().await
                })
                .unwrap()
                .unwrap()
                .unwrap();
            });
            crate::flume::run(handler, tick_time, rx);
        }
    }
}
