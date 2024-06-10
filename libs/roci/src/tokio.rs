use std::{net::SocketAddr, time::Duration};

use conduit::{
    bytes::Bytes,
    client::{Demux, Msg, MsgPair},
    server::{handle_socket, TcpServer},
    ControlMsg, Packet, Payload,
};
use tokio::{net::TcpStream, task::JoinSet};
use tracing::warn;

use crate::Handler;

pub struct TokioBuilder<H: Handler> {
    addr: SocketAddr,
    tick_time: Duration,
    subscriptions: Vec<(conduit::Query, Input)>,
    outputs: Vec<(conduit::Query, Output)>,
    handler: H,
}

pub fn builder<H: Handler>(handler: H, tick_time: Duration, addr: SocketAddr) -> TokioBuilder<H> {
    TokioBuilder {
        addr,
        tick_time,
        handler,
        subscriptions: vec![],
        outputs: vec![],
    }
}

pub enum Output {
    Tcp(SocketAddr),
    Channel(flume::Sender<Msg<Bytes>>),
}

pub enum Input {
    Tcp(SocketAddr),
    Channel(flume::Sender<MsgPair>),
}

impl From<flume::Sender<MsgPair>> for Input {
    fn from(v: flume::Sender<MsgPair>) -> Self {
        Self::Channel(v)
    }
}

impl From<SocketAddr> for Input {
    fn from(v: SocketAddr) -> Self {
        Self::Tcp(v)
    }
}

impl From<SocketAddr> for Output {
    fn from(v: SocketAddr) -> Self {
        Self::Tcp(v)
    }
}

impl From<flume::Sender<Msg<Bytes>>> for Output {
    fn from(v: flume::Sender<Msg<Bytes>>) -> Self {
        Self::Channel(v)
    }
}

impl<H: Handler + Send + 'static> TokioBuilder<H> {
    pub fn subscribe(mut self, query: conduit::Query, input: impl Into<Input>) -> Self {
        self.subscriptions.push((query, input.into()));
        self
    }

    pub fn tcp_output(mut self, query: conduit::Query, addr: SocketAddr) -> Self {
        self.outputs.push((query, Output::Tcp(addr)));
        self
    }

    pub fn output(mut self, query: conduit::Query, output: impl Into<Output>) -> Self {
        let output = output.into();
        self.outputs.push((query, output));
        self
    }

    pub fn run(self) -> (std::thread::JoinHandle<()>, flume::Sender<MsgPair>) {
        let Self {
            tick_time,
            handler,
            addr,
            subscriptions,
            outputs,
        } = self;
        let (tx, rx) = flume::unbounded();
        let server_tx = tx.clone();
        std::thread::spawn(move || {
            let tx = server_tx;
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async move {
                let server_tx = tx.clone();
                let mut set = JoinSet::new();
                set.spawn(async move {
                    let server = TcpServer::bind(server_tx, addr).await.unwrap();
                    server.run().await
                });
                for (query, output) in outputs {
                    let tx = tx.clone();
                    match output {
                        Output::Tcp(addr) => {
                            set.spawn(async move {
                                loop {
                                    let Ok(socket) = TcpStream::connect(addr).await else {
                                        tokio::time::sleep(std::time::Duration::from_millis(100))
                                            .await;
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
                        Output::Channel(channel) => {
                            set.spawn(async move {
                                let (outgoing_tx, outgoing_rx) =
                                    flume::unbounded::<Packet<Payload<Bytes>>>();
                                tx.send(MsgPair {
                                    msg: Msg::Control(ControlMsg::Subscribe {
                                        query: query.clone(),
                                    }),
                                    tx: Some(outgoing_tx.downgrade()),
                                })
                                .expect("socket closed");
                                let mut demux = Demux::default();
                                while let Ok(msg) = outgoing_rx.recv_async().await {
                                    match demux.handle(msg) {
                                        Ok(msg) => {
                                            if let Err(err) = channel.send(msg) {
                                                warn!(?err, "channel error");
                                            }
                                        }
                                        Err(err) => {
                                            warn!(?err, "demux error")
                                        }
                                    }
                                }

                                Ok(())
                            });
                        }
                    }
                }
                for (query, input) in subscriptions {
                    match input {
                        Input::Tcp(addr) => {
                            let tx = tx.clone();
                            set.spawn(async move {
                                loop {
                                    let Ok(socket) = TcpStream::connect(addr).await else {
                                        tokio::time::sleep(std::time::Duration::from_millis(100))
                                            .await;
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
                        Input::Channel(input_tx) => {
                            let tx = tx.clone();
                            set.spawn(async move {
                                let (outgoing_tx, outgoing_rx) =
                                    flume::unbounded::<Packet<Payload<Bytes>>>();
                                if input_tx
                                    .send(MsgPair {
                                        msg: Msg::Control(ControlMsg::Subscribe { query }),
                                        tx: Some(outgoing_tx.downgrade()),
                                    })
                                    .is_err()
                                {
                                    warn!("channel closed");
                                }
                                let mut demux = Demux::default();
                                while let Ok(msg) = outgoing_rx.recv_async().await {
                                    match demux.handle(msg) {
                                        Ok(msg) => {
                                            if tx.send(MsgPair { msg, tx: None }).is_err() {
                                                warn!("channel error");
                                            }
                                        }
                                        Err(err) => {
                                            warn!(?err, "demux error")
                                        }
                                    }
                                }

                                Ok(())
                            });
                        }
                    }
                }
                set.join_next().await
            })
            .unwrap()
            .unwrap()
            .unwrap();
        });
        let handle = std::thread::spawn(move || {
            crate::flume::run(handler, tick_time, rx);
        });
        (handle, tx)
    }
}
