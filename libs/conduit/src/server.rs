use std::io;

use bytes::{Bytes, BytesMut};
use tokio_util::codec::{FramedRead, FramedWrite, LengthDelimitedCodec};
use tracing::{info_span, Instrument};

use crate::{
    client::{AsyncClient, Msg, MsgPair},
    ControlMsg, Error, Packet, Payload,
};

pub struct TcpServer {
    tx: flume::Sender<MsgPair>,
    listener: tokio::net::TcpListener,
}

impl TcpServer {
    pub async fn bind(
        tx: flume::Sender<MsgPair>,
        addr: std::net::SocketAddr,
    ) -> Result<Self, Error> {
        tracing::info!(%addr, "listening");
        let listener = tokio::net::TcpListener::bind(addr).await?;
        Ok(Self { tx, listener })
    }

    pub async fn run(self) -> Result<(), Error> {
        loop {
            let (socket, addr) = self.listener.accept().await?;
            tracing::info!(%addr, "accepted connection");
            let (rx_socket, tx_socket) = socket.into_split();
            tokio::spawn(
                handle_socket(self.tx.clone(), tx_socket, rx_socket)
                    .instrument(info_span!("conn", %addr).or_current()),
            );
        }
    }
}

pub async fn handle_socket(
    incoming_tx: flume::Sender<MsgPair>,
    tx_socket: impl tokio::io::AsyncWrite + Unpin,
    rx_socket: impl tokio::io::AsyncRead + Unpin,
) -> Result<(), crate::Error> {
    handle_stream_sink(
        incoming_tx,
        FramedWrite::new(
            tokio::io::BufWriter::with_capacity(0x8000, tx_socket),
            LengthDelimitedCodec::new(),
        ),
        FramedRead::new(
            tokio::io::BufReader::with_capacity(0x8000, rx_socket),
            LengthDelimitedCodec::new(),
        ),
    )
    .await
}

pub async fn handle_stream_sink(
    incoming_tx: flume::Sender<MsgPair>,
    tx_socket: impl futures::Sink<Bytes, Error = io::Error> + Unpin,
    rx_socket: impl futures::stream::Stream<Item = Result<BytesMut, io::Error>> + Unpin,
) -> Result<(), crate::Error> {
    let (outgoing_tx, outgoing_rx) = flume::unbounded::<Packet<Payload<Bytes>>>();

    let init_msg = Msg::<Bytes>::Control(ControlMsg::Connect);
    let init_msg_pair = MsgPair {
        msg: init_msg,
        tx: outgoing_tx.downgrade(),
    };
    incoming_tx.send_async(init_msg_pair).await.unwrap();

    let rx = async move {
        let mut rx_client = AsyncClient::new(rx_socket);
        loop {
            let msg = match rx_client.recv().await {
                Ok(m) => m,
                Err(Error::EOF) => {
                    tracing::warn!("received EOF");
                    continue;
                }
                Err(err) => return Err(err),
            };
            incoming_tx
                .send_async(MsgPair {
                    msg,
                    tx: outgoing_tx.downgrade(),
                })
                .await
                .map_err(|_| Error::EOF)?;
        }
    };
    let tx = async move {
        let mut tx_client = AsyncClient::new(tx_socket);
        while let Ok(packet) = outgoing_rx.recv_async().await {
            tx_client.send(packet).await?;
        }
        Ok::<(), Error>(())
    };
    let res = tokio::select! {
        res = tx.instrument(info_span!("tx").or_current()) => res,
        res = rx.instrument(info_span!("rx").or_current()) => res,
    };
    match &res {
        Ok(()) => tracing::debug!("terminating socket"),
        Err(err) => tracing::debug!(?err, "terminating socket"),
    }
    res
}
