use crate::*;
use bevy::app::{Plugin, Update};
use impeller2::types::OwnedPacket;
use impeller2::types::{LenPacket, Msg, MsgBuf};
use impeller2_wkt::StreamId;
use miette::miette;
use miette::IntoDiagnostic;
use std::{net::SocketAddr, sync::Arc, time::Duration};
use stellarator::{buf::IoBuf, io::SplitExt, net::TcpStream, sync::Mutex, JoinHandle};
use thingbuf::mpsc;

pub struct TcpImpellerPlugin {
    addr: SocketAddr,
}

impl TcpImpellerPlugin {
    pub fn new(addr: SocketAddr) -> Self {
        Self { addr }
    }
}

impl Plugin for TcpImpellerPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        let addr = self.addr;
        let (mut incoming_packet_tx, incoming_packet_rx) = mpsc::with_recycle(512, FilledRecycle);
        let (outgoing_packet_tx, outgoing_packet_rx) = mpsc::channel::<Option<LenPacket>>(512);
        let outgoing_packet_rx = Arc::new(Mutex::new(outgoing_packet_rx));
        let stream_id = fastrand::u64(..);
        std::thread::spawn(move || {
            let res: Result<(), miette::Error> = stellarator::run(|| async move {
                loop {
                    if let Err(err) = tcp_connect(
                        addr,
                        outgoing_packet_rx.clone(),
                        &mut incoming_packet_tx,
                        stream_id,
                    )
                    .await
                    {
                        bevy::log::trace!(?err, "connection ended with error");
                        stellarator::sleep(Duration::from_millis(50)).await;
                    }
                }
            });
            if let Err(err) = res {
                bevy::log::error!(?err, "tcp plugin error");
            }
        });
        app.insert_resource(PacketTx(outgoing_packet_tx))
            .insert_resource(PacketRx(incoming_packet_rx))
            .insert_resource(CurrentStreamId(stream_id))
            .add_systems(Update, sink);
    }
}

async fn tcp_connect(
    addr: SocketAddr,
    outgoing_packet_rx: Arc<Mutex<mpsc::Receiver<Option<LenPacket>>>>,
    incoming_packet_tx: &mut mpsc::Sender<MaybeFilledPacket, FilledRecycle>,
    stream_id: StreamId,
) -> Result<(), miette::Error> {
    let stream = TcpStream::connect(addr).await.into_diagnostic()?;
    let (rx, tx) = stream.split();
    let tx = impeller2_stella::PacketSink::new(tx);
    let mut rx = impeller2_stella::PacketStream::new(rx);
    let incoming_packet_tx = incoming_packet_tx.clone();
    incoming_packet_tx
        .send(MaybeFilledPacket::Packet(OwnedPacket::Msg(MsgBuf {
            id: NewConnection::ID,
            buf: vec![0x0].try_slice(..).unwrap(),
        })))
        .await
        .map_err(|_| miette!("incoming_packet_tx closed"))?;

    for packet in new_connection_packets(stream_id) {
        tx.send(packet).await.0?;
    }
    let rx: JoinHandle<Result<(), miette::Error>> = stellarator::spawn(async move {
        loop {
            let mut send_ref = incoming_packet_tx.send_ref().await.into_diagnostic()?;
            let buf = send_ref.take_buf().expect("buffer already taken");
            let pkt = rx.next(buf).await.into_diagnostic()?;
            *send_ref = MaybeFilledPacket::Packet(pkt);
        }
    });
    let outgoing_packet_rx = outgoing_packet_rx.clone();
    let tx: JoinHandle<Result<(), miette::Error>> = stellarator::spawn(async move {
        let outgoing_packet_rx = outgoing_packet_rx.lock().await;
        while let Some(pkt) = outgoing_packet_rx.recv().await {
            let Some(pkt) = pkt else {
                continue;
            };
            tx.send(pkt).await.0.into_diagnostic()?;
        }
        Ok::<_, miette::Error>(())
    });
    futures_lite::future::race(
        async { rx.await.map_err(|_| miette!("join error")) },
        async { tx.await.map_err(|_| miette!("join error")) },
    )
    .await?
}
