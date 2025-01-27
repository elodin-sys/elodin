use futures_concurrency::future::Race;
use impeller2::types::{FilledRecycle, LenPacket, MaybeFilledPacket, Msg, MsgBuf, OwnedPacket};
use impeller2_wkt::NewConnection;
use impeller2_wkt::StreamId;
use miette::miette;
use miette::IntoDiagnostic;
use std::net::SocketAddr;
use stellarator::buf::IoBuf;
use stellarator::io::SplitExt;
use stellarator::net::TcpStream;
use thingbuf::mpsc;

pub async fn tcp_connect<I>(
    addr: SocketAddr,
    outgoing_packet_rx: &mut mpsc::Receiver<Option<LenPacket>>,
    incoming_packet_tx: &mut mpsc::Sender<MaybeFilledPacket, FilledRecycle>,
    stream_id: StreamId,
    new_connection_packets: &impl Fn(StreamId) -> I,
) -> Result<(), miette::Error>
where
    I: Iterator<Item = LenPacket>,
{
    let stream = TcpStream::connect(addr).await.into_diagnostic()?;
    let (rx, tx) = stream.split();
    let tx = crate::PacketSink::new(tx);
    let mut rx = crate::PacketStream::new(rx);
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
    let rx = async move {
        loop {
            let mut send_ref = incoming_packet_tx.send_ref().await.into_diagnostic()?;
            let buf = send_ref.take_buf().expect("buffer already taken");
            let pkt = rx.next(buf).await.into_diagnostic()?;
            *send_ref = MaybeFilledPacket::Packet(pkt);
        }
    };
    let tx = async move {
        while let Some(pkt) = outgoing_packet_rx.recv().await {
            let Some(pkt) = pkt else {
                continue;
            };
            tx.send(pkt).await.0.into_diagnostic()?;
        }
        Ok::<_, miette::Error>(())
    };
    (rx, tx).race().await
}
