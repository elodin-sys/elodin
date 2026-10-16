use futures_concurrency::future::Race;
use impeller::types::{LenPacket, Msg, Packet, PacketTy};
use impeller_bbq::*;
use impeller_wkt::{DbConfig, NewConnection, StreamId};
use miette::{IntoDiagnostic, miette};
use std::net::SocketAddr;
use stellarator::{
    io::{LengthDelReader, SplitExt},
    net::TcpStream,
};
use thingbuf::mpsc;
use zerocopy::TryFromBytes;

/// Builds the packets that open a connection's live stream from that
/// connection's `DbConfig` reply.
pub type StreamHandshake = fn(StreamId, &DbConfig) -> Vec<LenPacket>;

/// With a `stream_handshake`, incoming packets are forwarded until the first
/// `DbConfig`, then the handshake is written directly on this socket before
/// any queued packet, so it can never be replayed onto a later connection.
pub async fn tcp_connect<I>(
    addr: SocketAddr,
    outgoing_packet_rx: &mut mpsc::Receiver<Option<LenPacket>>,
    incoming_packet_tx: &mut AsyncArcQueueTx,
    stream_id: StreamId,
    new_connection_packets: &impl Fn(StreamId) -> I,
    stream_handshake: Option<StreamHandshake>,
    success: impl FnOnce(),
) -> Result<(), miette::Error>
where
    I: Iterator<Item = LenPacket>,
{
    let stream = TcpStream::connect(addr).await.into_diagnostic()?;
    let (rx, tx) = stream.split();
    let tx = crate::PacketSink::new(tx);
    let mut rx = LengthDelReader::<_, u32>::new(rx);

    let len_pkt = LenPacket::new(impeller::types::PacketTy::Msg, NewConnection::ID, 0);
    let grant = PacketGrantW::new(
        incoming_packet_tx
            .grant(128)
            .map_err(|err| miette!("channel error {err:?}"))?,
    );
    grant.commit_len_pkt(len_pkt);

    for packet in new_connection_packets(stream_id) {
        tx.send(packet).await.0?;
    }
    if let Some(stream_handshake) = stream_handshake {
        let config = loop {
            let grant_r = incoming_packet_tx.wait_grant(64 * 1024 * 1024).await;
            let grant_r = PacketGrantW::new(grant_r);
            let slice = rx.recv(grant_r).await.into_diagnostic()?;
            let config = parse_db_config(&slice).transpose()?;
            let len = slice.range().len();
            slice.into_inner().commit(len + 4);
            if let Some(config) = config {
                break config;
            }
        };
        for packet in stream_handshake(stream_id, &config) {
            tx.send(packet).await.0?;
        }
    }
    success();
    let rx = async move {
        loop {
            let grant_r = incoming_packet_tx.wait_grant(64 * 1024 * 1024).await;
            let grant_r = PacketGrantW::new(grant_r);
            let slice = rx.recv(grant_r).await.into_diagnostic()?;
            let len = slice.range().len();
            slice.into_inner().commit(len + 4);
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

fn parse_db_config(packet: &[u8]) -> Option<Result<DbConfig, miette::Error>> {
    let packet = Packet::try_ref_from_bytes(packet).ok()?;
    if packet.header.packet_ty != PacketTy::Msg || packet.header.id != DbConfig::ID {
        return None;
    }
    Some(postcard::from_bytes(&packet.body).into_diagnostic())
}

#[cfg(test)]
mod tests {
    use super::*;
    use impeller::types::IntoLenPacket;
    use impeller_wkt::GetDbSettings;
    use zerocopy::IntoBytes;

    fn packet_bytes(packet: &LenPacket) -> Vec<u8> {
        packet.as_packet().as_bytes().to_vec()
    }

    #[test]
    fn db_config_is_recognized_and_decoded() {
        let mut config = DbConfig::default();
        config.advertise_exact_stream_filter();
        let parsed = parse_db_config(&packet_bytes(&config.into_len_packet()))
            .expect("db config packet")
            .expect("valid db config");
        assert!(parsed.supports_exact_stream_filter());
    }

    #[test]
    fn other_messages_are_ignored() {
        assert!(parse_db_config(&packet_bytes(&GetDbSettings.into_len_packet())).is_none());
    }
}
