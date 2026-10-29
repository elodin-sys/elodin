//! Exercise the public connection API as an external crate, without listeners.

use std::{
    future::Future,
    sync::{Arc, Weak},
    time::Duration,
};

use elodin_db::{DB, Error, serve_connection};
use impeller::{
    types::{
        ComponentId, IntoLenPacket, LenPacket, OwnedPacket, PrimType, Request, Timestamp,
        TryFromPacket,
    },
    vtable::builder::{component, raw_field, raw_table, schema, timestamp, vtable},
};
use impeller_stellar::{PacketSink, PacketStream};
use impeller_wkt::*;
use stellarator::{
    BufResult,
    buf::{IoBuf, IoBufMut, Slice},
    io::{AsyncRead, AsyncWrite, OwnedReader, OwnedWriter, SplitExt},
    sync::Mutex,
    util::{AtomicCell, CancelToken},
};
use tokio::io::{AsyncReadExt, AsyncWriteExt, DuplexStream, ReadHalf, WriteHalf};

struct Probe {
    reading: AtomicCell<usize>,
    writing: AtomicCell<usize>,
}

impl Default for Probe {
    fn default() -> Self {
        Self {
            reading: AtomicCell::new(0),
            writing: AtomicCell::new(0),
        }
    }
}

struct IoGuard<'a>(&'a AtomicCell<usize>);

impl Drop for IoGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

struct MemoryIo {
    rx: Mutex<ReadHalf<DuplexStream>>,
    tx: Mutex<WriteHalf<DuplexStream>>,
    probe: Arc<Probe>,
    _life: Arc<()>,
}

impl MemoryIo {
    fn new(stream: DuplexStream) -> Self {
        let (rx, tx) = tokio::io::split(stream);
        Self {
            rx: Mutex::new(rx),
            tx: Mutex::new(tx),
            probe: Arc::new(Probe::default()),
            _life: Arc::new(()),
        }
    }
}

impl AsyncRead for MemoryIo {
    async fn read<B: IoBufMut>(&self, mut buf: B) -> BufResult<usize, B> {
        self.probe
            .reading
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let _guard = IoGuard(&self.probe.reading);
        let result = self
            .rx
            .lock()
            .await
            .read(stellarator::buf::deref_mut(&mut buf))
            .await;
        (result.map_err(Into::into), buf)
    }
}

impl AsyncWrite for MemoryIo {
    async fn write<B: IoBuf>(&self, buf: B) -> BufResult<usize, B> {
        self.probe
            .writing
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let _guard = IoGuard(&self.probe.writing);
        let result = self
            .tx
            .lock()
            .await
            .write(stellarator::buf::deref(&buf))
            .await;
        (result.map_err(Into::into), buf)
    }
}

struct Client {
    tx: PacketSink<OwnedWriter<MemoryIo>>,
    rx: PacketStream<OwnedReader<MemoryIo>>,
}

impl Client {
    async fn send(&self, packet: impl IntoLenPacket) {
        bounded(self.tx.send(packet)).await.0.unwrap();
    }

    async fn recv(&mut self) -> OwnedPacket<Slice<Vec<u8>>> {
        bounded(self.rx.next_grow(vec![0; 256])).await.unwrap()
    }

    async fn request<R: Request + IntoLenPacket>(
        &mut self,
        request: R,
    ) -> R::Reply<Slice<Vec<u8>>> {
        self.send(request.into_len_packet().with_request_id(42))
            .await;
        loop {
            let packet = self.recv().await;
            if packet.req_id() == 42 {
                return R::Reply::try_from_packet(&packet).unwrap();
            }
        }
    }
}

struct Connection {
    cancel: CancelToken,
    done: stellarator::JoinHandle<Result<(), Error>>,
    probe: Arc<Probe>,
    life: Weak<()>,
}

impl Connection {
    async fn stop(self) {
        self.cancel.cancel();
        assert!(matches!(
            bounded(self.done).await.unwrap(),
            Err(Error::Cancelled)
        ));
        assert!(
            self.life.upgrade().is_none(),
            "connection retained its transport"
        );
        assert_eq!(self.probe.reading.latest(), 0);
        assert_eq!(self.probe.writing.latest(), 0);
    }
}

fn connect(db: &Arc<DB>, capacity: usize) -> (Client, Connection) {
    let (server, client) = tokio::io::duplex(capacity);
    let server = MemoryIo::new(server);
    let life = Arc::downgrade(&server._life);
    let probe = server.probe.clone();
    let cancel = CancelToken::new();
    let done = stellarator::spawn(serve_connection(server, db.clone(), cancel.clone()));
    let (rx, tx) = MemoryIo::new(client).split();
    (
        Client {
            tx: PacketSink::new(tx),
            rx: PacketStream::new(rx),
        },
        Connection {
            cancel,
            done,
            probe,
            life,
        },
    )
}

async fn bounded<T>(future: impl Future<Output = T>) -> T {
    futures_lite::future::race(future, async {
        stellarator::sleep(Duration::from_secs(10)).await;
        panic!("connection operation timed out");
    })
    .await
}

fn database() -> (tempfile::TempDir, Arc<DB>) {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(DB::create(dir.path().join("db")).unwrap());
    (dir, db)
}

async fn seed(client: &mut Client) -> ComponentId {
    let id = ComponentId::new("embedded.value");
    client
        .send(&SetComponentMetadata::new(id, "embedded.value"))
        .await;
    client
        .send(&VTableMsg {
            id: [1, 0],
            vtable: vtable([raw_field(
                0,
                8,
                timestamp(raw_table(8, 8), schema(PrimType::U64, &[], component(id))),
            )]),
        })
        .await;
    let mut packet = LenPacket::table([1, 0], 16);
    packet.extend_from_slice(&123_u64.to_le_bytes());
    packet.extend_aligned(&[10_i64]);
    client.send(packet).await;
    // This request is a barrier after the writes on the same connection.
    client.request(&GetEarliestTimestamp).await;
    id
}

#[stellarator::test]
async fn memory_connection_reads_real_telemetry() {
    let (_dir, db) = database();
    let (mut client, connection) = connect(&db, 1024);
    let id = seed(&mut client).await;
    let response = client
        .request(&GetTimeSeries {
            id: [1, 0],
            component_id: id,
            range: Timestamp(0)..Timestamp(20),
            limit: None,
        })
        .await;
    assert_eq!(response.timestamps().unwrap(), &[Timestamp(10)]);
    assert_eq!(response.data().unwrap(), &123_u64.to_le_bytes());
    connection.stop().await;
    assert_eq!(Arc::strong_count(&db), 1);
}

#[stellarator::test]
async fn cancellation_before_first_poll_releases_resources() {
    let (_dir, db) = database();
    let (_client, connection) = connect(&db, 1024);
    connection.stop().await;
    assert_eq!(Arc::strong_count(&db), 1);
}

#[stellarator::test]
async fn cancelling_a_read_leaves_another_connection_usable() {
    let (_dir, db) = database();
    let (_client, first) = connect(&db, 1024);
    let (mut client, second) = connect(&db, 1024);
    bounded(first.probe.reading.wait_for(|count| count > 0)).await;
    first.stop().await;
    client.request(&GetEarliestTimestamp).await;
    second.stop().await;
    assert_eq!(Arc::strong_count(&db), 1);
}

#[stellarator::test]
async fn cancelling_a_blocked_write_releases_the_connection() {
    let (_dir, db) = database();
    let (client, connection) = connect(&db, 1);
    let (mut other, second) = connect(&db, 1024);
    client.send(&GetDbSettings).await;
    bounded(connection.probe.writing.wait_for(|count| count > 0)).await;
    connection.stop().await;
    other.request(&GetEarliestTimestamp).await;
    second.stop().await;
    assert_eq!(Arc::strong_count(&db), 1);
}

#[stellarator::test]
async fn cancelling_subscriptions_releases_nested_work() {
    let (_dir, db) = database();
    let (mut client, connection) = connect(&db, 65536);
    let (mut other, second) = connect(&db, 1024);
    seed(&mut client).await;
    client.send(&SubscribeLastUpdated).await;
    client.send(&MsgStream { msg_id: [70, 0] }).await;
    client
        .send(&FixedRateMsgStream {
            msg_id: [72, 0],
            fixed_rate: FixedRateOp {
                stream_id: 3,
                behavior: FixedRateBehavior::default(),
            },
        })
        .await;
    client.send(&TimestampedMsgStream { msg_id: [71, 0] }).await;
    client
        .send(&Stream {
            id: 1,
            behavior: StreamBehavior::RealTime,
        })
        .await;
    client
        .send(&Stream {
            id: 2,
            behavior: StreamBehavior::RealTimeBatched,
        })
        .await;
    client
        .send(&Stream {
            id: 3,
            behavior: StreamBehavior::FixedRate(FixedRateBehavior::default()),
        })
        .await;
    client.send(&VTableStream { id: [1, 0] }).await;
    client.request(&GetEarliestTimestamp).await;
    connection.stop().await;
    other.request(&GetEarliestTimestamp).await;
    second.stop().await;
    assert_eq!(Arc::strong_count(&db), 1, "subscription retained the DB");
}

#[stellarator::test]
async fn cancelling_a_query_releases_its_worker_and_other_connections_survive() {
    let (_dir, db) = database();
    let (mut client, connection) = connect(&db, 32);
    let (mut other, second) = connect(&db, 1024);
    // The first Arrow batch fills the bounded transport. The SQL worker then
    // fills its bounded batch queue, so cancellation must also stop that worker.
    client
        .send(&SQLQuery("SELECT * FROM generate_series(1, 100000)".into()))
        .await;
    let OwnedPacket::Msg(first_batch) = client.recv().await else {
        panic!("expected an Arrow batch");
    };
    assert_eq!(first_batch.id, <ArrowIPC<'_> as impeller::types::Msg>::ID);
    bounded(connection.probe.writing.wait_for(|count| count > 0)).await;
    connection.stop().await;
    other.request(&GetEarliestTimestamp).await;
    second.stop().await;
    assert_eq!(Arc::strong_count(&db), 1, "query worker retained the DB");
}

#[stellarator::test]
async fn framing_errors_return_and_release_subscriptions() {
    let (_dir, db) = database();
    let (client, connection) = connect(&db, 1024);
    client.send(&SubscribeLastUpdated).await;
    // Reject the declared length without allocating or reading the payload.
    client
        .tx
        .writer()
        .write_all((256_u32 * 1024 * 1024 + 1).to_le_bytes().to_vec())
        .await
        .0
        .unwrap();
    let error = bounded(connection.done).await.unwrap().unwrap_err();
    assert!(matches!(
        error,
        Error::Stellar(stellarator::Error::PacketTooLarge { .. })
    ));
    assert!(connection.life.upgrade().is_none());
    assert_eq!(Arc::strong_count(&db), 1);
}

#[stellarator::test]
async fn peer_eof_releases_subscriptions() {
    let (_dir, db) = database();
    let (mut client, connection) = connect(&db, 1024);
    client.send(&SubscribeLastUpdated).await;
    client.request(&GetEarliestTimestamp).await;
    drop(client);
    assert!(
        bounded(connection.done)
            .await
            .unwrap()
            .unwrap_err()
            .is_stream_closed()
    );
    assert!(connection.life.upgrade().is_none());
    assert_eq!(Arc::strong_count(&db), 1);
}
