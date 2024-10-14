use std::{
    collections::{HashMap, HashSet},
    net::SocketAddr,
};

use impeller::{
    bytes::{Bytes, BytesMut},
    client::{AsyncClient, Demux, Msg, TcpWriter},
    ser_de::ColumnValue,
    ColumnPayload, ComponentId, ComponentValue, ComponentValueDim, ControlMsg, EntityId, Error,
    Metadata, Packet, Payload, Query, StreamId,
};
use tokio::{
    net::{tcp::OwnedReadHalf, TcpListener, TcpStream},
    runtime::Runtime,
};
use tokio_stream::StreamExt;
use tokio_util::codec::{FramedRead, FramedWrite, LengthDelimitedCodec};
use tracing::warn;

use crate::{drivers::DriverMode, Componentize, Decomponentize, System};

#[derive(Hash, Debug, Eq, PartialEq, Clone, Copy)]
struct ConnectionId(u64);

struct Subscription {
    query: Query,
    stream_id: StreamId,
}

type Subscriptions = HashMap<ConnectionId, Vec<Subscription>>;
pub type MsgTriplet = (ComponentId, EntityId, ComponentValue<'static>);
pub type RxChannel = thingbuf::mpsc::Receiver<Option<MsgTriplet>>;
pub type TxChannel = thingbuf::mpsc::Sender<Option<MsgTriplet>>;

pub struct Server {
    demux: Demux,
    addr: SocketAddr,
    subscriptions: Subscriptions,
    write_halves: HashMap<ConnectionId, TcpWriter>,
    stream_map:
        tokio_stream::StreamMap<ConnectionId, FramedRead<OwnedReadHalf, LengthDelimitedCodec>>,
    outgoing_tx: TxChannel,
    incoming_rx: RxChannel,
    outgoing_filter: HashSet<(Option<ComponentId>, Option<EntityId>)>,
    metadata: HashMap<ComponentId, Metadata>,
}

enum Event {
    Connected(Result<(TcpStream, SocketAddr), std::io::Error>),
    Msg(ConnectionId, Result<BytesMut, std::io::Error>),
    Outgoing(Option<MsgTriplet>),
}

pub fn tcp_listen<D: DriverMode>(
    addr: SocketAddr,
    filters: &[Query],
    metadata: impl IntoIterator<Item = Metadata>,
) -> (Tx<D>, Rx<D>) {
    let (server, tx, rx) = Server::with_capacity(addr, filters, 1024, metadata);
    std::thread::spawn(move || {
        let rt = Runtime::new().unwrap();
        rt.block_on(server.run())
    });
    (
        Tx {
            channel: Some(tx),
            _phantom: std::marker::PhantomData,
        },
        Rx {
            channel: Some(rx),
            _phantom: std::marker::PhantomData,
        },
    )
}

impl Server {
    pub fn with_capacity(
        addr: SocketAddr,
        queries: &[Query],
        capacity: usize,
        metadata: impl IntoIterator<Item = Metadata>,
    ) -> (Self, TxChannel, RxChannel) {
        let (outgoing_tx, outgoing_rx) = thingbuf::mpsc::channel(capacity);
        let (incoming_tx, incoming_rx) = thingbuf::mpsc::channel(capacity);
        let mut outgoing_filter = HashSet::new();
        for q in queries {
            if q.entity_ids.is_empty() {
                outgoing_filter.insert((Some(q.component_id), None));
            } else {
                for entity_id in q.entity_ids.iter() {
                    outgoing_filter.insert((Some(q.component_id), Some(*entity_id)));
                }
            }
        }
        let metadata = metadata
            .into_iter()
            .map(|m| (m.component_id(), m))
            .collect();
        (
            Self {
                addr,
                outgoing_tx,
                incoming_rx,
                demux: Demux::default(),
                subscriptions: Default::default(),
                write_halves: Default::default(),
                stream_map: Default::default(),
                outgoing_filter,
                metadata,
            },
            incoming_tx,
            outgoing_rx,
        )
    }

    pub async fn run(mut self) -> Result<(), Error> {
        let listener = TcpListener::bind(self.addr).await?;
        loop {
            let event = tokio::select! {
                res = listener.accept() => Event::Connected(res),
                Some((id, msg)) = self.stream_map.next() => Event::Msg(id, msg),
                Some(msg) = self.incoming_rx.recv() => Event::Outgoing(msg)
            };
            match event {
                Event::Connected(Ok((stream, _))) => {
                    let id = ConnectionId(fastrand::u64(..));
                    let (rx, tx) = stream.into_split();
                    self.subscriptions.insert(id, vec![]);
                    self.write_halves.insert(
                        id,
                        AsyncClient::new(FramedWrite::new(tx, LengthDelimitedCodec::new())),
                    );
                    self.stream_map
                        .insert(id, FramedRead::new(rx, LengthDelimitedCodec::new()));
                }
                Event::Connected(Err(err)) => {
                    warn!(?err, "connection error");
                }
                Event::Msg(id, Ok(bytes)) => {
                    self.process_msg(id, bytes).await?;
                }
                Event::Msg(_, Err(err)) => {
                    warn!(?err, "read error");
                }
                Event::Outgoing(mut triple) => {
                    while let Some((component_id, entity_id, value)) = &triple {
                        for (connection_id, subs) in &self.subscriptions {
                            for sub in subs {
                                if sub.query.matches(*component_id, *entity_id) {
                                    if let Some(stream) = self.write_halves.get_mut(connection_id) {
                                        let packet: Packet<Payload<Bytes>> = Packet::column(
                                            sub.stream_id,
                                            ColumnPayload::try_from_value_iter(
                                                0,
                                                [ColumnValue {
                                                    entity_id: *entity_id,
                                                    value: value.clone(),
                                                }]
                                                .into_iter(),
                                            )
                                            .unwrap(),
                                        );
                                        stream.send(packet).await?;
                                    }
                                }
                            }
                        }
                        triple = self.incoming_rx.try_recv().ok().and_then(|msg| msg);
                    }
                }
            }
        }
    }

    async fn process_msg(&mut self, id: ConnectionId, buf: BytesMut) -> Result<(), Error> {
        let buf = buf.freeze();
        let packet = Packet::parse(buf)?;
        let msg = self.demux.handle(packet)?;
        match msg {
            Msg::Control(ControlMsg::Subscribe { query }) => {
                let Some(tx) = self.write_halves.get_mut(&id) else {
                    return Ok(());
                };
                let stream_id = StreamId::rand();
                if let Some(metadata) = self.metadata.get(&query.component_id) {
                    let packet: Packet<Payload<Bytes>> =
                        Packet::start_stream(stream_id, metadata.clone());
                    tx.send(packet).await?;
                    let subs = self.subscriptions.entry(id).or_default();
                    subs.push(Subscription { query, stream_id });
                }
            }
            Msg::Column(col) => {
                let payload = col.payload.as_ref();
                for res in payload.into_iter(col.metadata.component_type.clone()) {
                    match res {
                        Ok(ColumnValue { entity_id, value }) => {
                            let filter_result = self
                                .outgoing_filter
                                .contains(&(Some(col.metadata.component_id()), Some(entity_id)))
                                || self
                                    .outgoing_filter
                                    .contains(&(Some(col.metadata.component_id()), None))
                                || self.outgoing_filter.contains(&(None, Some(entity_id)));
                            if !filter_result {
                                continue;
                            }
                            if let Err(err) = self
                                .outgoing_tx
                                .send(Some((
                                    col.metadata.component_id(),
                                    entity_id,
                                    value.into_owned(),
                                )))
                                .await
                            {
                                warn!(?err, "error sending incoming message");
                            }
                        }
                        Err(err) => {
                            warn!(?err, "error decoding column");
                        }
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
}

pub struct Rx<D: DriverMode> {
    channel: Option<RxChannel>,
    _phantom: std::marker::PhantomData<D>,
}

impl Decomponentize for Option<RxChannel> {
    fn apply_value<D: ComponentValueDim>(
        &mut self,
        _: ComponentId,
        _: EntityId,
        _: ComponentValue<'_, D>,
    ) {
    }
}

impl Componentize for Option<RxChannel> {
    fn sink_columns(&self, output: &mut impl Decomponentize) {
        if let Some(rx) = self {
            while let Some((component_id, entity_id, value)) =
                rx.try_recv().ok().and_then(|msg| msg)
            {
                output.apply_value(component_id, entity_id, value);
            }
        }
    }
}

impl<D: DriverMode> System for Rx<D> {
    type World = Option<RxChannel>;

    type Driver = D;

    fn init_world(&mut self) -> Self::World {
        self.channel.take()
    }

    fn update(&mut self, _world: &mut Self::World) {}
}

pub struct Tx<D: DriverMode> {
    channel: Option<TxChannel>,
    _phantom: std::marker::PhantomData<D>,
}

impl Decomponentize for Option<TxChannel> {
    fn apply_value<D: ComponentValueDim>(
        &mut self,
        component_id: ComponentId,
        entity_id: EntityId,
        value: ComponentValue<'_, D>,
    ) {
        if let Some(tx) = self {
            if let Err(err) = tx.try_send(Some((
                component_id,
                entity_id,
                value.into_dyn().into_owned(),
            ))) {
                warn!(?err, "error sending outgoing message");
            }
        }
    }
}

impl Componentize for Option<TxChannel> {
    fn sink_columns(&self, _output: &mut impl Decomponentize) {}
}

impl<D: DriverMode> System for Tx<D> {
    type World = Option<TxChannel>;

    type Driver = D;

    fn init_world(&mut self) -> Self::World {
        self.channel.take()
    }

    fn update(&mut self, _world: &mut Self::World) {}
}

pub fn tcp_connect<D: DriverMode>(
    addr: SocketAddr,
    filters: &[Query],
    metadata: impl IntoIterator<Item = Metadata>,
) -> (Tx<D>, Rx<D>) {
    let (client, tx, rx) = Client::new(addr, filters, 1024, metadata);
    std::thread::spawn(move || {
        let rt = Runtime::new().unwrap();
        if let Err(err) = rt.block_on(client.run()) {
            warn!(?err, "error running client");
        }
    });
    (
        Tx {
            channel: Some(tx),
            _phantom: std::marker::PhantomData,
        },
        Rx {
            channel: Some(rx),
            _phantom: std::marker::PhantomData,
        },
    )
}

pub struct Client {
    addr: SocketAddr,
    queries: Vec<Query>,
    incoming_tx: TxChannel,
    outgoing_rx: RxChannel,
    streams: HashMap<ComponentId, StreamId>,
    metadata: HashMap<ComponentId, Metadata>,
}

impl Client {
    pub fn new(
        addr: SocketAddr,
        queries: &[Query],
        capacity: usize,
        metadata: impl IntoIterator<Item = Metadata>,
    ) -> (Self, TxChannel, RxChannel) {
        let (outgoing_tx, outgoing_rx) = thingbuf::mpsc::channel(capacity);
        let (incoming_tx, incoming_rx) = thingbuf::mpsc::channel(capacity);
        let metadata = metadata
            .into_iter()
            .map(|m| (m.component_id(), m))
            .collect();

        (
            Self {
                addr,
                queries: queries.to_vec(),
                incoming_tx,
                outgoing_rx,
                metadata,
                streams: HashMap::new(),
            },
            outgoing_tx,
            incoming_rx,
        )
    }

    pub async fn run(mut self) -> Result<(), Error> {
        #[derive(Debug)]
        enum Event {
            Incoming(Msg<Bytes>),
            Outgoing(Option<MsgTriplet>),
        }

        let tcp_stream = TcpStream::connect(self.addr).await?;
        let (rx, tx) = tcp_stream.into_split();
        let mut tx = AsyncClient::new(FramedWrite::new(tx, LengthDelimitedCodec::new()));
        let mut rx = AsyncClient::new(FramedRead::new(rx, LengthDelimitedCodec::new()));
        tx.send(Packet::<Payload<Bytes>>::control(ControlMsg::Connect))
            .await?;
        for query in self.queries.into_iter() {
            tx.send(Packet::<Payload<Bytes>>::subscribe(query)).await?;
        }
        loop {
            let event = tokio::select! {
                Ok(res) = rx.recv() => Event::Incoming(res),
                Some(msg) = self.outgoing_rx.recv() => Event::Outgoing(msg),
            };
            match event {
                Event::Incoming(msg) => match msg {
                    Msg::Control(_) => {}
                    Msg::Column(col) => {
                        let payload = col.payload.as_ref();
                        for res in payload.into_iter(col.metadata.component_type.clone()) {
                            let Ok(value) = res else {
                                continue;
                            };
                            if let Err(err) = self
                                .incoming_tx
                                .send(Some((
                                    col.metadata.component_id(),
                                    value.entity_id,
                                    value.value.into_owned(),
                                )))
                                .await
                            {
                                warn!(?err, "error sending incoming message");
                            }
                        }
                    }
                },
                Event::Outgoing(Some((component_id, entity_id, value))) => {
                    let stream_id = if let Some(stream_id) = self.streams.get(&component_id) {
                        *stream_id
                    } else {
                        let stream_id = StreamId::rand();
                        let Some(metadata) = self.metadata.get(&component_id) else {
                            //warn!(?component_id, "missing metadata for component");
                            continue;
                        };
                        self.streams.insert(component_id, stream_id);
                        tx.send(Packet::<Payload<Bytes>>::start_stream(
                            stream_id,
                            metadata.clone(),
                        ))
                        .await?;

                        stream_id
                    };
                    let packet: Packet<Payload<Bytes>> = Packet::column(
                        stream_id,
                        ColumnPayload::try_from_value_iter(
                            0,
                            [ColumnValue {
                                entity_id,
                                value: value.into_dyn(),
                            }]
                            .into_iter(),
                        )
                        .unwrap(),
                    );
                    tx.send(packet).await?;
                }
                Event::Outgoing(None) => {
                    println!("outgoing channel closed");
                }
            }
        }
    }
}
