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
        'recv_loop: while let Ok(MsgPair { msg, tx }) = rx.try_recv() {
            match msg {
                Msg::Control(ControlMsg::Subscribe { query }) => {
                    let Some(tx) = tx.and_then(|tx| tx.upgrade()) else {
                        continue 'recv_loop;
                    };
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
        handler.tick(&mut world);
        world.sink_columns(&mut output);
        for (query, stream_id, tx) in &subscriptions {
            if let Some(column) = output.get(&query.component_id) {
                if let Err(err) = tx.try_send(Packet::column(*stream_id, column.clone())) {
                    warn!(?err, "error sending packet");
                }
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
