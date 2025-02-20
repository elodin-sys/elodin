use bbq2::{queue::ArcBBQueue, traits::storage::BoxedSlice};
use impeller2::{
    com_de::{Componentize, Decomponentize},
    registry::HashMapRegistry,
    table::{Entry, VTableBuilder},
    types::{ComponentView, LenPacket, Msg, MsgExt, OwnedPacket, PacketId, Timestamp},
};
use impeller2_bbq::{AsyncArcQueueRx, RxExt};
use impeller2_stella::queue::tcp_connect;
use impeller2_wkt::{Stream, StreamFilter, VTableMsg};
use std::{marker::PhantomData, net::SocketAddr, time::Duration};
use thingbuf::mpsc;

use crate::{drivers::DriverMode, System};

pub struct TcpSink<W, D> {
    tx: mpsc::Sender<Option<LenPacket>>,
    world: PhantomData<W>,
    driver: PhantomData<D>,
    vtable_id: PacketId,
}

pub struct TcpSource<W, D> {
    rx: AsyncArcQueueRx,
    world: PhantomData<W>,
    driver: PhantomData<D>,
    vtable: HashMapRegistry,
}

impl<W, D> System for TcpSink<W, D>
where
    W: Componentize + Decomponentize + Default,
    D: DriverMode,
{
    type World = W;

    type Driver = D;

    fn update(&mut self, world: &mut Self::World) {
        let Ok(mut send_ref) = self.tx.try_send_ref() else {
            return;
        };
        let buf = send_ref.get_or_insert_with(|| {
            LenPacket::new(impeller2::types::PacketTy::Table, self.vtable_id, 1024)
        });
        world.sink_columns(&mut |_, _, value: ComponentView<'_>, _| match value {
            ComponentView::U8(view) => buf.extend_aligned(view.buf()),
            ComponentView::U16(view) => buf.extend_aligned(view.buf()),
            ComponentView::U32(view) => buf.extend_aligned(view.buf()),
            ComponentView::U64(view) => buf.extend_aligned(view.buf()),
            ComponentView::I8(view) => buf.extend_aligned(view.buf()),
            ComponentView::I16(view) => buf.extend_aligned(view.buf()),
            ComponentView::I32(view) => buf.extend_aligned(view.buf()),
            ComponentView::I64(view) => buf.extend_aligned(view.buf()),
            ComponentView::Bool(view) => buf.extend_aligned(view.buf()),
            ComponentView::F32(view) => buf.extend_aligned(view.buf()),
            ComponentView::F64(view) => buf.extend_aligned(view.buf()),
        });
    }
}

#[derive(Default)]
struct VTableSink {
    builder: VTableBuilder<Vec<Entry>, Vec<u8>>,
}

impl VTableSink {
    pub fn build(self) -> impeller2::table::VTable<Vec<Entry>, Vec<u8>> {
        self.builder.build()
    }
}

impl Decomponentize for VTableSink {
    fn apply_value(
        &mut self,
        component_id: impeller2::types::ComponentId,
        entity_id: impeller2::types::EntityId,
        value: ComponentView<'_>,
        _timestamp: Option<Timestamp>,
    ) {
        let prim_ty = value.prim_type();
        let shape = value.shape();
        let _ = self.builder.column(
            component_id,
            prim_ty,
            shape.iter().map(|&d| d as u64),
            [entity_id],
        );
    }
}

impl<W, D> System for TcpSource<W, D>
where
    W: Componentize + Decomponentize + Default,
    D: DriverMode,
{
    type World = W;

    type Driver = D;

    fn update(&mut self, world: &mut Self::World) {
        while let Some(pkt) = self.rx.try_recv_pkt() {
            match pkt {
                OwnedPacket::Table(table) => {
                    if let Err(err) = table.sink(&self.vtable, world) {
                        tracing::error!(?err, "error sinking table");
                    }
                }
                OwnedPacket::Msg(m) if m.id == VTableMsg::ID => {
                    let Ok(vtable) = m.parse::<VTableMsg>().inspect_err(|err| {
                        tracing::error!(?err, "error parsing vtable");
                    }) else {
                        continue;
                    };
                    self.vtable.map.insert(vtable.id, vtable.vtable);
                }
                _ => {}
            }
        }
    }
}

pub fn tcp_pair<W: Default + Componentize + Decomponentize, D>(
    addr: SocketAddr,
) -> (TcpSink<W, D>, TcpSource<W, D>) {
    let queue = ArcBBQueue::new_with_storage(BoxedSlice::new(1024 * 1024));
    let (incoming_packet_rx, mut incoming_packet_tx) = queue.framed_split();
    let (outgoing_packet_tx, mut outgoing_packet_rx) = mpsc::channel::<Option<LenPacket>>(512);
    let vtable_id: PacketId = fastrand::u64(..).to_le_bytes()[..3].try_into().unwrap();
    let sink = TcpSink {
        tx: outgoing_packet_tx,
        world: PhantomData,
        driver: PhantomData,
        vtable_id,
    };
    let source = TcpSource {
        rx: incoming_packet_rx,
        world: PhantomData,
        driver: PhantomData,
        vtable: Default::default(),
    };
    let mut vtable = VTableSink::default();
    let world = W::default();
    world.sink_columns(&mut vtable);
    let vtable = vtable.build();

    let initial_msgs = move |_| {
        vtable
            .id_pair_iter()
            .map(|(entity_id, component_id)| {
                Stream {
                    filter: StreamFilter {
                        component_id: Some(component_id),
                        entity_id: Some(entity_id),
                    },
                    id: fastrand::u64(..),
                    behavior: impeller2_wkt::StreamBehavior::RealTime,
                }
                .to_len_packet()
            })
            .chain(std::iter::once(
                VTableMsg {
                    id: vtable_id,
                    vtable: vtable.clone(),
                }
                .to_len_packet(),
            ))
            .collect::<Vec<_>>()
            .into_iter()
    };
    stellarator::struc_con::stellar(move || async move {
        loop {
            if let Err(err) = tcp_connect(
                addr,
                &mut outgoing_packet_rx,
                &mut incoming_packet_tx,
                0,
                &initial_msgs,
                || {},
            )
            .await
            {
                tracing::trace!(?err, "connection ended with error");
                stellarator::sleep(Duration::from_millis(50)).await;
            }
        }
    });
    (sink, source)
}
