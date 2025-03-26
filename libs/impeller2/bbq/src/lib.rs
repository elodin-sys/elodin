use bbq2::{
    prod_cons::framed::{FramedConsumer, FramedProducer},
    queue::ArcBBQueue,
    traits::{coordination::cas::AtomicCoord, notifier::maitake::MaiNotSpsc, storage::BoxedSlice},
};
use impeller2::types::{LenPacket, OwnedPacket, PrimType};
use std::{
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::Arc,
};
use stellarator_buf::{IoBuf, IoBufMut};

pub type AsyncArcQueue = ArcBBQueue<BoxedSlice, AtomicCoord, MaiNotSpsc>;
pub type AsyncArcQueueInner = Arc<bbq2::queue::BBQueue<BoxedSlice, AtomicCoord, MaiNotSpsc>>;
pub type AsyncArcQueueTx =
    FramedProducer<AsyncArcQueueInner, BoxedSlice, AtomicCoord, MaiNotSpsc, usize>;
pub type AsyncArcQueueRx =
    FramedConsumer<AsyncArcQueueInner, BoxedSlice, AtomicCoord, MaiNotSpsc, usize>;

type PacketGrantWInner = bbq2::prod_cons::framed::FramedGrantW<
    AsyncArcQueueInner,
    BoxedSlice,
    AtomicCoord,
    MaiNotSpsc,
    usize,
>;

pub struct PacketGrantW(PacketGrantWInner);

unsafe impl Send for PacketGrantW {}

impl PacketGrantW {
    pub fn new(inner: PacketGrantWInner) -> Self {
        let mut this = PacketGrantW(inner);
        let padding = this.padding();
        for x in &mut this.0[..padding] {
            *x = 0;
        }
        this
    }

    fn padding(&self) -> usize {
        PrimType::U64.padding(self.0.deref().as_ptr() as usize)
    }

    pub fn commit(self, used: usize) {
        let padding = self.padding();
        self.0.commit(padding + used)
    }

    pub fn commit_len_pkt(mut self, len_pkt: LenPacket) {
        self[..len_pkt.inner.len()].copy_from_slice(&len_pkt.inner);
        self.commit(len_pkt.inner.len());
    }
}

impl Deref for PacketGrantW {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        stellarator_buf::deref(self)
    }
}

impl DerefMut for PacketGrantW {
    fn deref_mut(&mut self) -> &mut Self::Target {
        stellarator_buf::deref_mut(self)
    }
}

unsafe impl IoBuf for PacketGrantW {
    fn stable_init_ptr(&self) -> *const u8 {
        self.0.deref().as_ptr().wrapping_add(self.padding())
    }

    fn init_len(&self) -> usize {
        self.0.deref().len() - self.padding()
    }

    fn total_len(&self) -> usize {
        self.0.deref().len() - self.padding()
    }
}

unsafe impl IoBufMut for PacketGrantW {
    fn stable_mut_ptr(&mut self) -> NonNull<MaybeUninit<u8>> {
        unsafe {
            NonNull::new_unchecked(
                self.0.deref_mut().as_ptr().wrapping_add(self.padding()) as *mut MaybeUninit<u8>
            )
        }
    }

    unsafe fn set_init(&mut self, _len: usize) {}
}

pub struct PacketGrantR(
    Option<
        bbq2::prod_cons::framed::FramedGrantR<
            AsyncArcQueueInner,
            BoxedSlice,
            AtomicCoord,
            MaiNotSpsc,
            usize,
        >,
    >,
);

unsafe impl Sync for PacketGrantR {}
impl Drop for PacketGrantR {
    fn drop(&mut self) {
        let inner = self.0.take().expect("missing inner");
        inner.release();
    }
}

impl PacketGrantR {
    fn padding(&self) -> usize {
        PrimType::U64.padding(self.inner().deref().as_ptr() as usize)
    }

    fn inner(
        &self,
    ) -> &bbq2::prod_cons::framed::FramedGrantR<
        AsyncArcQueueInner,
        BoxedSlice,
        AtomicCoord,
        MaiNotSpsc,
        usize,
    > {
        self.0.as_ref().expect("missing inner")
    }
}

unsafe impl IoBuf for PacketGrantR {
    fn stable_init_ptr(&self) -> *const u8 {
        self.inner().deref().as_ptr()
    }

    fn init_len(&self) -> usize {
        self.inner().deref().len()
    }

    fn total_len(&self) -> usize {
        self.inner().deref().len()
    }
}

pub trait RxExt {
    fn try_recv_pkt(&mut self) -> Option<OwnedPacket<PacketGrantR>>;
}

impl RxExt for AsyncArcQueueRx {
    fn try_recv_pkt(&mut self) -> Option<OwnedPacket<PacketGrantR>> {
        let grant = self.read().ok()?;
        let packet_buf = PacketGrantR(Some(grant));
        let padding = packet_buf.padding();
        OwnedPacket::parse_with_offset(packet_buf, 4 + padding)
            .inspect_err(|err| println!("{err:?}"))
            .ok()
    }
}
