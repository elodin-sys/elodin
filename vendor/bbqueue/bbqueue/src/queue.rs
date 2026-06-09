use core::marker::PhantomData;

use crate::{
    prod_cons::{
        framed::{FramedConsumer, FramedProducer},
        stream::{StreamConsumer, StreamProducer},
    },
    traits::{
        coordination::Coord,
        notifier::Notifier,
        storage::{ConstStorage, Storage},
    },
};

#[cfg(feature = "alloc")]
use crate::traits::bbqhdl::BbqHandle;

/// A standard bbqueue
pub struct BBQueue<S, C, N> {
    pub(crate) sto: S,
    pub(crate) cor: C,
    pub(crate) not: N,
}

impl<S: Storage, C: Coord, N: Notifier> BBQueue<S, C, N> {
    /// Create a new [`BBQueue`] with the given [`Storage`] impl
    pub fn new_with_storage(sto: S) -> Self {
        Self {
            sto,
            cor: C::INIT,
            not: N::INIT,
        }
    }
}

/// A BBQueue wrapped in an Arc
#[cfg(feature = "alloc")]
pub struct ArcBBQueue<S, C, N>(pub(crate) alloc::sync::Arc<BBQueue<S, C, N>>);

#[cfg(feature = "alloc")]
impl<S: Storage, C: Coord, N: Notifier> ArcBBQueue<S, C, N> {
    /// Create a new [`BBQueue`] with the given [`Storage`] impl
    pub fn new_with_storage(sto: S) -> Self {
        Self(alloc::sync::Arc::new(BBQueue::new_with_storage(sto)))
    }
}

#[allow(clippy::new_without_default)]
impl<S: ConstStorage, C: Coord, N: Notifier> BBQueue<S, C, N> {
    /// Create a new `BBQueue` in a const context
    pub const fn new() -> Self {
        Self {
            sto: S::INIT,
            cor: C::INIT,
            not: N::INIT,
        }
    }
}

impl<S: Storage, C: Coord, N: Notifier> BBQueue<S, C, N> {
    /// Create a new [`FramedProducer`] for this [`BBQueue`]
    ///
    /// Although mixing stream and framed consumer/producers will not result in UB,
    /// it will also not work correctly.
    pub const fn framed_producer(&self) -> FramedProducer<&'_ Self> {
        FramedProducer {
            bbq: self,
            pd: PhantomData,
        }
    }

    /// Create a new [`FramedConsumer`] for this [`BBQueue`]
    ///
    /// Although mixing stream and framed consumer/producers will not result in UB,
    /// it will also not work correctly.
    pub const fn framed_consumer(&self) -> FramedConsumer<&'_ Self> {
        FramedConsumer {
            bbq: self,
            pd: PhantomData,
        }
    }

    /// Create a new [`StreamProducer`] for this [`BBQueue`]
    ///
    /// Although mixing stream and framed consumer/producers will not result in UB,
    /// it will also not work correctly.
    pub const fn stream_producer(&self) -> StreamProducer<&'_ Self> {
        StreamProducer { bbq: self }
    }

    /// Create a new [`StreamConsumer`] for this [`BBQueue`]
    ///
    /// Although mixing stream and framed consumer/producers will not result in UB,
    /// it will also not work correctly.
    pub const fn stream_consumer(&self) -> StreamConsumer<&'_ Self> {
        StreamConsumer { bbq: self }
    }

    /// Get the total capacity of the buffer, e.g. how much space is present in [`Storage`]
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        // SAFETY: capacity never changes, therefore reading the len is safe
        unsafe {
            self.sto.ptr_len().1
        }
    }

    /// Get access to the internal storage implementation details
    ///
    /// NOTE: Although this method is safe, use of the `Storage` methods are not.
    /// You should *never* attempt to access or modify the underlying data contained
    /// in a storage implementation while the bbqueue is live. That will IMMEDIATELY
    /// lead to undefined behavior.
    ///
    /// As far as I am aware, the only reasonable use for this is for cases where you
    /// have a custom `Storage` implementation that has unique teardown/drop in place
    /// requirements. Treat any uses of this function with *extreme* caution!
    #[inline(always)]
    pub fn storage(&self) -> &S {
        &self.sto
    }
}

#[cfg(feature = "alloc")]
impl<S: Storage, C: Coord, N: Notifier> crate::queue::ArcBBQueue<S, C, N> {
    /// Create a new [`FramedProducer`] for this [`BBQueue`]
    ///
    /// Although mixing stream and framed consumer/producers will not result in UB,
    /// it will also not work correctly.
    pub fn framed_producer(&self) -> FramedProducer<alloc::sync::Arc<BBQueue<S, C, N>>> {
        FramedProducer {
            bbq: self.0.bbq_ref(),
            pd: PhantomData,
        }
    }

    /// Create a new [`FramedConsumer`] for this [`BBQueue`]
    ///
    /// Although mixing stream and framed consumer/producers will not result in UB,
    /// it will also not work correctly.
    pub fn framed_consumer(&self) -> FramedConsumer<alloc::sync::Arc<BBQueue<S, C, N>>> {
        FramedConsumer {
            bbq: self.0.bbq_ref(),
            pd: PhantomData,
        }
    }

    /// Create a new [`FramedProducer`] for this [`BBQueue`] with a custom
    /// frame length header type.
    ///
    /// This can be used to support large frame sizes by selecting a larger
    /// [`LenHeader`] type such as `usize`.
    pub fn framed_producer_with_header<H: crate::prod_cons::framed::LenHeader>(
        &self,
    ) -> FramedProducer<alloc::sync::Arc<BBQueue<S, C, N>>, H> {
        FramedProducer {
            bbq: self.0.bbq_ref(),
            pd: PhantomData,
        }
    }

    /// Create a new [`FramedConsumer`] for this [`BBQueue`] with a custom
    /// frame length header type.
    ///
    /// This can be used to support large frame sizes by selecting a larger
    /// [`LenHeader`] type such as `usize`.
    pub fn framed_consumer_with_header<H: crate::prod_cons::framed::LenHeader>(
        &self,
    ) -> FramedConsumer<alloc::sync::Arc<BBQueue<S, C, N>>, H> {
        FramedConsumer {
            bbq: self.0.bbq_ref(),
            pd: PhantomData,
        }
    }

    /// Create a new [`StreamProducer`] for this [`BBQueue`]
    ///
    /// Although mixing stream and framed consumer/producers will not result in UB,
    /// it will also not work correctly.
    pub fn stream_producer(&self) -> StreamProducer<alloc::sync::Arc<BBQueue<S, C, N>>> {
        StreamProducer {
            bbq: self.0.bbq_ref(),
        }
    }

    /// Create a new [`StreamConsumer`] for this [`BBQueue`]
    ///
    /// Although mixing stream and framed consumer/producers will not result in UB,
    /// it will also not work correctly.
    pub fn stream_consumer(&self) -> StreamConsumer<alloc::sync::Arc<BBQueue<S, C, N>>> {
        StreamConsumer {
            bbq: self.0.bbq_ref(),
        }
    }

    /// Get the total capacity of the buffer, e.g. how much space is present in [`Storage`]
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        // SAFETY: capacity never changes, therefore reading the len is safe
        unsafe {
            self.0.sto.ptr_len().1
        }
    }

    /// Get access to the internal storage implementation details
    ///
    /// NOTE: Although this method is safe, use of the `Storage` methods are not.
    /// You should *never* attempt to access or modify the underlying data contained
    /// in a storage implementation while the bbqueue is live. That will IMMEDIATELY
    /// lead to undefined behavior.
    ///
    /// As far as I am aware, the only reasonable use for this is for cases where you
    /// have a custom `Storage` implementation that has unique teardown/drop in place
    /// requirements. Treat any uses of this function with *extreme* caution!
    #[inline(always)]
    pub fn storage(&self) -> &S {
        &self.0.sto
    }
}

#[cfg(test)]
mod test {
    use crate::traits::{
        coordination::cas::AtomicCoord, notifier::polling::Polling, storage::Inline,
    };

    use super::*;

    type Queue = BBQueue<Inline<4096>, AtomicCoord, Polling>;
    static QUEUE: Queue = BBQueue::new();
    static PRODUCER: FramedProducer<&'static Queue, u16> = QUEUE.framed_producer();
    static CONSUMER: FramedConsumer<&'static Queue, u16> = QUEUE.framed_consumer();

    #[test]
    fn handles() {
        let mut wgr = PRODUCER.grant(16).unwrap();
        wgr.iter_mut().for_each(|w| *w = 123);
        wgr.commit(16);

        let rgr = CONSUMER.read().unwrap();
        assert_eq!(rgr.len(), 16);
        for b in rgr.iter() {
            assert_eq!(*b, 123);
        }
        rgr.release();
    }
}
