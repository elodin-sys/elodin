use crate::arena::*;
use crate::peripheral::*;

pub struct DmaBuf<const N: usize, H: HalDmaRegExt> {
    pub dma_ch: DmaChannel<H>,
    pub shared_buf: &'static mut [u16; N],
    pub staging_buf: &'static mut [u16; N],
    pub transfer_in_progress: bool,
}

impl<const N: usize, H: HalDmaRegExt> DmaBuf<N, H> {
    pub fn new<B: AsMut<[u8]> + 'static>(dma_ch: DmaChannel<H>, alloc: &mut ArenaAlloc<B>) -> Self {
        let shared_buf = alloc.leak([0; N]);
        let staging_buf = alloc.leak([0; N]);
        Self {
            dma_ch,
            shared_buf,
            staging_buf,
            transfer_in_progress: false,
        }
    }
}
