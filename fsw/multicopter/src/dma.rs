use crate::arena::*;
use crate::peripheral::*;

pub struct DmaBuf<'a, const N: usize, const CHANNEL: u8, H: HalDmaRegExt> {
    pub dma_ch: DmaCh<'a, CHANNEL, H>,
    pub shared_buf: &'static mut [u16; N],
    pub staging_buf: &'static mut [u16; N],
    pub transfer_in_progress: bool,
}

impl<'a, const N: usize, const CHANNEL: u8, H: HalDmaRegExt> DmaBuf<'a, N, CHANNEL, H> {
    pub fn new<B: AsMut<[u8]> + 'static>(
        dma_ch: DmaCh<'a, CHANNEL, H>,
        alloc: &mut DmaAlloc<B>,
    ) -> Self {
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
