use core::borrow::{Borrow, BorrowMut};

use hal::{clocks::Clocks, dma, pac, timer};

use crate::arena::DmaAlloc;
use crate::dma::DmaBuf;

const TIMX_CCR1_OFFSET: u8 = 0x34;

pub trait HalTimerRegExt: Sized {
    const TIMER: u8;
    fn timer(
        self,
        freq: fugit::Rate<u32, 1, 1_000>,
        cfg: timer::TimerConfig,
        clocks: &Clocks,
    ) -> timer::Timer<Self>;
}

pub trait HalDmaRegExt: Sized {
    const DMA: u8;
    fn dma(self) -> dma::Dma<Self>;
    fn dma_periph() -> dma::DmaPeriph {
        match Self::DMA {
            1 => dma::DmaPeriph::Dma1,
            2 => dma::DmaPeriph::Dma2,
            _ => unreachable!(),
        }
    }
}

pub trait HalTimerExt: Sized + Borrow<timer::Timer<Self::HalTimerReg>> {
    type HalTimerReg: HalTimerRegExt;

    fn enable_pwm(&mut self);
    fn enable_dma_interrupt(&mut self);
    fn max_duty_cycle(&self) -> u32;
    #[allow(clippy::too_many_arguments)]
    /// # Safety
    /// The caller must ensure that the `buf` is valid for the lifetime of the program
    /// and that `buf` is not written to while the DMA transfer is in progress.
    unsafe fn hal_write_dma_burst(
        &mut self,
        buf: &[u16],
        base_address: u8,
        burst_len: u8,
        dma_channel: dma::DmaChannel,
        channel_cfg: dma::ChannelCfg,
        ds_32_bits: bool,
        dma_periph: dma::DmaPeriph,
    );

    fn write<const N: usize, const CHANNEL: u8, H: HalDmaRegExt>(
        &mut self,
        dma_buf: &mut DmaBuf<N, CHANNEL, H>,
    ) -> bool
    where
        dma::Dma<H>: HalDmaExt,
    {
        if dma_buf.transfer_in_progress && !dma_buf.dma_ch.transfer_complete() {
            defmt::warn!("DMA transfer is not complete");
            return false;
        }

        dma_buf.dma_ch.clear_interrupt();
        dma_buf.transfer_in_progress = false;

        let base_addr = TIMX_CCR1_OFFSET / 4;
        let burst_len = 4u8;
        assert_eq!(dma_buf.staging_buf.len() % burst_len as usize, 0);
        let dma_channel = DmaCh::<CHANNEL, H>::dma_channel();
        let channel_cfg = dma::ChannelCfg::default();
        let dma_periph = H::dma_periph();

        dma_buf.shared_buf.copy_from_slice(dma_buf.staging_buf);
        dma_buf.transfer_in_progress = true;
        // SAFETY: `dma_buf.buf` is leaked memory, and will remain valid for the lifetime of the program
        // `dma_buf.buf` can only be written to when the DMA transfer is not in progress to prevent data corruption
        unsafe {
            self.hal_write_dma_burst(
                dma_buf.shared_buf,
                base_addr,
                burst_len,
                dma_channel,
                channel_cfg,
                false,
                dma_periph,
            );
        }
        true
    }
    fn ch<const CHANNEL: u8>(&self) -> TimCh<CHANNEL, Self::HalTimerReg> {
        TimCh {
            _timer: self.borrow(),
        }
    }
    fn ch1(&self) -> TimCh<1, Self::HalTimerReg> {
        self.ch()
    }
    fn ch2(&self) -> TimCh<2, Self::HalTimerReg> {
        self.ch()
    }
    fn ch3(&self) -> TimCh<3, Self::HalTimerReg> {
        self.ch()
    }
    fn ch4(&self) -> TimCh<4, Self::HalTimerReg> {
        self.ch()
    }
}

pub trait HalDmaExt: Sized + BorrowMut<dma::Dma<Self::HalDmaReg>> {
    type HalDmaReg: HalDmaRegExt;
    fn hal_transfer_is_complete(&mut self, channel: dma::DmaChannel) -> bool;
    fn ch<const CHANNEL: u8>(&mut self) -> DmaCh<CHANNEL, Self::HalDmaReg> {
        DmaCh {
            dma: self.borrow_mut(),
        }
    }
    fn ch1(&mut self) -> DmaCh<1, Self::HalDmaReg> {
        self.ch()
    }
}

pub trait DmaMuxInput {
    const DMA_INPUT: dma::DmaInput;
}

pub struct TimCh<'a, const CHANNEL: u8, H: HalTimerRegExt> {
    _timer: &'a timer::Timer<H>,
}

pub type Tim1<'a, const CHANNEL: u8> = TimCh<'a, CHANNEL, pac::TIM1>;
pub type Tim2<'a, const CHANNEL: u8> = TimCh<'a, CHANNEL, pac::TIM2>;
pub type Tim3<'a, const CHANNEL: u8> = TimCh<'a, CHANNEL, pac::TIM3>;

pub type Tim1Ch1<'a> = Tim1<'a, 1>;
pub type Tim1Ch2<'a> = Tim1<'a, 2>;
pub type Tim1Ch3<'a> = Tim1<'a, 3>;
pub type Tim1Ch4<'a> = Tim1<'a, 4>;

pub type Tim2Ch1<'a> = Tim2<'a, 1>;
pub type Tim2Ch2<'a> = Tim2<'a, 2>;
pub type Tim2Ch3<'a> = Tim2<'a, 3>;
pub type Tim2Ch4<'a> = Tim2<'a, 4>;

pub type Tim3Ch1<'a> = Tim3<'a, 1>;
pub type Tim3Ch2<'a> = Tim3<'a, 2>;
pub type Tim3Ch3<'a> = Tim3<'a, 3>;
pub type Tim3Ch4<'a> = Tim3<'a, 4>;

pub struct DmaCh<'a, const CHANNEL: u8, H: HalDmaRegExt> {
    dma: &'a mut dma::Dma<H>,
}

macro_rules! impl_hal_timer {
    ($tim:literal) => {
        paste::paste! {
        impl HalTimerExt for timer::Timer<pac::[<TIM $tim>]>
        {
            type HalTimerReg = pac::[<TIM $tim>];
            fn enable_pwm(&mut self) {
                self.enable_pwm_output(timer::TimChannel::C1, timer::OutputCompare::Pwm1, 0.);
                self.enable_pwm_output(timer::TimChannel::C2, timer::OutputCompare::Pwm1, 0.);
                self.enable_pwm_output(timer::TimChannel::C3, timer::OutputCompare::Pwm1, 0.);
                self.enable_pwm_output(timer::TimChannel::C4, timer::OutputCompare::Pwm1, 0.);
            }
            fn enable_dma_interrupt(&mut self) {
                self.enable_interrupt(timer::TimerInterrupt::UpdateDma);
            }
            fn max_duty_cycle(&self) -> u32 {
                self.get_max_duty().into()
            }
            unsafe fn hal_write_dma_burst(
                &mut self,
                buf: &[u16],
                base_address: u8,
                burst_len: u8,
                dma_channel: dma::DmaChannel,
                channel_cfg: dma::ChannelCfg,
                ds_32_bits: bool,
                dma_periph: dma::DmaPeriph,
            ) {
                self.write_dma_burst(
                    buf,
                    base_address,
                    burst_len,
                    dma_channel,
                    channel_cfg,
                    ds_32_bits,
                    dma_periph,
                );
            }
        }
        }
    };
}

impl_hal_timer!(1);
impl_hal_timer!(2);
impl_hal_timer!(3);

macro_rules! impl_hal_dma {
    ($dma:literal) => {
        paste::paste! {
        impl HalDmaExt for dma::Dma<pac::[<DMA $dma>]>
        {
            type HalDmaReg = pac::[<DMA $dma>];
            fn hal_transfer_is_complete(&mut self, channel: dma::DmaChannel) -> bool {
                self.transfer_is_complete(channel)
            }
        }
        }
    };
}

impl_hal_dma!(1);
impl_hal_dma!(2);

macro_rules! impl_hal_timer_reg {
    ($tim_num:literal) => {
        paste::paste! {
        impl HalTimerRegExt for pac::[<TIM $tim_num>] {
            const TIMER: u8 = $tim_num;
            fn timer(
                self,
                freq: fugit::Rate<u32, 1, 1_000>,
                cfg: timer::TimerConfig,
                clocks: &Clocks,
            ) -> timer::Timer<Self> {
                timer::Timer::[<new_tim $tim_num>](
                    self,
                    freq.to_Hz() as f32,
                    cfg,
                    clocks,
                )
            }
        }
        }
    };
}

impl_hal_timer_reg!(1);
impl_hal_timer_reg!(2);
impl_hal_timer_reg!(3);

impl HalDmaRegExt for pac::DMA1 {
    const DMA: u8 = 1;
    fn dma(self) -> dma::Dma<Self> {
        dma::Dma::new(self)
    }
}

impl HalDmaRegExt for pac::DMA2 {
    const DMA: u8 = 2;
    fn dma(self) -> dma::Dma<Self> {
        dma::Dma::new(self)
    }
}

macro_rules! impl_timer_dma_mux {
    ($tim:literal) => {
        paste::paste! {
        impl DmaMuxInput for timer::Timer<pac::[<TIM $tim>]> {
            const DMA_INPUT: dma::DmaInput = dma::DmaInput::[<Tim $tim Up>];
        }
        }
    };
}

impl_timer_dma_mux!(1);
impl_timer_dma_mux!(2);
impl_timer_dma_mux!(3);

impl<'a, const CHANNEL: u8, H: HalDmaRegExt> DmaCh<'a, CHANNEL, H>
where
    dma::Dma<H>: HalDmaExt,
{
    const fn dma_channel() -> dma::DmaChannel {
        match CHANNEL {
            0 => dma::DmaChannel::C0,
            1 => dma::DmaChannel::C1,
            2 => dma::DmaChannel::C2,
            3 => dma::DmaChannel::C3,
            4 => dma::DmaChannel::C4,
            5 => dma::DmaChannel::C5,
            6 => dma::DmaChannel::C6,
            7 => dma::DmaChannel::C7,
            _ => unreachable!(),
        }
    }

    pub fn mux<M: DmaMuxInput>(&self, _: &mut M) {
        dma::mux(H::dma_periph(), Self::dma_channel(), M::DMA_INPUT);
    }

    pub fn clear_interrupt(&self) {
        dma::clear_interrupt(
            H::dma_periph(),
            Self::dma_channel(),
            dma::DmaInterrupt::TransferComplete,
        );
    }

    pub fn buf<const N: usize, B: AsMut<[u8]>>(
        self,
        alloc: &mut DmaAlloc<B>,
    ) -> DmaBuf<'a, N, CHANNEL, H> {
        DmaBuf::new(self, alloc)
    }

    pub fn transfer_complete(&mut self) -> bool {
        self.dma.hal_transfer_is_complete(Self::dma_channel())
    }
}
