use core::{borrow::Borrow, cell::UnsafeCell, ops::Deref};

use hal::{clocks::Clocks, dma, pac, timer};

use crate::dma::DmaBuf;

const TIMX_CCR1_OFFSET: u8 = 0x34;

pub trait HalTimerRegExt: Sized {
    const TIMER: u8;
    fn timer(
        self,
        freq: fugit::Hertz<u32>,
        cfg: timer::TimerConfig,
        clocks: &Clocks,
    ) -> timer::Timer<Self>;
    fn clock_speed(&self, clocks: &Clocks) -> fugit::Hertz<u32> {
        let timer_freq = match Self::TIMER {
            1 | 8 => clocks.apb2_timer(),
            _ => clocks.apb1_timer(),
        };
        fugit::Hertz::<u32>::from_raw(timer_freq)
    }
}

pub trait HalDmaRegExt: Sized + Deref<Target = pac::dma1::RegisterBlock> + 'static {
    const DMA: dma::DmaPeriph;
    fn dma(self) -> dma::Dma<Self> {
        dma::Dma::new(self)
    }

    fn split(self) -> [DmaChannel<Self>; 8] {
        let mut dma = self.dma();
        let dma_channels = [
            dma::DmaChannel::C0,
            dma::DmaChannel::C1,
            dma::DmaChannel::C2,
            dma::DmaChannel::C3,
            dma::DmaChannel::C4,
            dma::DmaChannel::C5,
            dma::DmaChannel::C6,
            dma::DmaChannel::C7,
        ];
        dma_channels.map(|channel| DmaChannel {
            dma: unsafe {
                &mut *(&mut dma as *mut dma::Dma<Self> as *mut UnsafeCell<dma::Dma<Self>>)
            },
            channel,
        })
    }
}

pub trait HalTimerExt: Sized + Borrow<timer::Timer<Self::HalTimerReg>> {
    type HalTimerReg: HalTimerRegExt;
    const ADVANCED_CONTROL: bool;

    fn overflow(&mut self) -> bool;
    fn read_count(&self) -> u32;
    fn enable(&mut self);
    fn update_psc_arr(&mut self, psc: u16, arr: u32);
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

    fn write<const N: usize, H: HalDmaRegExt>(&mut self, dma_buf: &mut DmaBuf<N, H>) -> bool {
        if dma_buf.transfer_in_progress && !dma_buf.dma_ch.transfer_complete() {
            defmt::warn!("DMA transfer is not complete");
            return false;
        }

        dma_buf.dma_ch.clear_interrupt();
        dma_buf.transfer_in_progress = false;

        let base_addr = TIMX_CCR1_OFFSET / 4;
        let burst_len = 4u8;
        assert_eq!(dma_buf.staging_buf.len() % burst_len as usize, 0);
        let channel_cfg = dma::ChannelCfg::default();

        dma_buf.shared_buf.copy_from_slice(dma_buf.staging_buf);
        dma_buf.transfer_in_progress = true;
        // SAFETY: `dma_buf.shared_buf` is leaked memory, and will remain valid for the lifetime of the program
        // `dma_buf.shared_buf` can only be written to when the DMA transfer is not in progress to prevent data corruption
        unsafe {
            self.hal_write_dma_burst(
                dma_buf.shared_buf,
                base_addr,
                burst_len,
                dma_buf.dma_ch.channel,
                channel_cfg,
                false,
                H::DMA,
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

pub struct DmaChannel<H: HalDmaRegExt> {
    dma: &'static mut UnsafeCell<dma::Dma<H>>,
    channel: dma::DmaChannel,
}

macro_rules! cond {
    (true, $val:expr) => {
        $val
    };
    (false, $val:expr) => {};
}

macro_rules! impl_hal_timer {
    ($tim:literal, $advanced:literal) => {
        paste::paste! {
        impl HalTimerExt for timer::Timer<pac::[<TIM $tim>]>
        {
            type HalTimerReg = pac::[<TIM $tim>];
            const ADVANCED_CONTROL: bool = $advanced;

            fn overflow(&mut self) -> bool {
                let uif = self.regs.sr.read().uif().bit_is_set();
                self.regs.sr.write(|w| w.uif().clear_bit());
                uif
            }
            fn update_psc_arr(&mut self, psc: u16, arr: u32) {
                self.set_prescaler(psc);
                self.set_auto_reload(arr);
                self.reinitialize();
            }
            fn read_count(&self) -> u32 {
                timer::Timer::<pac::[<TIM $tim>]>::read_count(self)
            }
            fn enable(&mut self) {
                timer::Timer::<pac::[<TIM $tim>]>::enable(self)
            }
            fn enable_pwm(&mut self) {
                // if advanced, set MOE bit to enable outputs
                cond!($advanced, self.regs.bdtr.modify(|_, w| w.moe().set_bit()));
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

// Advanced timers
impl_hal_timer!(1, true);
impl_hal_timer!(8, true);

// General purpose timers
impl_hal_timer!(2, false);
impl_hal_timer!(3, false);
impl_hal_timer!(4, false);
impl_hal_timer!(5, false);

macro_rules! impl_hal_timer_reg {
    ($tim_num:literal) => {
        paste::paste! {
        impl HalTimerRegExt for pac::[<TIM $tim_num>] {
            const TIMER: u8 = $tim_num;
            fn timer(
                self,
                freq: fugit::Hertz<u32>,
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
impl_hal_timer_reg!(4);
impl_hal_timer_reg!(5);
impl_hal_timer_reg!(8);

impl HalDmaRegExt for pac::DMA1 {
    const DMA: dma::DmaPeriph = dma::DmaPeriph::Dma1;
}

impl HalDmaRegExt for pac::DMA2 {
    const DMA: dma::DmaPeriph = dma::DmaPeriph::Dma2;
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
impl_timer_dma_mux!(4);

impl<H: HalDmaRegExt> DmaChannel<H> {
    pub fn mux<M: DmaMuxInput>(&self, _: &mut M) {
        dma::mux(H::DMA, self.channel, M::DMA_INPUT);
    }

    pub fn clear_interrupt(&mut self) {
        self.dma
            .get_mut()
            .clear_interrupt(self.channel, dma::DmaInterrupt::TransferComplete);
    }

    pub fn transfer_complete(&mut self) -> bool {
        self.dma.get_mut().transfer_is_complete(self.channel)
    }
}
