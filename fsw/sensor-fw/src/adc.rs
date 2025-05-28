use core::ops::Deref;

use defmt::Format;
use hal::pac;

const VREFINT_CAL_ADDR: *const u16 = 0x1FF1E860 as *const u16;
const TS_CAL1_ADDR: *const u16 = 0x1FF1E820 as *const u16;
const TS_CAL2_ADDR: *const u16 = 0x1FF1E840 as *const u16;
const VREFINT_CHANNEL: u8 = 19;
const VSENSE_CHANNEL: u8 = 18;
const VBAT_CHANNEL: u8 = 17;
const TS_CAL1_TEMP: f32 = 30.0;
const TS_CAL2_TEMP: f32 = 110.0;

unsafe fn read_factory_u16(addr: *const u16) -> u16 {
    unsafe { core::ptr::read_volatile(addr) }
}

#[derive(Clone, Copy, Format)]
pub enum SampleTime {
    Cycles1_5 = 0,
    Cycles2_5 = 1,
    Cycles8_5 = 2,
    Cycles16_5 = 3,
    Cycles32_5 = 4,
    Cycles64_5 = 5,
    Cycles387_5 = 6,
    Cycles810_5 = 7,
}

pub struct Adc<ADC> {
    adc: ADC,
    resolution_bits: u8,
    vdda: f32,
}

/// Calibrate VDDA using VREFINT measurement from ADC3
pub fn calibrate_vdda(rcc: &pac::RCC) -> f32 {
    let vrefint_cal = unsafe { read_factory_u16(VREFINT_CAL_ADDR) };

    rcc.ahb4enr.modify(|_, w| w.adc3en().set_bit());
    rcc.d3ccipr.modify(|_, w| w.adcsel().pll2_p());

    let adc3_common = unsafe { &*pac::ADC3_COMMON::ptr() };
    adc3_common
        .ccr
        .modify(|_, w| w.presc().div1().vrefen().set_bit());

    let adc3 = unsafe { &*pac::ADC3::ptr() };
    adc3.cr.modify(|_, w| w.deeppwd().set_bit());
    adc3.cr
        .modify(|_, w| w.deeppwd().clear_bit().advregen().set_bit());
    cortex_m::asm::delay(1000);
    adc3.cr.modify(|_, w| w.adcal().set_bit());
    while adc3.cr.read().adcal().bit_is_set() {}
    adc3.cfgr.modify(|_, w| w.res().sixteen_bit());
    adc3.smpr2.modify(|_, w| w.smp19().cycles64_5());
    adc3.isr.modify(|_, w| w.adrdy().set_bit());
    adc3.cr.modify(|_, w| w.aden().set_bit());
    while !adc3.isr.read().adrdy().bit_is_set() {}

    adc3.sqr1
        .modify(|_, w| w.sq1().variant(VREFINT_CHANNEL & 0x1f));
    adc3.sqr1.modify(|_, w| w.l().bits(0));
    adc3.cr.modify(|_, w| w.adstart().set_bit());
    while !adc3.isr.read().eoc().bit_is_set() {}
    let vrefint_data = adc3.dr.read().rdata().bits() as u16;

    let vdda = (3.3 * vrefint_cal as f32) / vrefint_data as f32;
    defmt::info!(
        "VREFINT_CAL: {}, VREFINT_DATA: {}, VDDA: {}V",
        vrefint_cal,
        vrefint_data,
        vdda
    );
    vdda
}

impl Adc<pac::ADC1> {
    pub fn new_adc1(adc: pac::ADC1, rcc: &pac::RCC, vdda: f32) -> Self {
        rcc.ahb1enr.modify(|_, w| w.adc12en().set_bit());
        rcc.d3ccipr.modify(|_, w| w.adcsel().pll2_p());

        let adc12_common = unsafe { &*pac::ADC12_COMMON::ptr() };
        adc12_common.ccr.modify(|_, w| w.presc().div1());

        adc.cr.modify(|_, w| w.deeppwd().set_bit());
        adc.cr
            .modify(|_, w| w.deeppwd().clear_bit().advregen().set_bit());
        cortex_m::asm::delay(1000);
        adc.cr.modify(|_, w| w.adcal().set_bit());
        while adc.cr.read().adcal().bit_is_set() {}
        adc.cfgr.modify(|_, w| w.res().sixteen_bit());

        let sample_time = SampleTime::Cycles64_5 as u8;
        adc.smpr1.write(|w| {
            w.smp0()
                .bits(sample_time)
                .smp1()
                .bits(sample_time)
                .smp2()
                .bits(sample_time)
                .smp3()
                .bits(sample_time)
                .smp4()
                .bits(sample_time)
                .smp5()
                .bits(sample_time)
                .smp6()
                .bits(sample_time)
                .smp7()
                .bits(sample_time)
                .smp8()
                .bits(sample_time)
                .smp9()
                .bits(sample_time)
        });
        adc.smpr2.write(|w| {
            w.smp10()
                .bits(sample_time)
                .smp11()
                .bits(sample_time)
                .smp12()
                .bits(sample_time)
                .smp13()
                .bits(sample_time)
                .smp14()
                .bits(sample_time)
                .smp15()
                .bits(sample_time)
                .smp16()
                .bits(sample_time)
                .smp17()
                .bits(sample_time)
                .smp18()
                .bits(sample_time)
                .smp19()
                .bits(sample_time)
        });

        adc.isr.modify(|_, w| w.adrdy().set_bit());
        adc.cr.modify(|_, w| w.aden().set_bit());
        while !adc.isr.read().adrdy().bit_is_set() {}

        Self {
            adc,
            resolution_bits: 16,
            vdda,
        }
    }
}

impl Adc<pac::ADC3> {
    pub fn new_adc3(adc: pac::ADC3, rcc: &pac::RCC, vdda: f32) -> Self {
        rcc.ahb4enr.modify(|_, w| w.adc3en().set_bit());
        rcc.d3ccipr.modify(|_, w| w.adcsel().pll2_p());

        let adc3_common = unsafe { &*pac::ADC3_COMMON::ptr() };
        adc3_common.ccr.modify(|_, w| {
            w.presc()
                .div1()
                .vrefen()
                .set_bit()
                .vsenseen()
                .set_bit()
                .vbaten()
                .set_bit()
        });

        adc.cr.modify(|_, w| w.deeppwd().set_bit());
        adc.cr
            .modify(|_, w| w.deeppwd().clear_bit().advregen().set_bit());
        cortex_m::asm::delay(1000);
        adc.cr.modify(|_, w| w.adcal().set_bit());
        while adc.cr.read().adcal().bit_is_set() {}
        adc.cfgr.modify(|_, w| w.res().sixteen_bit());

        let sample_time = SampleTime::Cycles64_5 as u8;
        adc.smpr1.write(|w| {
            w.smp0()
                .bits(sample_time)
                .smp1()
                .bits(sample_time)
                .smp2()
                .bits(sample_time)
                .smp3()
                .bits(sample_time)
                .smp4()
                .bits(sample_time)
                .smp5()
                .bits(sample_time)
                .smp6()
                .bits(sample_time)
                .smp7()
                .bits(sample_time)
                .smp8()
                .bits(sample_time)
                .smp9()
                .bits(sample_time)
        });
        adc.smpr2.write(|w| {
            w.smp10()
                .bits(sample_time)
                .smp11()
                .bits(sample_time)
                .smp12()
                .bits(sample_time)
                .smp13()
                .bits(sample_time)
                .smp14()
                .bits(sample_time)
                .smp15()
                .bits(sample_time)
                .smp16()
                .bits(sample_time)
                .smp17()
                .bits(sample_time)
                .smp18()
                .bits(sample_time)
                .smp19()
                .bits(sample_time)
        });

        adc.isr.modify(|_, w| w.adrdy().set_bit());
        adc.cr.modify(|_, w| w.aden().set_bit());
        while !adc.isr.read().adrdy().bit_is_set() {}

        Self {
            adc,
            resolution_bits: 16,
            vdda,
        }
    }

    pub fn read_vbat(&mut self) -> f32 {
        self.read(VBAT_CHANNEL) * 4.0
    }

    pub fn read_temp(&mut self) -> f32 {
        let ts_data = self.read_raw(VSENSE_CHANNEL);
        let ts_cal1 = unsafe { read_factory_u16(TS_CAL1_ADDR) };
        let ts_cal2 = unsafe { read_factory_u16(TS_CAL2_ADDR) };

        // Temperature formula from datasheet:
        // Temperature = (TS_CAL2_TEMP - TS_CAL1_TEMP) / (TS_CAL2 - TS_CAL1) * (TS_DATA - TS_CAL1) + TS_CAL1_TEMP
        (TS_CAL2_TEMP - TS_CAL1_TEMP) / (ts_cal2 as f32 - ts_cal1 as f32)
            * (ts_data as f32 - ts_cal1 as f32)
            + TS_CAL1_TEMP
    }
}

impl<ADC> Adc<ADC>
where
    ADC: AdcInstance,
{
    pub fn read_raw(&mut self, channel: u8) -> u16 {
        self.adc.set_sequence_channel(1, channel);
        self.adc.set_sequence_length(1);
        self.adc.start_conversion();
        while !self.adc.is_conversion_complete() {}
        self.adc.read_data() as u16
    }

    pub fn read(&mut self, channel: u8) -> f32 {
        let raw = self.read_raw(channel);
        let slope = 1 << self.resolution_bits;
        (raw as f32 * self.vdda) / slope as f32
    }

    pub fn vdda_calibrated(&self) -> f32 {
        self.vdda
    }
}

/// Trait for ADC peripheral instances
pub trait AdcInstance: Deref<Target = pac::adc1::RegisterBlock> {
    fn set_sequence_channel(&mut self, rank: u8, channel: u8) {
        if rank == 1 {
            self.sqr1.modify(|_, w| w.sq1().variant(channel & 0x1f))
        } else {
            panic!("Only rank 1 is supported for now");
        }
    }

    fn set_sequence_length(&mut self, length: u8) {
        self.sqr1
            .modify(|_, w| w.l().bits((length.saturating_sub(1)) & 0xf));
    }

    fn start_conversion(&mut self) {
        self.cr.modify(|_, w| w.adstart().set_bit());
    }

    fn is_conversion_complete(&self) -> bool {
        self.isr.read().eoc().bit_is_set()
    }

    fn read_data(&self) -> u32 {
        self.dr.read().rdata().bits()
    }
}

impl<R: Deref<Target = pac::adc1::RegisterBlock>> AdcInstance for R {}
