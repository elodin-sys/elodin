use fugit::MicrosDuration;
use hal::pac;

use crate::{adc, monotonic::Instant};

/// Update rate in Hz for the Monitor
const UPDATE_RATE: u32 = 100;
const UPDATE_PERIOD: MicrosDuration<u32> =
    MicrosDuration::<u32>::from_ticks(1_000_000 / UPDATE_RATE);

/// Monitor for board voltages, currents, and related hardware metrics
///
/// Reads VIN and VBAT voltage levels and 5V AUX current through ADC channels
/// and applies appropriate scaling based on hardware configuration.
pub struct Monitor {
    adc1: adc::Adc<pac::ADC1>,
    adc3: adc::Adc<pac::ADC3>,
    vin_channel: u8,
    vbat_channel: u8,
    aux_current_channel: u8,
    vin_divider: f32,
    vbat_divider: f32,
    current_gain: f32, // Conversion factor in A/V
    next_update: Instant,
    pub data: MonitorData,
}

/// Monitor readings that can be shared with other modules
#[derive(Clone, Copy, defmt::Format)]
pub struct MonitorData {
    pub vin: f32,
    pub vbat: f32,
    pub aux_current: f32,
    pub rtc_vbat: f32,
    pub cpu_temp: f32,
}

impl Default for MonitorData {
    fn default() -> Self {
        Self {
            vin: 0.0,
            vbat: 0.0,
            aux_current: 0.0,
            rtc_vbat: 0.0,
            cpu_temp: 0.0,
        }
    }
}

impl Monitor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        adc1: pac::ADC1,
        adc3: pac::ADC3,
        rcc: &pac::RCC,
        vin_channel: u8,
        vbat_channel: u8,
        aux_current_channel: u8,
        vin_divider: f32,
        vbat_divider: f32,
        current_gain: f32,
    ) -> Self {
        // Calibrate VDDA using VREFINT measurement
        let vdda = adc::calibrate_vdda(rcc);

        // Create and initialize the ADCs with calibrated VDDA
        let adc1 = adc::Adc::new_adc1(adc1, rcc, vdda);
        let adc3 = adc::Adc::new_adc3(adc3, rcc, vdda);

        Self {
            adc1,
            adc3,
            vin_channel,
            vbat_channel,
            aux_current_channel,
            vin_divider,
            vbat_divider,
            current_gain,
            next_update: Instant::from_ticks(0),
            data: MonitorData::default(),
        }
    }

    pub fn update(&mut self, now: Instant) -> bool {
        if now < self.next_update {
            return false;
        }

        // Schedule next update
        self.next_update = now + UPDATE_PERIOD;

        defmt::trace!(
            "Monitor update at {}ms",
            now.duration_since_epoch().to_millis()
        );

        // Simple direct reads - get calibrated voltages
        let vbat_voltage = self.adc1.read(self.vbat_channel);
        let vin_voltage = self.adc3.read(self.vin_channel);
        let aux_voltage = self.adc3.read(self.aux_current_channel);
        let rtc_battery_voltage = self.adc3.read_vbat();
        let cpu_temp = self.adc3.read_temp();

        // Apply the scaling
        self.data.vin = vin_voltage * self.vin_divider;
        self.data.vbat = vbat_voltage * self.vbat_divider;
        self.data.aux_current = aux_voltage * self.current_gain;
        self.data.rtc_vbat = rtc_battery_voltage;
        self.data.cpu_temp = cpu_temp;
        true
    }
}
