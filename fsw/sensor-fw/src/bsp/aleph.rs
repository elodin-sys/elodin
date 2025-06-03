use hal::{
    clocks,
    gpio::{OutputSpeed, OutputType, Pin, PinMode, Port, Pull},
    pac,
};

static mut PINS_TAKEN: bool = false;
pub const HSE_FREQ: fugit::Hertz<u32> = fugit::Hertz::<u32>::MHz(24);

// Monitor configuration constants
pub mod monitor {
    // Voltage divider for VIN: (147kΩ + 10kΩ) / 10kΩ = 15.7
    pub const VIN_DIVIDER: f32 = 15.7;

    // Voltage divider for VBAT: (147kΩ + 10kΩ) / 10kΩ = 15.7
    pub const VBAT_DIVIDER: f32 = 15.7;

    // Current gain for 5V AUX with TPS2521 and RILM = 1kΩ
    pub const AUX_CURRENT_GAIN: f32 = 5.49; // A/V

    // ADC channel mappings for specific ADCs
    pub const VIN_CHANNEL: u8 = 0; // PC2_C -> ADC3_INP0
    pub const VBAT_CHANNEL: u8 = 0; // PA0_C -> ADC12_INP0
    pub const AUX_CURRENT_CHANNEL: u8 = 1; // PC3_C -> ADC3_INP1
}

pub struct Pins {
    // I2C1, AF: 4
    pub pb8: Pin, // SCL
    pub pb9: Pin, // SDA

    // I2C2, AF: 4
    pub pf1: Pin, // SCL
    pub pf0: Pin, // SDA

    // I2C3, AF: 4
    pub ph7: Pin, // SCL
    pub ph8: Pin, // SDA

    // I2C4, AF: 4
    pub ph11: Pin, // SCL
    pub ph12: Pin, // SDA

    // TIM1
    pub pe9: Pin,  // CH1
    pub pe11: Pin, // CH2
    pub pe13: Pin, // CH3
    pub pe14: Pin, // CH4

    // TIM2
    pub pa5: Pin, // CH1
    pub pb3: Pin, // CH2
    pub pa2: Pin, // CH3
    pub pa3: Pin, // CH4

    // TIM4
    pub pd12: Pin, // CH1
    pub pd13: Pin, // CH2
    pub pd14: Pin, // CH3
    pub pd15: Pin, // CH4

    // TIM8
    pub pi5: Pin, // CH1
    pub pi6: Pin, // CH2
    pub pi7: Pin, // CH3
    pub pi2: Pin, // CH4

    // UART1
    pub pa10: Pin, // RX
    pub pa9: Pin,  // TX

    // UART2
    pub pd6: Pin, // RX
    pub pd5: Pin, // TX

    // UART3
    pub pd9: Pin, // RX
    pub pd8: Pin, // TX

    // UART6
    pub pc7: Pin, // RX
    pub pc6: Pin, // TX

    // UART7
    pub pf6: Pin, // RX
    pub pf7: Pin, // TX

    // UART8
    pub pe0: Pin, // RX
    pub pe1: Pin, // TX

    // SPI1
    pub pg11: Pin, // SCK
    pub pg9: Pin,  // MISO
    pub pd7: Pin,  // MOSI
    pub pg10: Pin, // CS

    // SPI5
    pub pk0: Pin,  // CLK
    pub pj11: Pin, // MISO
    pub pj10: Pin, // MOSI
    pub pk1: Pin,  // CS

    // CAN1
    pub ph14: Pin, // RX
    pub ph13: Pin, // TX

    // CAN2
    pub pb5: Pin, // RX
    pub pb6: Pin, // TX

    // USB_FS
    pub pa11: Pin, // DM
    pub pa12: Pin, // DP

    // USB_HS
    pub pb14: Pin, // DM
    pub pb15: Pin, // DP

    // SDMMC1, AF: 12
    pub pd2: Pin,  // CMD
    pub pc12: Pin, // CLK
    pub pc8: Pin,  // D0
    pub pc9: Pin,  // D1
    pub pc10: Pin, // D2
    pub pc11: Pin, // D3

    // LEDs
    pub red_led: Pin,
    pub green_led: Pin,
    pub blue_led: Pin,
    pub amber_led: Pin,

    // ADC
    pub vmon_vbat: Pin,
    pub vmon_vin: Pin,
    pub imon_5v_aux: Pin,

    // GPIO
    pub gpio: [Pin; 8],

    // AUX
    pub aux_5v_en: Pin,
}

impl Pins {
    pub fn take() -> Option<Self> {
        cortex_m::interrupt::free(|_| {
            if unsafe { PINS_TAKEN } {
                None
            } else {
                Some(unsafe { Self::steal() })
            }
        })
    }

    unsafe fn steal() -> Self {
        unsafe { PINS_TAKEN = true };
        let mut pins = Self {
            // I2C1, AF: 4
            pb8: Pin::new(Port::B, 8, PinMode::Alt(4)),
            pb9: Pin::new(Port::B, 9, PinMode::Alt(4)),

            // I2C2, AF: 4
            pf1: Pin::new(Port::F, 1, PinMode::Alt(4)),
            pf0: Pin::new(Port::F, 0, PinMode::Alt(4)),

            // I2C3, AF: 4
            ph7: Pin::new(Port::H, 7, PinMode::Alt(4)),
            ph8: Pin::new(Port::H, 8, PinMode::Alt(4)),

            // I2C4, AF: 4
            ph11: Pin::new(Port::H, 11, PinMode::Alt(4)),
            ph12: Pin::new(Port::H, 12, PinMode::Alt(4)),

            // TIM1, AF: 1
            pe9: Pin::new(Port::E, 9, PinMode::Alt(1)),
            pe11: Pin::new(Port::E, 11, PinMode::Alt(1)),
            pe13: Pin::new(Port::E, 13, PinMode::Alt(1)),
            pe14: Pin::new(Port::E, 14, PinMode::Alt(1)),

            // TIM2, AF: 1
            pa5: Pin::new(Port::A, 5, PinMode::Alt(1)),
            pb3: Pin::new(Port::B, 3, PinMode::Alt(1)),
            pa2: Pin::new(Port::A, 2, PinMode::Alt(1)),
            pa3: Pin::new(Port::A, 3, PinMode::Alt(1)),

            // TIM4, AF: 2
            pd12: Pin::new(Port::D, 12, PinMode::Alt(2)),
            pd13: Pin::new(Port::D, 13, PinMode::Alt(2)),
            pd14: Pin::new(Port::D, 14, PinMode::Alt(2)),
            pd15: Pin::new(Port::D, 15, PinMode::Alt(2)),

            // TIM8, AF: 3
            pi5: Pin::new(Port::I, 5, PinMode::Alt(3)),
            pi6: Pin::new(Port::I, 6, PinMode::Alt(3)),
            pi7: Pin::new(Port::I, 7, PinMode::Alt(3)),
            pi2: Pin::new(Port::I, 2, PinMode::Alt(3)),

            // UART1, AF: 7
            pa10: Pin::new(Port::A, 10, PinMode::Alt(7)),
            pa9: Pin::new(Port::A, 9, PinMode::Alt(7)),

            // UART2, AF: 7
            pd6: Pin::new(Port::D, 6, PinMode::Alt(7)),
            pd5: Pin::new(Port::D, 5, PinMode::Alt(7)),

            // UART3, AF: 7
            pd9: Pin::new(Port::D, 9, PinMode::Alt(7)),
            pd8: Pin::new(Port::D, 8, PinMode::Alt(7)),

            // UART6, AF: 7
            pc7: Pin::new(Port::C, 7, PinMode::Alt(7)),
            pc6: Pin::new(Port::C, 6, PinMode::Alt(7)),

            // UART7, AF: 7
            pf6: Pin::new(Port::F, 6, PinMode::Alt(7)),
            pf7: Pin::new(Port::F, 7, PinMode::Alt(7)),

            // UART8, AF: 8
            pe0: Pin::new(Port::E, 0, PinMode::Alt(8)),
            pe1: Pin::new(Port::E, 1, PinMode::Alt(8)),

            // SPI1, AF: 5
            pg11: Pin::new(Port::G, 11, PinMode::Alt(5)),
            pg9: Pin::new(Port::G, 9, PinMode::Alt(5)),
            pd7: Pin::new(Port::D, 7, PinMode::Alt(5)),
            pg10: Pin::new(Port::G, 10, PinMode::Alt(5)), // CS pin typically used as GPIO

            // SPI5, AF: 5
            pk0: Pin::new(Port::K, 0, PinMode::Alt(5)),
            pj11: Pin::new(Port::J, 11, PinMode::Alt(5)),
            pj10: Pin::new(Port::J, 10, PinMode::Alt(5)),
            pk1: Pin::new(Port::K, 1, PinMode::Alt(5)), // CS pin typically used as GPIO

            // CAN1, AF: 9
            ph14: Pin::new(Port::H, 14, PinMode::Alt(9)),
            ph13: Pin::new(Port::H, 13, PinMode::Alt(9)),

            // CAN2, AF: 9
            pb5: Pin::new(Port::B, 5, PinMode::Alt(9)),
            pb6: Pin::new(Port::B, 6, PinMode::Alt(9)),

            // USB_FS, AF: 10
            pa11: Pin::new(Port::A, 11, PinMode::Alt(10)),
            pa12: Pin::new(Port::A, 12, PinMode::Alt(10)),

            // USB_HS, AF: 10
            pb14: Pin::new(Port::B, 14, PinMode::Alt(10)),
            pb15: Pin::new(Port::B, 15, PinMode::Alt(10)),

            // SDMMC1, AF: 12
            pd2: Pin::new(Port::D, 2, PinMode::Alt(12)),
            pc12: Pin::new(Port::C, 12, PinMode::Alt(12)),
            pc8: Pin::new(Port::C, 8, PinMode::Alt(12)),
            pc9: Pin::new(Port::C, 9, PinMode::Alt(12)),
            pc10: Pin::new(Port::C, 10, PinMode::Alt(12)),
            pc11: Pin::new(Port::C, 11, PinMode::Alt(12)),

            // LEDs
            red_led: Pin::new(Port::E, 5, PinMode::Output),
            green_led: Pin::new(Port::A, 15, PinMode::Output),
            blue_led: Pin::new(Port::I, 0, PinMode::Output),
            amber_led: Pin::new(Port::I, 9, PinMode::Output),

            // ADC
            vmon_vbat: Pin::new(Port::A, 0, PinMode::Analog),
            vmon_vin: Pin::new(Port::C, 2, PinMode::Analog),
            imon_5v_aux: Pin::new(Port::C, 3, PinMode::Analog),

            // GPIO
            gpio: [
                Pin::new(Port::K, 2, PinMode::Output),
                Pin::new(Port::G, 2, PinMode::Output),
                Pin::new(Port::G, 3, PinMode::Output),
                Pin::new(Port::G, 4, PinMode::Output),
                Pin::new(Port::G, 5, PinMode::Output),
                Pin::new(Port::G, 6, PinMode::Output),
                Pin::new(Port::G, 7, PinMode::Output),
                Pin::new(Port::G, 8, PinMode::Output),
            ],

            // AUX
            aux_5v_en: Pin::new(Port::H, 15, PinMode::Output),
        };
        // Enable open drain for I2C and add pull-ups
        pins.pb8.output_type(OutputType::OpenDrain);
        pins.pb8.pull(Pull::Up);
        pins.pb9.output_type(OutputType::OpenDrain);
        pins.pb9.pull(Pull::Up);
        pins.pf0.output_type(OutputType::OpenDrain);
        pins.pf0.pull(Pull::Up);
        pins.pf1.output_type(OutputType::OpenDrain);
        pins.pf1.pull(Pull::Up);
        pins.ph7.output_type(OutputType::OpenDrain);
        pins.ph7.pull(Pull::Up);
        pins.ph8.output_type(OutputType::OpenDrain);
        pins.ph8.pull(Pull::Up);
        pins.ph11.output_type(OutputType::OpenDrain);
        pins.ph11.pull(Pull::Up);
        pins.ph12.output_type(OutputType::OpenDrain);
        pins.ph12.pull(Pull::Up);

        // Set pull-up for UART pins
        pins.pa10.pull(Pull::Up);
        pins.pd6.pull(Pull::Up);
        pins.pd9.pull(Pull::Up);
        pins.pc7.pull(Pull::Up);
        pins.pf6.pull(Pull::Up);
        pins.pe0.pull(Pull::Up);

        // Set LED pins to low speed
        pins.red_led.output_speed(OutputSpeed::Low);
        pins.green_led.output_speed(OutputSpeed::Low);
        pins.blue_led.output_speed(OutputSpeed::Low);
        pins.amber_led.output_speed(OutputSpeed::Low);

        // Set high speed for SPI
        pins.pg11.output_speed(OutputSpeed::VeryHigh);
        pins.pg9.output_speed(OutputSpeed::VeryHigh);
        pins.pd7.output_speed(OutputSpeed::VeryHigh);
        pins.pk0.output_speed(OutputSpeed::VeryHigh);
        pins.pj11.output_speed(OutputSpeed::VeryHigh);
        pins.pj10.output_speed(OutputSpeed::VeryHigh);

        // Set high speed for SDMMC
        pins.pd2.output_speed(OutputSpeed::VeryHigh);
        pins.pc12.output_speed(OutputSpeed::VeryHigh);
        pins.pc8.output_speed(OutputSpeed::VeryHigh);
        pins.pc9.output_speed(OutputSpeed::VeryHigh);
        pins.pc10.output_speed(OutputSpeed::VeryHigh);
        pins.pc11.output_speed(OutputSpeed::VeryHigh);

        pins.aux_5v_en.set_high();

        pins
    }
}

pub fn clock_cfg(pwr: pac::PWR) -> clocks::Clocks {
    pwr.cr3.modify(|_, w| {
        // Disable bypass, SMPS
        w.bypass().clear_bit();
        w.sden().clear_bit();
        // Enable LDO
        w.ldoen().set_bit()
    });

    let clock_cfg = clocks::Clocks {
        pll_src: clocks::PllSrc::Hse(HSE_FREQ.to_Hz()), // 24 MHz
        pll1: clocks::PllCfg {
            enabled: true,
            pllp_en: true,
            pllq_en: true,
            pllr_en: false,
            divm: 3,   // pll_input_speed = 8 MHz
            divn: 100, // vco_speed = 800 MHz
            divp: 2,   // sysclk = 400 MHz
            divq: 8,   // 100 MHz
            ..Default::default()
        },
        pll2: clocks::PllCfg {
            enabled: true,
            pllp_en: true,
            pllq_en: false,
            pllr_en: false,
            divm: 3,  // pll_input_speed = 8 MHz (24 MHz / 3)
            divn: 25, // vco_speed = 200 MHz (8 MHz * 25)
            divp: 8,  // pll2_p_ck = 25 MHz (200 MHz / 8) - max for 16-bit ADC
            ..Default::default()
        },
        hsi48_on: true,
        input_src: clocks::InputSrc::Pll1, // 400 MHz (sysclk)
        d1_core_prescaler: clocks::HclkPrescaler::Div1, // 400 MHz (M7 core)
        hclk_prescaler: clocks::HclkPrescaler::Div2, // 200 MHz (hclk, M4 core)
        d2_prescaler1: clocks::ApbPrescaler::Div2, // 100 MHz
        d2_prescaler2: clocks::ApbPrescaler::Div2, // 100 MHz
        d3_prescaler: clocks::ApbPrescaler::Div2, // 100 MHz
        vos_range: clocks::VosRange::VOS1,
        can_src: clocks::CanSrc::Hse,
        hse_bypass: true,
        ..Default::default()
    };

    assert_eq!(clock_cfg.sysclk(), 400_000_000);
    assert_eq!(clock_cfg.d1cpreclk(), 400_000_000);
    assert_eq!(clock_cfg.hclk(), 200_000_000);
    assert_eq!(clock_cfg.apb1(), 100_000_000);
    assert_eq!(clock_cfg.apb2(), 100_000_000);

    clock_cfg
}
