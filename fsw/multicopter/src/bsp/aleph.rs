use hal::{
    clocks,
    gpio::{OutputSpeed, OutputType, Pin, PinMode, Port, Pull},
    pac,
};

static mut PINS_TAKEN: bool = false;
pub const HSE_FREQ: fugit::Hertz<u32> = fugit::Hertz::<u32>::MHz(16);

pub struct Pins {
    // I2C1, AF: 4
    pub pb6: Pin, // SCL
    pub pb7: Pin, // SDA

    // I2C2, AF: 4
    pub pf0: Pin, // SDA
    pub pf1: Pin, // SCL

    // I2C4, AF: 4
    pub pf14: Pin, // SCL
    pub pf15: Pin, // SDA

    // TIM1, AF: 1
    pub pe9: Pin,  // CH1
    pub pe11: Pin, // CH2
    pub pe13: Pin, // CH3
    pub pe14: Pin, // CH4

    // TIM2, AF: 1
    pub pa5: Pin, // CH1
    pub pa1: Pin, // CH2
    pub pa2: Pin, // CH3
    pub pa3: Pin, // CH4

    // TIM3, AF: 2
    pub pa6: Pin, // CH1
    pub pa7: Pin, // CH2
    pub pb0: Pin, // CH3
    pub pb1: Pin, // CH4

    // TIM4, AF: 2
    pub pd12: Pin, // CH1
    pub pd13: Pin, // CH2
    pub pd14: Pin, // CH3
    pub pd15: Pin, // CH4

    // USART1, AF: 7
    pub pa9: Pin,  // TX
    pub pa10: Pin, // RX

    // USART2, AF: 7
    pub pd5: Pin, // TX
    pub pd6: Pin, // RX

    // USART3, AF: 7
    pub pd8: Pin, // TX
    pub pd9: Pin, // RX

    // UART4, AF: 8
    pub pb8: Pin, // RX
    pub pb9: Pin, // TX

    // UART7, AF: 11
    pub pb4: Pin, // TX

    // USART6, AF: 7
    pub pg14: Pin, // TX

    // SPI 1, AF: 5
    pub pg11: Pin, // SCK
    pub pd7: Pin,  // MOSI
    pub pg9: Pin,  // MISO
    pub pg10: Pin, // CS

    // SPI 4, AF: 5
    pub pe2: Pin, // SCK
    pub pe6: Pin, // MOSI
    pub pe5: Pin, // MISO
    pub pe4: Pin, // CS

    // CAN 1, AF: 9
    pub pd0: Pin, // RX
    pub pd1: Pin, // TX

    // CAN 2, AF: 9
    pub pb12: Pin, // RX
    pub pb13: Pin, // TX

    // USB, AF: 10
    pub pa11: Pin, // DM
    pub pa12: Pin, // DP

    // SDMMC1, AF: 12
    pub pd2: Pin,  // CMD
    pub pc12: Pin, // CLK
    pub pc8: Pin,  // D0
    pub pc9: Pin,  // D1
    pub pc10: Pin, // D2
    pub pc11: Pin, // D3

    // LEDs
    pub pd11: Pin, // Red 0
    pub pd10: Pin, // Green 0
    pub pb15: Pin, // Blue 0
    pub pb14: Pin, // Orange 0

    // ADC
    pub pf11: Pin, // ADC1 IN2 (current)
    pub pf12: Pin, // ADC1 IN6 (voltage)
    pub pc4: Pin,  // ADC2 IN4 (gpio0)
    pub pc5: Pin,  // ADC2 IN8 (gpio1)
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
        PINS_TAKEN = true;
        let mut pins = Self {
            pb6: Pin::new(Port::B, 6, PinMode::Alt(4)),
            pb7: Pin::new(Port::B, 7, PinMode::Alt(4)),

            pf0: Pin::new(Port::F, 0, PinMode::Alt(4)),
            pf1: Pin::new(Port::F, 1, PinMode::Alt(4)),

            pf14: Pin::new(Port::F, 14, PinMode::Alt(4)),
            pf15: Pin::new(Port::F, 15, PinMode::Alt(4)),

            pe9: Pin::new(Port::E, 9, PinMode::Alt(1)),
            pe11: Pin::new(Port::E, 11, PinMode::Alt(1)),
            pe13: Pin::new(Port::E, 13, PinMode::Alt(1)),
            pe14: Pin::new(Port::E, 14, PinMode::Alt(1)),

            pa5: Pin::new(Port::A, 5, PinMode::Alt(1)),
            pa1: Pin::new(Port::A, 1, PinMode::Alt(1)),
            pa2: Pin::new(Port::A, 2, PinMode::Alt(1)),
            pa3: Pin::new(Port::A, 3, PinMode::Alt(1)),

            pa6: Pin::new(Port::A, 6, PinMode::Alt(2)),
            pa7: Pin::new(Port::A, 7, PinMode::Alt(2)),
            pb0: Pin::new(Port::B, 0, PinMode::Alt(2)),
            pb1: Pin::new(Port::B, 1, PinMode::Alt(2)),

            pd12: Pin::new(Port::D, 12, PinMode::Alt(2)),
            pd13: Pin::new(Port::D, 13, PinMode::Alt(2)),
            pd14: Pin::new(Port::D, 14, PinMode::Alt(2)),
            pd15: Pin::new(Port::D, 15, PinMode::Alt(2)),

            pa9: Pin::new(Port::A, 9, PinMode::Alt(7)),
            pa10: Pin::new(Port::A, 10, PinMode::Alt(7)),

            pd5: Pin::new(Port::D, 5, PinMode::Alt(7)),
            pd6: Pin::new(Port::D, 6, PinMode::Alt(7)),

            pd8: Pin::new(Port::D, 8, PinMode::Alt(7)),
            pd9: Pin::new(Port::D, 9, PinMode::Alt(7)),

            pb8: Pin::new(Port::B, 8, PinMode::Alt(8)),
            pb9: Pin::new(Port::B, 9, PinMode::Alt(8)),

            pb4: Pin::new(Port::B, 4, PinMode::Alt(11)),

            pg14: Pin::new(Port::G, 14, PinMode::Alt(7)),

            pg11: Pin::new(Port::G, 11, PinMode::Alt(5)),
            pd7: Pin::new(Port::D, 7, PinMode::Alt(5)),
            pg9: Pin::new(Port::G, 9, PinMode::Alt(5)),
            pg10: Pin::new(Port::G, 10, PinMode::Alt(5)),

            pe2: Pin::new(Port::E, 2, PinMode::Alt(5)),
            pe6: Pin::new(Port::E, 6, PinMode::Alt(5)),
            pe5: Pin::new(Port::E, 5, PinMode::Alt(5)),
            pe4: Pin::new(Port::E, 4, PinMode::Alt(5)),

            pd0: Pin::new(Port::D, 0, PinMode::Alt(9)),
            pd1: Pin::new(Port::D, 1, PinMode::Alt(9)),

            pb12: Pin::new(Port::B, 12, PinMode::Alt(9)),
            pb13: Pin::new(Port::B, 13, PinMode::Alt(9)),

            pa11: Pin::new(Port::A, 11, PinMode::Alt(10)),
            pa12: Pin::new(Port::A, 12, PinMode::Alt(10)),

            pd2: Pin::new(Port::D, 2, PinMode::Alt(12)),
            pc12: Pin::new(Port::C, 12, PinMode::Alt(12)),
            pc8: Pin::new(Port::C, 8, PinMode::Alt(12)),
            pc9: Pin::new(Port::C, 9, PinMode::Alt(12)),
            pc10: Pin::new(Port::C, 10, PinMode::Alt(12)),
            pc11: Pin::new(Port::C, 11, PinMode::Alt(12)),

            pd11: Pin::new(Port::D, 11, PinMode::Output),
            pd10: Pin::new(Port::D, 10, PinMode::Output),
            pb15: Pin::new(Port::B, 15, PinMode::Output),
            pb14: Pin::new(Port::B, 14, PinMode::Output),
            pf11: Pin::new(Port::F, 11, PinMode::Analog),
            pf12: Pin::new(Port::F, 12, PinMode::Analog),
            pc4: Pin::new(Port::C, 4, PinMode::Analog),
            pc5: Pin::new(Port::C, 5, PinMode::Analog),
        };
        // Enable open drain for I2C
        pins.pb6.output_type(OutputType::OpenDrain);
        pins.pb7.output_type(OutputType::OpenDrain);
        pins.pf0.output_type(OutputType::OpenDrain);
        pins.pf1.output_type(OutputType::OpenDrain);
        pins.pf14.output_type(OutputType::OpenDrain);
        pins.pf15.output_type(OutputType::OpenDrain);

        // For SDIO, set to high speed and enable internal pull-ups for D0-D3 and CMD
        pins.pd2.output_speed(OutputSpeed::High);
        pins.pc12.output_speed(OutputSpeed::High);
        pins.pc8.output_speed(OutputSpeed::High);
        pins.pc9.output_speed(OutputSpeed::High);
        pins.pc10.output_speed(OutputSpeed::High);
        pins.pc11.output_speed(OutputSpeed::High);
        pins.pd2.pull(Pull::Up);
        pins.pc8.pull(Pull::Up);
        pins.pc9.pull(Pull::Up);
        pins.pc10.pull(Pull::Up);
        pins.pc11.pull(Pull::Up);

        // Set CAN pins to very high speed
        pins.pd0.output_speed(OutputSpeed::VeryHigh);
        pins.pd1.output_speed(OutputSpeed::VeryHigh);
        pins.pb12.output_speed(OutputSpeed::VeryHigh);
        pins.pb13.output_speed(OutputSpeed::VeryHigh);

        // Set LED pins to low speed
        pins.pd11.output_speed(OutputSpeed::Low);
        pins.pd10.output_speed(OutputSpeed::Low);
        pins.pb15.output_speed(OutputSpeed::Low);
        pins.pb14.output_speed(OutputSpeed::Low);

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
        pll_src: clocks::PllSrc::Hse(HSE_FREQ.to_Hz()), // 16 MHz
        pll1: clocks::PllCfg {
            enabled: true,
            pllp_en: true,
            pllq_en: true,
            pllr_en: false,
            divm: 2,   // pll_input_speed = 8 MHz
            divn: 100, // vco_speed = 800 MHz
            divp: 2,   // sysclk = 400 MHz
            divq: 16,  // 50 MHz
            ..Default::default()
        },
        input_src: clocks::InputSrc::Pll1, // 400 MHz (sysclk)
        d1_core_prescaler: clocks::HclkPrescaler::Div1, // 400 MHz (M7 core)
        hclk_prescaler: clocks::HclkPrescaler::Div2, // 200 MHz (hclk, M4 core)
        d2_prescaler1: clocks::ApbPrescaler::Div2, // 100 MHz
        d2_prescaler2: clocks::ApbPrescaler::Div2, // 100 MHz
        d3_prescaler: clocks::ApbPrescaler::Div2, // 100 MHz
        vos_range: clocks::VosRange::VOS1,
        can_src: clocks::CanSrc::Hse,
        ..Default::default()
    };

    assert_eq!(clock_cfg.sysclk(), 400_000_000);
    assert_eq!(clock_cfg.d1cpreclk(), 400_000_000);
    assert_eq!(clock_cfg.hclk(), 200_000_000);
    assert_eq!(clock_cfg.apb1(), 100_000_000);
    assert_eq!(clock_cfg.apb2(), 100_000_000);

    clock_cfg
}
