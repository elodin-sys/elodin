use hal::{
    clocks,
    gpio::{OutputType, Pin, PinMode, Port},
    pac,
};

static mut PINS_TAKEN: bool = false;

pub struct Pins {
    // I2C1
    pub pb6: Pin, // SCL
    pub pb7: Pin, // SDA

    // I2C2
    pub pf0: Pin, // SDA
    pub pf1: Pin, // SCL

    // I2C4
    pub pf14: Pin, // SCL
    pub pf15: Pin, // SDA

    // TIM1
    pub pe9: Pin,  // CH1
    pub pe11: Pin, // CH2
    pub pe13: Pin, // CH3
    pub pe14: Pin, // CH4

    // TIM2
    pub pa5: Pin, // CH1
    pub pa1: Pin, // CH2
    pub pa2: Pin, // CH3
    pub pa3: Pin, // CH4

    // TIM3
    pub pa6: Pin, // CH1
    pub pa7: Pin, // CH2
    pub pb0: Pin, // CH3
    pub pb1: Pin, // CH4

    // TIM4
    pub pd12: Pin, // CH1
    pub pd13: Pin, // CH2
    pub pd14: Pin, // CH3
    pub pd15: Pin, // CH4

    // USART1
    pub pa9: Pin,  // TX
    pub pa10: Pin, // RX

    // USART2
    pub pd5: Pin, // TX
    pub pd6: Pin, // RX

    // USART3
    pub pd8: Pin, // TX
    pub pd9: Pin, // RX

    // UART4
    pub pb8: Pin, // RX
    pub pb9: Pin, // TX

    // UART7
    pub pb4: Pin, // TX

    // USART6
    pub pg14: Pin, // TX
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
        };
        pins.pb6.output_type(OutputType::OpenDrain);
        pins.pb7.output_type(OutputType::OpenDrain);
        pins.pf0.output_type(OutputType::OpenDrain);
        pins.pf1.output_type(OutputType::OpenDrain);
        pins.pf14.output_type(OutputType::OpenDrain);
        pins.pf15.output_type(OutputType::OpenDrain);
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
        pll_src: clocks::PllSrc::Hse(16_000_000), // 16 MHz
        pll1: clocks::PllCfg {
            enabled: true,
            pllp_en: true,
            pllq_en: false,
            pllr_en: false,
            divm: 2,   // pll_input_speed = 8 MHz
            divn: 100, // vco_speed = 800 MHz
            divp: 2,   // sysclk = 400 MHz
            ..Default::default()
        },
        input_src: clocks::InputSrc::Pll1, // 400 MHz (sysclk)
        d1_core_prescaler: clocks::HclkPrescaler::Div1, // 400 MHz (M7 core)
        hclk_prescaler: clocks::HclkPrescaler::Div2, // 200 MHz (hclk, M4 core)
        d2_prescaler1: clocks::ApbPrescaler::Div2, // 100 MHz
        d2_prescaler2: clocks::ApbPrescaler::Div2, // 100 MHz
        d3_prescaler: clocks::ApbPrescaler::Div2, // 100 MHz
        vos_range: clocks::VosRange::VOS1,
        ..Default::default()
    };

    assert_eq!(clock_cfg.sysclk(), 400_000_000);
    assert_eq!(clock_cfg.d1cpreclk(), 400_000_000);
    assert_eq!(clock_cfg.hclk(), 200_000_000);
    assert_eq!(clock_cfg.apb1(), 100_000_000);
    assert_eq!(clock_cfg.apb2(), 100_000_000);

    clock_cfg
}
