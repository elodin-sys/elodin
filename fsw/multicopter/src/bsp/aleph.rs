use hal::{clocks, pac};

pub fn clock_cfg(pwr: pac::PWR) -> clocks::Clocks {
    pwr.cr3.modify(|_, w| {
        // Disable bypass, SMPS
        w.bypass().clear_bit();
        w.sden().clear_bit();
        // Enable LDO
        w.ldoen().set_bit()
    });

    let clock_cfg = clocks::Clocks {
        pll_src: clocks::PllSrc::Hse(25_000_000), // 25 MHz
        pll1: clocks::PllCfg {
            enabled: true,
            pllp_en: true,
            pllq_en: false,
            pllr_en: false,
            divm: 5,   // pll_input_speed = 5 MHz
            divn: 192, // vco_speed = 960 MHz
            divp: 2,   // sysclk = 480 MHz
            ..Default::default()
        },
        input_src: clocks::InputSrc::Pll1, // 480 MHz (sysclk)
        d1_core_prescaler: clocks::HclkPrescaler::Div1, // 480 MHz (M7 core)
        hclk_prescaler: clocks::HclkPrescaler::Div2, // 240 MHz (hclk, M4 core)
        d2_prescaler1: clocks::ApbPrescaler::Div2, // 120 MHz
        d2_prescaler2: clocks::ApbPrescaler::Div2, // 120 MHz
        d3_prescaler: clocks::ApbPrescaler::Div2, // 120 MHz
        vos_range: clocks::VosRange::VOS0,
        ..Default::default()
    };

    assert_eq!(clock_cfg.sysclk(), 480_000_000);
    assert_eq!(clock_cfg.d1cpreclk(), 480_000_000);
    assert_eq!(clock_cfg.hclk(), 240_000_000);
    assert_eq!(clock_cfg.apb1(), 120_000_000);
    assert_eq!(clock_cfg.apb2(), 120_000_000);

    clock_cfg
}
