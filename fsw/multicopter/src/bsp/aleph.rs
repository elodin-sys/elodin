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
