use hal::pac;

/// Storage type for the CAN controller
#[derive(Debug)]
pub struct Can<FDCAN = pac::FDCAN1> {
    rb: FDCAN,
}
impl<FDCAN> Can<FDCAN> {
    /// Returns a reference to the inner peripheral
    fn inner(&self) -> &FDCAN {
        &self.rb
    }
}

/// This code is taken from: https://github.com/stm32-rs/stm32h7xx-hal/blob/0735f793e6d432653f2a7f1d8a2d05f073ca4f17/src/can.rs
/// Configure Message RAM layout on H7 to match the fixed sized used on G4
///
/// These are protected bits, write access is only possible when bit CCE and bit
/// INIT for FDCAN_CCCR are set to 1
macro_rules! message_ram_layout {
    ($can:ident, $start_word_addr:expr) => {
        use fdcan::message_ram::*;
        let mut word_adr: u16 = $start_word_addr;

        // 11-bit filter
        $can.sidfc
            .modify(|_, w| unsafe { w.flssa().bits(word_adr) });
        word_adr += STANDARD_FILTER_MAX as u16;
        // 29-bit filter
        $can.xidfc
            .modify(|_, w| unsafe { w.flesa().bits(word_adr) });
        word_adr += 2 * EXTENDED_FILTER_MAX as u16;
        // Rx FIFO 0
        $can.rxf0c.modify(|_, w| unsafe {
            w.f0sa()
                .bits(word_adr)
                .f0s()
                .bits(RX_FIFO_MAX)
                .f0wm()
                .bits(RX_FIFO_MAX)
        });
        word_adr += 18 * RX_FIFO_MAX as u16;
        // Rx FIFO 1
        $can.rxf1c.modify(|_, w| unsafe {
            w.f1sa()
                .bits(word_adr)
                .f1s()
                .bits(RX_FIFO_MAX)
                .f1wm()
                .bits(RX_FIFO_MAX)
        });
        word_adr += 18 * RX_FIFO_MAX as u16;
        // Rx buffer - see below
        // Tx event FIFO
        $can.txefc.modify(|_, w| unsafe {
            w.efsa()
                .bits(word_adr)
                .efs()
                .bits(TX_EVENT_MAX)
                .efwm()
                .bits(TX_EVENT_MAX)
        });
        word_adr += 2 * TX_EVENT_MAX as u16;
        // Tx buffers
        $can.txbc
            .modify(|_, w| unsafe { w.tbsa().bits(word_adr).tfqs().bits(TX_FIFO_MAX) });
        word_adr += 18 * TX_FIFO_MAX as u16;

        // Rx Buffer - not used
        $can.rxbc.modify(|_, w| unsafe { w.rbsa().bits(word_adr) });

        // TX event FIFO?
        // Trigger memory?

        // Set the element sizes to 16 bytes
        $can.rxesc
            .modify(|_, w| unsafe { w.rbds().bits(0b111).f1ds().bits(0b111).f0ds().bits(0b111) });
        $can.txesc.modify(|_, w| unsafe { w.tbds().bits(0b111) });
    };
}

mod fdcan1 {
    use super::Can;
    use hal::pac;

    impl Can<pac::FDCAN1> {
        pub fn fdcan1(rb: pac::FDCAN1, rcc: &pac::RCC) -> fdcan::FdCan<Self, fdcan::ConfigMode> {
            rcc.apb1henr.modify(|_, w| w.fdcanen().set_bit());
            while rcc.apb1henr.read().fdcanen().bit_is_clear() {}

            // Initialisation and RAM layout configuration
            let mut fdcan = fdcan::FdCan::new(Self { rb }).into_config_mode();
            let can = fdcan.instance().inner();
            message_ram_layout!(can, 0x000);
            fdcan
        }
    }
    unsafe impl fdcan::Instance for Can<pac::FDCAN1> {
        const REGISTERS: *mut fdcan::RegisterBlock = pac::FDCAN1::ptr() as *mut _;
    }
    unsafe impl fdcan::message_ram::Instance for Can<pac::FDCAN1> {
        const MSG_RAM: *mut fdcan::message_ram::RegisterBlock = (0x4000_ac00 as *mut _);
    }
}

mod fdcan2 {
    use super::Can;
    use hal::pac;

    impl Can<pac::FDCAN2> {
        pub fn fdcan2(rb: pac::FDCAN2, rcc: &pac::RCC) -> fdcan::FdCan<Self, fdcan::ConfigMode> {
            rcc.apb1henr.modify(|_, w| w.fdcanen().set_bit());
            while rcc.apb1henr.read().fdcanen().bit_is_clear() {}

            // Initialisation and RAM layout configuration
            let mut fdcan = fdcan::FdCan::new(Self { rb }).into_config_mode();
            let can = fdcan.instance().inner();
            message_ram_layout!(can, 0x400); // + 1k words = 4kB

            fdcan
        }
    }
    unsafe impl fdcan::Instance for Can<pac::FDCAN2> {
        const REGISTERS: *mut fdcan::RegisterBlock = pac::FDCAN2::ptr() as *mut _;
    }
    unsafe impl fdcan::message_ram::Instance for Can<pac::FDCAN2> {
        const MSG_RAM: *mut fdcan::message_ram::RegisterBlock = ((0x4000_ac00 + 0x1000) as *mut _); // FDCAN1 + 4kB
    }
}

pub fn setup_can(
    rb: pac::FDCAN1,
    rcc: &pac::RCC,
) -> fdcan::FdCan<Can<pac::FDCAN1>, fdcan::NormalOperationMode> {
    let mut can = Can::fdcan1(rb, rcc);

    // HSE is 16MHz
    // Total bit time = 1 + seg1 + seg2 = 16 time quanta
    // Sample point at (1 + seg1) / (1 + seg1 + seg2) = 14/16 = 87.5%
    let nominal_bit_timing = fdcan::config::NominalBitTiming {
        prescaler: 1.try_into().unwrap(),
        seg1: 13.try_into().unwrap(),
        seg2: 2.try_into().unwrap(),
        sync_jump_width: 1.try_into().unwrap(),
    };
    let data_bit_timing = fdcan::config::DataBitTiming {
        prescaler: 1.try_into().unwrap(),
        seg1: 13.try_into().unwrap(),
        seg2: 2.try_into().unwrap(),
        sync_jump_width: 1.try_into().unwrap(),
        transceiver_delay_compensation: true,
    };
    can.set_protocol_exception_handling(false);
    can.set_nominal_bit_timing(nominal_bit_timing);
    can.set_data_bit_timing(data_bit_timing);
    can.set_standard_filter(
        fdcan::filter::StandardFilterSlot::_0,
        fdcan::filter::StandardFilter::reject_all(),
    );
    can.set_extended_filter(
        fdcan::filter::ExtendedFilterSlot::_0,
        fdcan::filter::ExtendedFilter::accept_all_into_fifo0(),
    );
    can.set_frame_transmit(fdcan::config::FrameTransmissionConfig::AllowFdCanAndBRS);
    // can.enable_interrupt(fdcan::config::Interrupt::RxFifo0NewMsg);
    // can.enable_interrupt_line(fdcan::config::InterruptLine::_0, true);
    can.into_normal()
}
