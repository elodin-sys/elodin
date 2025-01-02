use embedded_sdmmc::{Block, BlockCount, BlockDevice, BlockIdx, TimeSource, VolumeManager};
use fugit::RateExtU32 as _;
use hal::{clocks, pac};
use sdio_host::{
    common_cmd::{self, ResponseLen},
    emmc::{CardCapacity, CardStatus, CurrentState, CID, CSD, OCR, RCA},
    sd::{BusWidth, SDStatus, CIC, SCR, SD},
    sd_cmd, Cmd,
};

type Hertz = fugit::Hertz<u32>;

const SD_INIT_FREQ: Hertz = Hertz::kHz(400);
const SDR12_MAX_FREQ: Hertz = Hertz::MHz(25);
const SD_NORMAL_FREQ: Hertz = Hertz::MHz(50);
const SD_KERNEL_FREQ: Hertz = Hertz::MHz(50);

#[derive(defmt::Format, Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum SdCardSignaling {
    SDR12,
    #[default]
    SDR25,
    SDR50,
    SDR104,
    DDR50,
}

#[derive(defmt::Format, Debug, Copy, Clone, PartialEq, Eq)]
pub enum Error {
    Timeout,
    SoftwareTimeout,
    UnsupportedCardVersion,
    UnsupportedCardType,
    Crc,
    DataCrcFail,
    RxOverFlow,
    TxUnderFlow,
    NoCard,
    BadClock,
    InvalidConfiguration,
    SignalingSwitchFailed,
}

impl Error {
    pub fn timeout(&self) -> bool {
        matches!(self, Error::Timeout | Error::SoftwareTimeout)
    }
}

#[derive(Clone, Copy, Default)]
pub struct SdCard {
    pub capacity: CardCapacity,
    pub rca: RCA<SD>,
    pub ocr: OCR<SD>,
    pub cid: CID<SD>,
    pub csd: CSD<SD>,
    pub scr: SCR,
    pub status: SDStatus,
}

impl SdCard {
    pub fn size(&self) -> u64 {
        self.csd.card_size()
    }
}

enum Dir {
    CardToHost,
    HostToCard,
}

enum PowerControl {
    Off = 0b00,
    On = 0b11,
}

pub struct Sdmmc {
    rb: pac::SDMMC1,
    clk: Hertz,
    card: Option<SdCard>,
}

impl Sdmmc {
    pub fn new(rcc: &pac::RCC, rb: pac::SDMMC1, clocks: &clocks::Clocks) -> Result<Self, Error> {
        // Validate kernel frequency
        let pll_src_freq = match clocks.pll_src {
            clocks::PllSrc::Hse(freq) => freq,
            _ => return Err(Error::BadClock),
        };
        let clocks::PllCfg {
            divm,
            divn,
            divq,
            enabled,
            pllq_en,
            ..
        } = clocks.pll1;
        if !enabled || !pllq_en {
            return Err(Error::BadClock);
        }
        let pll1_q: fugit::Hertz<u32> = pll_src_freq.Hz() / divm as u32 * divn as u32 / divq as u32;
        if pll1_q != SD_KERNEL_FREQ {
            return Err(Error::BadClock);
        }

        // Select PLL1Q as SDMMC kernel clock
        rcc.d1ccipr.modify(|_, w| w.sdmmcsel().pll1_q());

        // Enable and reset peripheral
        rcc.ahb3enr.modify(|_, w| w.sdmmc1en().set_bit());
        rcc.ahb3rstr.modify(|_, w| w.sdmmc1rst().set_bit());
        rcc.ahb3rstr.modify(|_, w| w.sdmmc1rst().clear_bit());

        let (clkdiv, clk) = Self::clk_div(SD_INIT_FREQ.to_Hz())?;
        defmt::debug!("CLKDIV: {}, SDMMC clock: {}Hz", clkdiv, clk.to_Hz());

        // Configure SDMMC clock
        rb.clkcr.write(|w| unsafe {
            w.widbus()
                .bits(0) // Set bus width to 1
                .clkdiv()
                .bits(clkdiv) // 400kHz
                .pwrsav()
                .clear_bit() // Power saving disabled
                .negedge()
                .clear_bit()
                .hwfc_en()
                .set_bit()
        });

        let mut sdmmc = Sdmmc {
            rb,
            clk,
            card: None,
        };

        sdmmc.power_off();
        sdmmc.try_connect()?;
        Ok(sdmmc)
    }

    fn clk_div(sdmmc_ck: u32) -> Result<(u16, Hertz), Error> {
        match SD_KERNEL_FREQ.raw().div_ceil(sdmmc_ck) {
            0 | 1 => Ok((0, SD_KERNEL_FREQ)),
            x @ 2..=2046 => {
                let clk_div = ((x + 1) / 2) as u16;
                let clk = Hertz::from_raw(SD_KERNEL_FREQ.raw() / (clk_div as u32 * 2));

                Ok((clk_div, clk))
            }
            _ => Err(Error::BadClock),
        }
    }

    fn power_off(&mut self) {
        defmt::debug!("Powering off SDMMC");
        self.rb
            .power
            .modify(|_, w| unsafe { w.pwrctrl().bits(PowerControl::Off as u8) });
    }

    fn power_on(&mut self) {
        defmt::debug!("Powering on SDMMC");
        self.rb
            .power
            .modify(|_, w| unsafe { w.pwrctrl().bits(PowerControl::On as u8) });
    }

    fn cmd<R: common_cmd::Resp>(&self, cmd: Cmd<R>) -> Result<(), Error> {
        // Clear interrupts
        self.rb.icr.modify(|_, w| {
            w.ccrcfailc() // CRC FAIL
                .set_bit()
                .ctimeoutc() // TIMEOUT
                .set_bit()
                .cmdrendc() // CMD R END
                .set_bit()
                .cmdsentc() // CMD SENT
                .set_bit()
                .dataendc()
                .set_bit()
                .dbckendc()
                .set_bit()
                .dcrcfailc()
                .set_bit()
                .dtimeoutc()
                .set_bit()
                .sdioitc() // SDIO IT
                .set_bit()
                .rxoverrc()
                .set_bit()
                .txunderrc()
                .set_bit()
        });

        // CP state machine must be idle
        while self.rb.star.read().cpsmact().bit_is_set() {}

        // Command arg
        self.rb.argr.write(|w| unsafe { w.cmdarg().bits(cmd.arg) });

        // Determine what kind of response the CPSM should wait for
        let waitresp = match cmd.response_len() {
            ResponseLen::Zero => 0,
            ResponseLen::R48 => 1, // short response, expect CMDREND or CCRCFAIL
            ResponseLen::R136 => 3, // long response, expect CMDREND or CCRCFAIL
        };

        // Special mode in CP State Machine
        // CMD12: Stop Transmission
        let cpsm_stop_transmission = cmd.cmd == 12;

        // Command index and start CP State Machine
        self.rb.cmdr.write(|w| unsafe {
            w.waitint()
                .clear_bit()
                .waitresp() // No / Short / Long
                .bits(waitresp)
                .cmdstop() // CPSM Stop Transmission
                .bit(cpsm_stop_transmission)
                .cmdindex()
                .bits(cmd.cmd)
                .cpsmen()
                .set_bit()
        });

        let mut timeout: u32 = 0xFFFF_FFFF;

        let mut status;
        if cmd.response_len() == ResponseLen::Zero {
            // Wait for CMDSENT or a timeout
            while {
                status = self.rb.star.read();
                !(status.ctimeout().bit() || status.cmdsent().bit()) && timeout > 0
            } {
                timeout -= 1;
            }
        } else {
            // Wait for CMDREND or CCRCFAIL or a timeout
            while {
                status = self.rb.star.read();
                !(status.ctimeout().bit() || status.cmdrend().bit() || status.ccrcfail().bit())
                    && timeout > 0
            } {
                timeout -= 1;
            }
        }

        if status.ctimeout().bit_is_set() {
            return Err(Error::Timeout);
        } else if timeout == 0 {
            return Err(Error::SoftwareTimeout);
        } else if status.ccrcfail().bit() {
            return Err(Error::Crc);
        }

        Ok(())
    }

    fn card_rca(&self) -> u16 {
        self.card.as_ref().map_or(0, |c| c.rca.address())
    }

    fn app_cmd<R: common_cmd::Resp>(&self, acmd: Cmd<R>) -> Result<(), Error> {
        self.cmd(common_cmd::app_cmd(self.card_rca()))?;
        self.cmd(acmd)
    }

    fn read_sd_status(&mut self) -> Result<(), Error> {
        // Prepare the transfer
        self.start_datapath_transfer(64, 6, Dir::CardToHost);
        self.app_cmd(common_cmd::card_status(self.card_rca(), false))?; // ACMD13

        let mut status = [0u32; 16];
        let mut idx = 0;
        let mut sta_reg;
        while {
            sta_reg = self.rb.star.read();
            !(sta_reg.rxoverr().bit()
                || sta_reg.dcrcfail().bit()
                || sta_reg.dtimeout().bit()
                || sta_reg.dbckend().bit())
        } {
            if sta_reg.rxfifohf().bit() {
                for _ in 0..8 {
                    status[15 - idx] = u32::from_be(self.rb.fifor.read().bits());
                    idx += 1;
                }
            }

            if idx == status.len() {
                break;
            }
        }
        self.status_to_result()?;
        let card = self.card.as_mut().ok_or(Error::NoCard)?;
        card.status = status.into();

        let bus_width = match card.status.bus_width() {
            BusWidth::One => 1,
            BusWidth::Four => 4,
            BusWidth::Eight => 8,
            _ => return Err(Error::InvalidConfiguration),
        };
        defmt::debug!("SD card bus width: {}", bus_width);
        defmt::debug!("SD card secured mode: {}", card.status.secure_mode());
        defmt::debug!("SD card type: {}", card.status.sd_memory_card_type());
        defmt::debug!(
            "SD card protected area size: {}",
            card.status.protected_area_size()
        );
        defmt::debug!("SD card speed class: {}", card.status.speed_class());
        defmt::debug!(
            "SD card video speed class: {}",
            card.status.video_speed_class()
        );
        defmt::debug!(
            "SD card application performance class: {}",
            card.status.app_perf_class()
        );
        defmt::debug!(
            "SD card move performance (MB/s): {}",
            card.status.move_performance()
        );
        defmt::debug!(
            "SD card allocation unit size: {}",
            card.status.allocation_unit_size()
        );
        defmt::debug!(
            "SD card erase size (units of AU): {}",
            card.status.erase_size()
        );
        defmt::debug!("SD card erase timeout (s): {}", card.status.erase_timeout());
        defmt::debug!("SD card discard support: {}", card.status.discard_support());
        defmt::debug!("SD card size (MB): {}", card.size() / 1_000_000);
        defmt::debug!(
            "SD card blocks: {}",
            card.size() / embedded_sdmmc::Block::LEN as u64
        );

        Ok(())
    }

    fn status_to_result(&self) -> Result<(), Error> {
        let status = self.rb.star.read();
        if status.dcrcfail().bit() {
            return Err(Error::DataCrcFail);
        } else if status.rxoverr().bit() {
            return Err(Error::RxOverFlow);
        } else if status.txunderr().bit() {
            return Err(Error::TxUnderFlow);
        } else if status.dtimeout().bit() {
            return Err(Error::Timeout);
        }
        Ok(())
    }

    fn start_datapath_transfer(&self, length_bytes: u32, block_size: u8, direction: Dir) {
        assert!(block_size <= 14, "Block size up to 2^14 bytes");

        // Block Size must be greater than 0 ( != 1 byte) in DDR mode
        let ddr = self.rb.clkcr.read().ddr().bit_is_set();
        assert!(
            !ddr || block_size != 0,
            "Block size must be >= 1, or >= 2 in DDR mode"
        );

        let dtdir = matches!(direction, Dir::CardToHost);

        // Command AND Data state machines must be idle
        while self.rb.star.read().dpsmact().bit_is_set()
            || self.rb.star.read().cpsmact().bit_is_set()
        {}

        // Data timeout, in bus cycles
        self.rb
            .dtimer
            .write(|w| unsafe { w.datatime().bits(5_000_000) });
        // Data length, in bytes
        self.rb
            .dlenr
            .write(|w| unsafe { w.datalength().bits(length_bytes) });
        // Transfer
        self.rb.dctrl.write(|w| unsafe {
            w.dblocksize()
                .bits(block_size) // 2^n bytes block size
                .dtdir()
                .bit(dtdir)
                .dten()
                .set_bit() // Enable transfer
        });
    }

    pub fn read_block(&self, address: u32, buffer: &mut [u8; 512]) -> Result<(), Error> {
        self.card.ok_or(Error::NoCard)?;
        self.start_datapath_transfer(512, 9, Dir::CardToHost);
        self.cmd(common_cmd::read_single_block(address))?;

        let mut i = 0;
        let mut status;
        while {
            status = self.rb.star.read();
            !(status.rxoverr().bit()
                || status.dcrcfail().bit()
                || status.dtimeout().bit()
                || status.dataend().bit())
        } {
            if status.rxfifohf().bit() {
                for _ in 0..8 {
                    let bytes = self.rb.fifor.read().bits().to_le_bytes();
                    buffer[i..i + 4].copy_from_slice(&bytes);
                    i += 4;
                }
            }

            if i >= buffer.len() {
                break;
            }
        }
        Ok(())
    }

    pub fn write_block(&self, address: u32, buffer: &[u8; 512]) -> Result<(), Error> {
        self.write_blocks(address, buffer)
    }

    pub fn write_blocks(&self, address: u32, buffer: &[u8]) -> Result<(), Error> {
        self.write_blocks_begin(address, buffer.len())?;
        self.write_blocks_feed(buffer);
        self.write_blocks_conclude()?;
        Ok(())
    }

    fn write_blocks_begin(&self, address: u32, buffer_len: usize) -> Result<(), Error> {
        self.card.ok_or(Error::NoCard)?;
        assert!(
            buffer_len % 512 == 0,
            "Buffer length must be a multiple of 512"
        );
        let n_blocks = buffer_len / 512;
        self.start_datapath_transfer(512 * n_blocks as u32, 9, Dir::HostToCard);
        self.cmd(common_cmd::write_multiple_blocks(address))?; // CMD25
        Ok(())
    }

    fn write_blocks_feed(&self, buffer: &[u8]) {
        let mut i = 0;
        let mut status;
        while {
            status = self.rb.star.read();
            !(status.txunderr().bit()
                || status.dcrcfail().bit()
                || status.dtimeout().bit()
                || status.dataend().bit())
        } {
            if status.txfifohe().bit() {
                for _ in 0..8 {
                    let mut wb = [0u8; 4];
                    wb.copy_from_slice(&buffer[i..i + 4]);
                    let word = u32::from_le_bytes(wb);
                    self.rb.fifor.write(|w| unsafe { w.bits(word) });
                    i += 4;
                }
            }

            if i >= buffer.len() {
                break;
            }
        }
    }

    fn read_status(&self) -> Result<CardStatus<Self>, Error> {
        self.cmd(common_cmd::card_status(self.card_rca(), false))?;
        Ok(CardStatus::from(self.rb.resp1r.read().bits()))
    }

    fn wait_card_ready(&self) -> Result<(), Error> {
        let mut timeout: u32 = 0xFFFF_FFFF;
        // Try to read card status (CMD13)
        while timeout > 0 {
            if self.card_ready()? {
                return Ok(());
            }
            timeout -= 1;
        }
        Err(Error::SoftwareTimeout)
    }

    fn card_ready(&self) -> Result<bool, Error> {
        Ok(self.read_status()?.state() == CurrentState::Transfer)
    }

    fn clear_static_interrupt_flags(&self) {
        self.rb.icr.modify(|_, w| {
            w.dcrcfailc()
                .set_bit()
                .dtimeoutc()
                .set_bit()
                .txunderrc()
                .set_bit()
                .rxoverrc()
                .set_bit()
                .dataendc()
                .set_bit()
                .dholdc()
                .set_bit()
                .dbckendc()
                .set_bit()
                .dabortc()
                .set_bit()
                .idmatec()
                .set_bit()
                .idmabtcc()
                .set_bit()
        });
    }

    fn write_blocks_conclude(&self) -> Result<(), Error> {
        let mut status;
        while {
            status = self.rb.star.read();
            !(status.txunderr().bit()
                || status.dcrcfail().bit()
                || status.dtimeout().bit()
                || status.dataend().bit())
        } {}
        self.cmd(common_cmd::stop_transmission())?; // CMD12
        self.status_to_result()?;
        self.clear_static_interrupt_flags();
        self.wait_card_ready()
    }

    fn get_scr(&self, rca: u16) -> Result<SCR, Error> {
        self.cmd(common_cmd::app_cmd(rca))?;
        self.start_datapath_transfer(8, 3, Dir::CardToHost);
        self.cmd(sd_cmd::send_scr())?;

        let mut scr = [0; 2];
        let mut i = 0;
        let mut status;
        while {
            status = self.rb.star.read();

            !(status.rxoverr().bit()
                || status.dcrcfail().bit()
                || status.dtimeout().bit()
                || status.dbckend().bit())
        } {
            if status.rxfifoe().bit_is_clear() {
                // FIFO not empty
                scr[i] = self.rb.fifor.read().bits();
                i += 1;
            }

            if i == 2 {
                break;
            }
        }

        self.status_to_result()?;
        Ok(SCR::from(scr))
    }

    fn try_connect(&mut self) -> Result<Option<SdCard>, Error> {
        self.power_on();
        self.cmd(common_cmd::idle())?;

        match self.cmd(sd_cmd::send_if_cond(1, 0xAA)) {
            Ok(_) => {
                defmt::debug!("SD card detected");
            }
            Err(Error::Timeout) => {
                defmt::debug!("No SD card detected");
                self.power_off();
                return Ok(None);
            }
            Err(err) => return Err(err),
        }
        let cic = CIC::from(self.rb.resp1r.read().bits());
        if cic.pattern() != 0xAA {
            return Err(Error::UnsupportedCardVersion);
        };

        defmt::debug!("Initializing SD card");
        let ocr = loop {
            // Host supports 3.2V-3.3V
            let host_high_capacity_support = true;
            let sdxc_power_control = false;
            let switch_to_1_8v_request = true;
            let voltage_window = 1 << 5;
            self.app_cmd(sd_cmd::sd_send_op_cond(
                host_high_capacity_support,
                sdxc_power_control,
                switch_to_1_8v_request,
                voltage_window,
            ))
            .ignore_crc()?;
            let ocr = OCR::from(self.rb.resp1r.read().bits());
            if !ocr.is_busy() {
                break ocr;
            }
        };

        self.cmd(common_cmd::all_send_cid())?; // CMD2
        let cid = ((self.rb.resp1r.read().bits() as u128) << 96)
            | ((self.rb.resp2r.read().bits() as u128) << 64)
            | ((self.rb.resp3r.read().bits() as u128) << 32)
            | self.rb.resp4r.read().bits() as u128;

        self.cmd(sd_cmd::send_relative_address())?;
        let rca = RCA::from(self.rb.resp1r.read().bits());

        self.cmd(common_cmd::send_csd(rca.address()))?;
        let csd = ((self.rb.resp1r.read().bits() as u128) << 96)
            | ((self.rb.resp2r.read().bits() as u128) << 64)
            | ((self.rb.resp3r.read().bits() as u128) << 32)
            | self.rb.resp4r.read().bits() as u128;

        self.cmd(common_cmd::select_card(rca.address()))
            .ignore_timeout()?;
        let scr = self.get_scr(rca.address())?;

        self.card = Some(SdCard {
            ocr,
            cid: cid.into(),
            csd: csd.into(),
            rca,
            scr,
            ..Default::default()
        });

        // Set bus width to 4
        defmt::debug!("Setting SD card bus width to 4, frequency to 50MHz");
        self.app_cmd(sd_cmd::cmd6(2))?;
        while self.rb.star.read().dpsmact().bit_is_set()
            || self.rb.star.read().cpsmact().bit_is_set()
        {}
        self.rb.clkcr.modify(|_, w| unsafe { w.widbus().bits(1) });

        // Set frequency to maximum SDR12 frequency
        self.set_freq(SDR12_MAX_FREQ)?;
        self.read_sd_status()?;

        // Switch to SDR25 signaling
        let signaling = self.switch_signaling_mode(SdCardSignaling::SDR25)?;
        defmt::debug!("Switched to {:?}", signaling);
        self.set_freq(SD_NORMAL_FREQ)?;
        if signaling != SdCardSignaling::SDR25 || !self.card_ready()? {
            return Err(Error::SignalingSwitchFailed);
        }
        self.read_sd_status()?;

        Ok(self.card)
    }

    fn set_freq(&mut self, freq: Hertz) -> Result<(), Error> {
        let (clkdiv, clk) = Self::clk_div(freq.to_Hz())?;
        defmt::debug!("CLKDIV: {}, SDMMC clock: {}Hz", clkdiv, clk.to_Hz());
        self.rb
            .clkcr
            .modify(|_, w| unsafe { w.clkdiv().bits(clkdiv) });
        self.clk = clk;
        Ok(())
    }

    fn switch_signaling_mode(&self, signaling: SdCardSignaling) -> Result<SdCardSignaling, Error> {
        // NB PLSS v7_10 4.3.10.4: "the use of SET_BLK_LEN command is not
        // necessary"

        let set_function = 0x8000_0000
            | match signaling {
                // See PLSS v7_10 Table 4-11
                SdCardSignaling::DDR50 => 0xFF_FF04,
                SdCardSignaling::SDR104 => 0xFF_1F03,
                SdCardSignaling::SDR50 => 0xFF_1F02,
                SdCardSignaling::SDR25 => 0xFF_FF01,
                SdCardSignaling::SDR12 => 0xFF_FF00,
            };

        // Prepare the transfer
        self.start_datapath_transfer(64, 6, Dir::CardToHost);
        self.cmd(sd_cmd::cmd6(set_function))?; // CMD6

        let mut status = [0u32; 16];
        let mut idx = 0;
        let mut sta_reg;
        while {
            sta_reg = self.rb.star.read();
            !(sta_reg.rxoverr().bit()
                || sta_reg.dcrcfail().bit()
                || sta_reg.dtimeout().bit()
                || sta_reg.dbckend().bit())
        } {
            if sta_reg.rxfifohf().bit() {
                for _ in 0..8 {
                    status[idx] = self.rb.fifor.read().bits();
                    idx += 1;
                }
            }

            if idx == status.len() {
                break;
            }
        }

        self.status_to_result()?;

        // Host is allowed to use the new functions at least 8
        // clocks after the end of the switch command
        // transaction. We know the current clock period is < 80ns,
        // so a total delay of 640ns is required here
        for _ in 0..300 {
            cortex_m::asm::nop();
        }

        // Support Bits of Functions in Function Group 1
        let _support_bits = u32::from_be(status[3]) >> 16;
        // Function Selection of Function Group 1
        let selection = (u32::from_be(status[4]) >> 24) & 0xF;

        match selection {
            0 => Ok(SdCardSignaling::SDR12),
            1 => Ok(SdCardSignaling::SDR25),
            2 => Ok(SdCardSignaling::SDR50),
            3 => Ok(SdCardSignaling::SDR104),
            4 => Ok(SdCardSignaling::DDR50),
            _ => Err(Error::UnsupportedCardType),
        }
    }

    pub fn connected(&self) -> bool {
        self.card.is_some()
    }

    pub fn disconnect(&mut self) {
        self.power_off();
        self.card = None;
    }

    pub fn volume_manager<T: TimeSource>(self, time_source: T) -> VolumeManager<Self, T> {
        VolumeManager::new(self, time_source)
    }
}

impl BlockDevice for Sdmmc {
    type Error = Error;

    fn read(
        &self,
        blocks: &mut [Block],
        start_block_idx: BlockIdx,
        _reason: &str,
    ) -> Result<(), Self::Error> {
        let start = start_block_idx.0;
        for block_idx in start..(start + blocks.len() as u32) {
            self.read_block(
                block_idx,
                &mut blocks[(block_idx - start) as usize].contents,
            )?;
        }
        Ok(())
    }

    fn write(&self, blocks: &[Block], start_block_idx: BlockIdx) -> Result<(), Self::Error> {
        let start = start_block_idx.0;
        let total_length = embedded_sdmmc::Block::LEN * blocks.len();
        self.write_blocks_begin(start, total_length)?;
        for block in blocks.iter() {
            self.write_blocks_feed(&block.contents);
        }
        self.write_blocks_conclude()?;
        Ok(())
    }

    fn num_blocks(&self) -> Result<BlockCount, Self::Error> {
        let card = self.card.ok_or(Error::NoCard)?;
        let blocks = card.size() / embedded_sdmmc::Block::LEN as u64;
        Ok(BlockCount(blocks as u32))
    }
}

trait ResultExt {
    fn ignore_timeout(self) -> Self;
    fn ignore_crc(self) -> Self;
}

impl ResultExt for Result<(), Error> {
    fn ignore_timeout(self) -> Self {
        match self {
            Err(Error::Timeout) => Ok(()),
            res => res,
        }
    }

    fn ignore_crc(self) -> Self {
        match self {
            Err(Error::Crc) => Ok(()),
            res => res,
        }
    }
}
