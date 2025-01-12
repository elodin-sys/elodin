use core::ops::Deref;

use fugit::{ExtU32, RateExtU32 as _};
use hal::{clocks, pac};
use sdio_host::{
    common_cmd::{self, ResponseLen},
    emmc::{CardCapacity, CardStatus, CurrentState, CID, CSD, OCR, RCA},
    sd::{BusWidth, SDStatus, CIC, SCR, SD},
    sd_cmd, Cmd,
};
use zerocopy::{FromBytes, Immutable, IntoBytes};

use crate::blackbox::SdmmcFs;
use crate::dwt::DwtTimer;

const DMA_BUF_SIZE: usize = 512 * 64; // 32KB

type Hertz = fugit::Hertz<u32>;

const SD_INIT_FREQ: Hertz = Hertz::kHz(400);
const SDR12_MAX_FREQ: Hertz = Hertz::MHz(25);
const SD_NORMAL_FREQ: Hertz = Hertz::MHz(50);
const SD_KERNEL_FREQ: Hertz = Hertz::MHz(100);

const BLOCK_LEN: usize = 512;

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
    UnexpectedEof,
    WriteZero,
    FatFsNotFound,
    Busy,
    WriteCacheFull,
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
    pub fatfs_info: FatFsInfo,
}

#[derive(Clone, Copy, Default, defmt::Format)]
pub struct FatFsInfo {
    start_lba: u64,
    sector_size: u64,
    num_sectors: u64,
    cursor: u64,
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

#[derive(Clone)]
pub struct Sdmmc {
    rb: &'static pac::sdmmc1::RegisterBlock,
    card: Option<SdCard>,
    dwt: DwtTimer,
    block_cache: FifoBlockCache,
    dma_buf: alloc::boxed::Box<[u8]>,
}

#[derive(Default, Clone)]
struct FifoBlockCache {
    blocks: [Block; 8],
    next: usize,
}

#[derive(Clone)]
struct Block {
    index: u32,
    data: [u8; 512],
    dirty: bool,
    stale: bool,
}

impl Default for Block {
    fn default() -> Self {
        Block {
            index: 0,
            data: [0; 512],
            dirty: false,
            stale: true,
        }
    }
}

impl Block {
    fn needs_writeback(&self) -> bool {
        self.dirty && !self.stale
    }
}

impl FifoBlockCache {
    fn clear(&mut self) {
        self.blocks.iter_mut().for_each(|b| b.stale = true);
    }

    fn get(&self, index: u32) -> Option<&[u8; 512]> {
        self.blocks
            .iter()
            .filter(|b| !b.stale)
            .find(|b| b.index == index)
            .map(|b| &b.data)
    }

    fn get_mut(&mut self, index: u32) -> Option<&mut [u8; 512]> {
        self.blocks
            .iter_mut()
            .filter(|b| !b.stale)
            .find(|b| b.index == index)
            .map(|b| {
                b.dirty = true;
                &mut b.data
            })
    }
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

        let (clkdiv, clk) = Self::clk_div(SD_INIT_FREQ)?;
        defmt::debug!("CLKDIV: {}, SDMMC clock: {}kHz", clkdiv, clk.to_kHz());

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
                .set_bit() // Hardware flow control enabled
                .busspeed()
                .clear_bit() // Normal speed
        });

        Ok(Sdmmc {
            rb: unsafe { &*(rb.deref() as *const _) },
            card: None,
            dwt: DwtTimer::new(clocks),
            block_cache: FifoBlockCache::default(),
            dma_buf: alloc::vec![0u8; DMA_BUF_SIZE].into_boxed_slice(),
        })
    }

    fn clk_div(sdmmc_ck: Hertz) -> Result<(u16, Hertz), Error> {
        match SD_KERNEL_FREQ.raw().div_ceil(sdmmc_ck.raw()) {
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

    fn cmd<R: common_cmd::Resp>(&self, command: Cmd<R>) -> Result<(), Error> {
        cmd(self.rb, command)
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
            sta_reg = self.rb.star.read().to_result()?;
            sta_reg.dbckend().bit_is_clear()
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
        let card = self.card.as_mut().ok_or(Error::NoCard)?;
        card.status = status.into();

        Ok(())
    }

    fn log_sd_status(&self) -> Result<(), Error> {
        let card = self.card.as_ref().ok_or(Error::NoCard)?;
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
        defmt::debug!("SD card blocks: {}", card.size() / BLOCK_LEN as u64);
        Ok(())
    }

    fn start_datapath_transfer(&self, len: u32, block_size: u8, direction: Dir) {
        let dtdir = matches!(direction, Dir::CardToHost);

        // Command AND Data state machines must be idle
        while self.rb.star.read().dpsmact().bit_is_set()
            || self.rb.star.read().cpsmact().bit_is_set()
        {}

        self.rb.dtimer.write(|w| w.datatime().variant(0xFFFF_FFFF));
        self.rb.dlenr.write(|w| w.datalength().variant(len));
        self.rb.dctrl.write(|w| unsafe {
            w.dblocksize()
                .bits(block_size) // 2^n bytes block size
                .dtdir()
                .bit(dtdir)
                .dtmode()
                .variant(0)
                .dten()
                .set_bit() // Enable transfer
        });
    }

    fn start_dma_transfer(&self, len: u32, block_size: u8, direction: Dir) {
        let dtdir = matches!(direction, Dir::CardToHost);

        self.rb.dtimer.write(|w| w.datatime().variant(0xFFFF_FFFF));
        self.rb.dlenr.write(|w| w.datalength().variant(len));
        // This delay is needed due to some hardware bug in SDMMC + IDMA
        for _ in 0..300 {
            cortex_m::asm::nop();
        }
        self.rb.dctrl.write(|w| {
            w.dblocksize()
                .variant(block_size) // 2^n bytes block size
                .dtdir()
                .bit(dtdir)
                .dtmode()
                .variant(0)
                .dten()
                .clear_bit() // Enable transfer
        });
        self.rb.cmdr.modify(|_, w| w.cmdtrans().set_bit());
        self.rb
            .idmabase0r
            .write(|w| w.idmabase0().variant(self.dma_buf.as_ptr() as u32));
        self.rb.idmactrlr.write(|w| w.idmaen().set_bit());
    }

    pub fn bench(&mut self, start_block: u32) -> Result<(), Error> {
        let block_counts = [1, 4, 8, 16, 32, 64, 128];
        for &blocks in &block_counts {
            let len = 512 * blocks;
            let mut total_duration = fugit::MicrosDuration::<u32>::from_ticks(0);
            const ITERATIONS: u32 = 10;
            for _ in 0..ITERATIONS {
                let duration = self.bench_write(start_block, blocks)?;
                total_duration += duration;
            }
            let avg_duration = total_duration / ITERATIONS;
            let mb_per_sec = len as f32 / avg_duration.to_micros() as f32;
            defmt::info!(
                "Wrote {} blocks ({} bytes) in {}: {} MB/s",
                blocks,
                len,
                avg_duration,
                mb_per_sec
            );
        }
        Ok(())
    }

    fn bench_write(
        &mut self,
        start_block: u32,
        num_blocks: usize,
    ) -> Result<fugit::MicrosDuration<u32>, Error> {
        let buf = alloc::vec![0u8; 512 * num_blocks].into_boxed_slice();
        let start = self.dwt.now();
        self.write_blocks(start_block, &buf)?;
        self.wait_card_ready(true)?;
        let elapsed = start.elapsed();
        Ok(elapsed)
    }

    fn write_block(&mut self, address: u32, buffer: [u8; BLOCK_LEN]) -> Result<(), Error> {
        let start = self.dwt.now();
        self.wait_card_ready(false)?;
        self.start_dma_transfer(BLOCK_LEN as u32, 9, Dir::HostToCard);
        self.dma_buf[..BLOCK_LEN].copy_from_slice(&buffer);
        self.cmd(common_cmd::write_single_block(address))?;
        self.rb.dctrl.modify(|_, w| w.dten().set_bit());
        let elapsed = start.elapsed();
        if elapsed > fugit::MillisDuration::<u32>::millis(1) {
            defmt::warn!("SDMMC single-block write took {}", elapsed);
        }
        Ok(())
    }

    fn write_blocks(&mut self, address: u32, buffer: &[u8]) -> Result<(), Error> {
        let start = self.dwt.now();
        let n_blocks = buffer.len() as u32 / 512;
        assert!(
            buffer.len() % 512 == 0,
            "Buffer length must be a multiple of 512"
        );
        assert!(
            buffer.len() <= self.dma_buf.len(),
            "Buffer length must be less than or equal to dma buffer size"
        );
        self.card_ready(true)?;
        self.invalidate(address..=address + n_blocks)?;

        self.start_dma_transfer(buffer.len() as u32, 9, Dir::HostToCard);
        self.dma_buf[..buffer.len()].copy_from_slice(buffer);
        if n_blocks == 1 {
            self.cmd(common_cmd::write_single_block(address))?;
        } else {
            self.cmd(common_cmd::cmd::<common_cmd::R1>(23, n_blocks))?;
            self.cmd(common_cmd::write_multiple_blocks(address))?;
        }
        self.rb.dctrl.modify(|_, w| w.dten().set_bit());
        let elapsed = start.elapsed();
        if elapsed > fugit::MillisDuration::<u32>::millis(1) {
            defmt::warn!("SDMMC multi-block write took {}", elapsed);
        }
        Ok(())
    }

    pub fn read_block(&mut self, address: u32) -> Result<[u8; BLOCK_LEN], Error> {
        let start = self.dwt.now();
        self.card_ready(false)?;
        self.start_dma_transfer(BLOCK_LEN as u32, 9, Dir::CardToHost);
        self.cmd(common_cmd::read_single_block(address))?;
        self.rb.dctrl.modify(|_, w| w.dten().set_bit());
        while self.rb.star.read().dataend().bit_is_clear() {}
        let mut buffer = [0u8; BLOCK_LEN];
        buffer.copy_from_slice(&self.dma_buf[..BLOCK_LEN]);
        let _elapsed = start.elapsed();
        Ok(buffer)
    }

    fn read_status(&self) -> Result<CardStatus<Self>, Error> {
        self.cmd(common_cmd::card_status(self.card_rca(), false))?;
        Ok(CardStatus::from(self.rb.resp1r.read().bits()))
    }

    fn wait_card_ready(&mut self, write_back: bool) -> Result<(), Error> {
        let timeout: fugit::MicrosDuration<u32> = 1000u32.millis();
        let start = self.dwt.now();
        while start.elapsed() < timeout {
            match self.card_ready(write_back) {
                Ok(_) => return Ok(()),
                Err(Error::Busy) => {}
                Err(err) => return Err(err),
            }
        }
        Err(Error::SoftwareTimeout)
    }

    fn card_ready(&mut self, write_back: bool) -> Result<(), Error> {
        let status = self.rb.star.read().to_result()?;
        if status.dpsmact().bit_is_set() || status.cpsmact().bit_is_set() {
            return Err(Error::Busy);
        }
        let sd_state = self.read_status()?.state();
        if sd_state != CurrentState::Transfer {
            return Err(Error::Busy);
        }
        if write_back {
            let dirty_block = self
                .block_cache
                .blocks
                .iter_mut()
                .find(|b| b.needs_writeback());
            if let Some(block) = dirty_block {
                block.dirty = false;
                let block = block.clone();
                self.write_block(block.index, block.data)?;
                return Err(Error::Busy);
            }
        }
        Ok(())
    }

    fn get_scr(&self, rca: u16) -> Result<SCR, Error> {
        self.cmd(common_cmd::app_cmd(rca))?;
        self.start_datapath_transfer(8, 3, Dir::CardToHost);
        self.cmd(sd_cmd::send_scr())?;

        let mut scr = [0; 2];
        let mut i = 0;
        let mut status;
        while {
            status = self.rb.star.read().to_result()?;
            status.dbckend().bit_is_clear()
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

        Ok(SCR::from(scr))
    }

    pub fn try_connect(&mut self) -> Result<(), Error> {
        self.disconnect();
        self.power_on();
        self.cmd(common_cmd::idle())?;

        match self.cmd(sd_cmd::send_if_cond(1, 0xAA)) {
            Ok(_) => {
                defmt::debug!("SD card detected");
            }
            Err(Error::Timeout) => {
                self.power_off();
                return Err(Error::NoCard);
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
        if signaling != SdCardSignaling::SDR25 || self.card_ready(false).is_err() {
            return Err(Error::SignalingSwitchFailed);
        }
        self.read_sd_status()?;
        self.log_sd_status()?;
        self.wait_card_ready(false)?;
        self.find_fatfs()?;
        Ok(())
    }

    fn set_freq(&mut self, freq: Hertz) -> Result<(), Error> {
        let (clkdiv, clk) = Self::clk_div(freq)?;
        defmt::debug!("CLKDIV: {}, SDMMC clock: {}kHz", clkdiv, clk.to_kHz());
        self.rb
            .clkcr
            .modify(|_, w| unsafe { w.clkdiv().bits(clkdiv) });
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
            sta_reg = self.rb.star.read().to_result()?;
            sta_reg.dbckend().bit_is_clear()
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
        let sd_signaling = match selection {
            0 => SdCardSignaling::SDR12,
            1 => SdCardSignaling::SDR25,
            2 => SdCardSignaling::SDR50,
            3 => SdCardSignaling::SDR104,
            4 => SdCardSignaling::DDR50,
            _ => return Err(Error::UnsupportedCardType),
        };

        if selection > 0 {
            while self.rb.star.read().dpsmact().bit_is_set()
                || self.rb.star.read().cpsmact().bit_is_set()
            {}
            self.rb.clkcr.modify(|_, w| w.busspeed().set_bit());
        }

        Ok(sd_signaling)
    }

    pub fn fatfs(self, led: hal::gpio::Pin) -> SdmmcFs {
        SdmmcFs::new(self, led)
    }

    pub fn connected(&mut self) -> bool {
        self.card.is_some() && self.read_sd_status().is_ok()
    }

    pub fn disconnect(&mut self) {
        self.power_off();
        self.block_cache.clear();
        self.card = None;
    }

    // Find the first fat32 partition in the MBR
    fn find_fatfs(&mut self) -> Result<(), Error> {
        let buffer = self.read_block(0)?;
        let mbr = Mbr::read_from_bytes(&buffer).map_err(|_| Error::InvalidConfiguration)?;
        // Validate MBR signature
        if mbr.signature != 0xAA55 {
            return Err(Error::InvalidConfiguration);
        }
        let card = self.card.as_mut().ok_or(Error::NoCard)?;
        for partition in mbr.partitions {
            if partition.partition_type == 0x0C || partition.partition_type == 0x0B {
                card.fatfs_info = FatFsInfo {
                    start_lba: partition.start_lba as u64,
                    num_sectors: partition.num_sectors as u64,
                    sector_size: 512,
                    cursor: partition.start_lba as u64 * 512,
                };
                defmt::debug!("Found FAT32 partition: {}", card.fatfs_info);
                return Ok(());
            }
        }
        Err(Error::FatFsNotFound)
    }

    fn fatfs_info(&mut self) -> Result<&mut FatFsInfo, Error> {
        let card = self.card.as_mut().ok_or(Error::NoCard)?;
        Ok(&mut card.fatfs_info)
    }

    fn insert(&mut self, index: u32, data: [u8; 512], dirty: bool) -> Result<(), Error> {
        let cache_index = self.block_cache.next;
        self.block_cache.next = (cache_index + 1) % self.block_cache.blocks.len();
        let block = &mut self.block_cache.blocks[cache_index];
        if block.needs_writeback() {
            return Err(Error::Busy);
        }
        *block = Block {
            index,
            data,
            dirty,
            stale: false,
        };
        Ok(())
    }

    fn invalidate(&mut self, block_range: core::ops::RangeInclusive<u32>) -> Result<(), Error> {
        self.block_cache
            .blocks
            .iter_mut()
            .filter(|b| block_range.contains(&b.index))
            .for_each(|b| b.stale = true);
        Ok(())
    }
}

fn cmd<R: common_cmd::Resp>(rb: &pac::sdmmc1::RegisterBlock, cmd: Cmd<R>) -> Result<(), Error> {
    // CP state machine must be idle
    while rb.star.read().cpsmact().bit_is_set() {}

    // Clear status flags
    rb.icr.write(|w| unsafe { w.bits(0xFFFF_FFFF) });

    // Command arg
    rb.argr.write(|w| unsafe { w.cmdarg().bits(cmd.arg) });

    // Determine what kind of response the CPSM should wait for
    let waitresp = match cmd.response_len() {
        ResponseLen::Zero => 0,
        ResponseLen::R48 => 1,  // short response, expect CMDREND or CCRCFAIL
        ResponseLen::R136 => 3, // long response, expect CMDREND or CCRCFAIL
    };

    // Special mode in CP State Machine
    // CMD12: Stop Transmission
    let cpsm_stop_transmission = cmd.cmd == 12;

    // Command index and start CP State Machine
    rb.cmdr.write(|w| unsafe {
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

    let mut attempts_remaining = 100_000_000;
    let mut status;
    if cmd.response_len() == ResponseLen::Zero {
        // Wait for CMDSENT or a timeout
        while {
            status = rb.star.read();
            !(status.ctimeout().bit() || status.cmdsent().bit()) && attempts_remaining > 0
        } {
            attempts_remaining -= 1;
        }
    } else {
        // Wait for CMDREND or CCRCFAIL or a timeout
        while {
            status = rb.star.read();
            !(status.ctimeout().bit() || status.cmdrend().bit() || status.ccrcfail().bit())
                && attempts_remaining > 0
        } {
            attempts_remaining -= 1;
        }
    }

    status.to_result()?;
    if attempts_remaining == 0 {
        return Err(Error::SoftwareTimeout);
    }
    Ok(())
}

#[derive(Copy, Clone, Debug, FromBytes, IntoBytes, Immutable, defmt::Format)]
#[repr(C)]
pub struct PartitionEntry {
    pub boot_flag: u8,
    pub start_head: u8,
    pub start_sector_cylinder: u16,
    pub partition_type: u8,
    pub end_head: u8,
    pub end_sector_cylinder: u16,
    pub start_lba: u32,
    pub num_sectors: u32,
}

#[derive(Copy, Clone, Debug, FromBytes, IntoBytes, Immutable)]
#[repr(C, packed)]
struct Mbr {
    pub bootstrap: [u8; 446],
    pub partitions: [PartitionEntry; 4],
    pub signature: u16,
}

impl fatfs::IoError for Error {
    fn is_interrupted(&self) -> bool {
        matches!(self, Error::Busy)
    }

    fn new_unexpected_eof_error() -> Self {
        Error::UnexpectedEof
    }

    fn new_write_zero_error() -> Self {
        Error::WriteZero
    }
}

impl fatfs::IoBase for Sdmmc {
    type Error = Error;
}

impl fatfs::Seek for Sdmmc {
    fn seek(&mut self, pos: fatfs::SeekFrom) -> Result<u64, Self::Error> {
        let card = self.card.as_mut().ok_or(Error::NoCard)?;
        let fatfs = &mut card.fatfs_info;
        match pos {
            fatfs::SeekFrom::Start(offset) => {
                let start = fatfs.start_lba * fatfs.sector_size;
                fatfs.cursor = start.saturating_add(offset);
                defmt::trace!("Seek {} from 0 -> {}", offset, fatfs.cursor);
            }
            fatfs::SeekFrom::End(offset) => {
                let end = (fatfs.start_lba + fatfs.num_sectors) * fatfs.sector_size;
                fatfs.cursor = (end as i64).saturating_sub(offset) as u64;
                defmt::trace!("Seek {} from {} -> {}", offset, end, fatfs.cursor);
            }
            fatfs::SeekFrom::Current(offset) => {
                let current = fatfs.cursor as i64;
                fatfs.cursor = current.saturating_add(offset) as u64;
                defmt::trace!("Seek {} from {} -> {}", offset, current, fatfs.cursor);
            }
        }
        Ok(fatfs.cursor)
    }
}

impl fatfs::Read for Sdmmc {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
        let start = self.dwt.now();
        let cursor = self.fatfs_info()?.cursor as usize;
        let block_index = (cursor / BLOCK_LEN) as u32;
        let start_offset = cursor % BLOCK_LEN;
        defmt::trace!(
            "Reading {} bytes from {} (block {})",
            buf.len(),
            cursor,
            block_index
        );

        let len = buf.len().min(BLOCK_LEN - start_offset);
        match self.block_cache.get(block_index) {
            Some(block) => {
                buf[..len].copy_from_slice(&block[start_offset..start_offset + len]);
            }
            None => {
                let block = self.read_block(block_index)?;
                defmt::trace!(
                    "Read block {} after cache miss in {}",
                    block_index,
                    start.elapsed()
                );
                buf[..len].copy_from_slice(&block[start_offset..start_offset + len]);
                self.insert(block_index, block, false)?;
            }
        };
        self.fatfs_info()?.cursor += len as u64;
        Ok(len)
    }
}

impl fatfs::Write for Sdmmc {
    fn write(&mut self, mut buf: &[u8]) -> Result<usize, Self::Error> {
        let start = self.dwt.now();
        let cursor = self.fatfs_info()?.cursor as usize;
        buf = &buf[..buf.len().min(self.dma_buf.len())];
        let block_index = (cursor / BLOCK_LEN) as u32;
        let start_offset = cursor % BLOCK_LEN;
        let mut end_offset = (cursor + buf.len()) % BLOCK_LEN;
        defmt::trace!(
            "Writing {} bytes to {} (block {})",
            buf.len(),
            cursor,
            block_index,
        );
        if buf.len() > BLOCK_LEN && end_offset != 0 {
            // Align large writes to block boundary
            // Can only do this with the end of the buffer
            buf = &buf[..buf.len() - end_offset];
            end_offset = 0;
        }
        if start_offset != 0 || end_offset != 0 {
            buf = &buf[..buf.len().min(BLOCK_LEN - start_offset)];
            match self.block_cache.get_mut(block_index) {
                Some(block) => {
                    block[start_offset..start_offset + buf.len()].copy_from_slice(buf);
                }
                None => {
                    let mut block = self.read_block(block_index)?;
                    defmt::trace!(
                        "Read block {} for writing after cache miss in {}",
                        block_index,
                        start.elapsed()
                    );
                    block[start_offset..start_offset + buf.len()].copy_from_slice(buf);
                    self.insert(block_index, block, true)?;
                }
            };
        } else {
            self.write_blocks(block_index, buf)?;
        };
        self.fatfs_info()?.cursor += buf.len() as u64;
        Ok(buf.len())
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        let _ = self.card_ready(true);
        Ok(())
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

trait StatusExt: Sized {
    fn to_result(self) -> Result<Self, Error>;
}

impl StatusExt for pac::sdmmc1::star::R {
    fn to_result(self) -> Result<Self, Error> {
        if self.dcrcfail().bit() {
            Err(Error::DataCrcFail)
        } else if self.rxoverr().bit() {
            Err(Error::RxOverFlow)
        } else if self.txunderr().bit() {
            Err(Error::TxUnderFlow)
        } else if self.dtimeout().bit() {
            Err(Error::Timeout)
        } else {
            Ok(self)
        }
    }
}
