use crate::i2c_dma::I2cRegs;
use bitfield_struct::bitfield;
use embedded_hal::i2c::{I2c, Operation};
use hal::i2c;

const ADDR: u8 = 0b0101100;

#[bitfield(u8)]
pub struct ConfigByte1 {
    pub power_switching: bool,
    #[bits(2)]
    pub overcurrent: u8,
    pub eop_disable: bool,
    pub multi_tt: bool,
    pub high_speed_disable: bool,
    pub reserved: bool,
    pub self_powered: bool,
}

#[bitfield(u8)]
pub struct ConfigByte2 {
    #[bits(3)]
    pub reserved_low: u8,
    pub compound_device: bool,
    #[bits(2)]
    pub overcurrent_timer: u8,
    pub reserved_high: bool,
    pub dynamic_power: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct UsbIds {
    pub vid: u16,
    pub pid: u16,
    pub did: u16,
}

#[derive(Debug, Clone, Copy)]
pub struct PowerSettings {
    pub max_power_self: u16,   // in mA
    pub max_power_bus: u16,    // in mA
    pub hub_current_self: u16, // in mA
    pub hub_current_bus: u16,  // in mA
    pub power_on_time: u16,    // in ms
}

/// Standard configuration for self-powered operation
pub const SELF_POWERED_CONFIG: [u8; 13] = [
    0x06, // CMD: ConfigByte1 register
    0x0B, // BYTE_COUNT: 11 bytes
    0x9C, // Self-powered, multi-TT, no OC sense, EOP disabled, ganged power
    0x00, // Dynamic power off, standard OC timer, not compound device
    0x00, // No port mapping, no string descriptors
    0x00, // All ports removable
    0x08, // Disable port 3 when self-powered
    0x08, // Disable port 3 when bus-powered
    0xFA, // Max power self-powered = 500mA
    0x00, // Max power bus-powered = 0mA
    0xFA, // Hub controller current self-powered = 500mA
    0x00, // Hub controller current bus-powered = 0mA
    0x64, // Power-on time = 200ms
];

/// Port swap configuration to fix hardware DP/DM reversal
/// Register 0xFA (PRTSP) - Port Swap register
pub const PORT_SWAP_CONFIG: [u8; 3] = [
    0xFA,        // CMD: PRTSP register (Port Swap)
    0x01,        // BYTE_COUNT: 1 byte
    0b0000_0100, // PRTSP: Bit 2 = 1 (swap downstream port 2 DM/DP)
];

/// Register addresses for the USB2513B hub
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum UsbHubRegister {
    VendorIdLsb = 0x00,
    VendorIdMsb = 0x01,
    ProductIdLsb = 0x02,
    ProductIdMsb = 0x03,
    DeviceIdLsb = 0x04,
    DeviceIdMsb = 0x05,
    ConfigByte1 = 0x06,
    ConfigByte2 = 0x07,
    ConfigByte3 = 0x08,
    NonRemovableDevice = 0x09,
    PortDisableSelf = 0x0A,
    PortDisableBus = 0x0B,
    MaxPowerSelf = 0x0C,
    MaxPowerBus = 0x0D,
    HubControllerCurrentSelf = 0x0E,
    HubControllerCurrentBus = 0x0F,
    PowerOnTime = 0x10,
    PortSwap = 0xFA, // PRTSP register for DP/DM swapping
    StatusCmd = 0xFF,
}

#[derive(Debug, defmt::Format)]
pub enum Error {
    I2c(i2c::Error),
    InvalidArgument,
    ReadbackMismatch,
    AlreadyConfigured,
    InvalidData,
}

impl From<i2c::Error> for Error {
    fn from(err: i2c::Error) -> Self {
        Error::I2c(err)
    }
}

pub struct Usb2513b {
    address: u8,
}

impl Default for Usb2513b {
    fn default() -> Self {
        Self { address: ADDR }
    }
}

impl Usb2513b {
    /// Check if the USB hub is already configured and finalized
    /// Returns true if the hub is already configured (USB_ATTACH bit is set)
    pub fn is_configured(&self, i2c_dev: &mut i2c::I2c<I2cRegs>) -> Result<bool, Error> {
        defmt::debug!("USB HUB: Checking if already configured...");

        // Try to read the STATUS/CMD register (0xFF) to check USB_ATTACH bit
        let status_cmd_reg = [UsbHubRegister::StatusCmd as u8];
        let mut status_response = [0u8; 2]; // SMBus block read: [byte_count, data]

        let mut ops = [
            Operation::Write(&status_cmd_reg),
            Operation::Read(&mut status_response),
        ];

        match i2c_dev.transaction(self.address, &mut ops) {
            Ok(()) => {
                let byte_count = status_response[0];
                let status_value = status_response[1];
                defmt::debug!(
                    "USB HUB: STATUS/CMD read: byte_count={}, value=0x{:02X}",
                    byte_count,
                    status_value
                );

                // USB_ATTACH is bit 0. If set, hub is already configured and finalized
                let is_configured = (status_value & 0x01) != 0;
                defmt::info!(
                    "USB HUB: Configuration status: {}",
                    if is_configured {
                        "Already configured"
                    } else {
                        "Not configured"
                    }
                );
                Ok(is_configured)
            }
            Err(e) => {
                defmt::warn!("USB HUB: Failed to read STATUS/CMD register: {:?}", e);
                // If we can't read the status, assume it's not configured
                Ok(false)
            }
        }
    }

    /// Read and dump the current configuration registers for debugging
    pub fn dump_config(&self, i2c_dev: &mut i2c::I2c<I2cRegs>) -> Result<(), Error> {
        defmt::debug!("USB HUB: Reading current configuration...");

        // We access registers up to 0x10 (power_on_time), so need at least 0x10 + 1 bytes + 1 for byte count
        const MIN_REQUIRED_BYTES: usize = 0x10 + 1 + 1;

        // Read core configuration registers (0x00-0x10) in one block
        let config_reg = [UsbHubRegister::VendorIdLsb as u8];
        let mut config_buffer = [0u8; MIN_REQUIRED_BYTES];

        let mut ops = [
            Operation::Write(&config_reg),
            Operation::Read(&mut config_buffer),
        ];

        i2c_dev.transaction(self.address, &mut ops)?;

        // Raw hex dump first
        defmt::trace!("USB HUB: Raw config: {:02X}", config_buffer);

        // Skip the first byte (byte count) and use actual register data
        let byte_count = config_buffer[0];
        defmt::trace!("USB HUB: Byte count: {}", byte_count);

        // Validate we have enough data for all parsing operations
        if (byte_count as usize) < (MIN_REQUIRED_BYTES - 1) {
            defmt::warn!(
                "USB HUB: Insufficient data - got {} bytes, need at least {} for full parsing",
                byte_count,
                MIN_REQUIRED_BYTES - 1
            );
            return Err(Error::InvalidData);
        }

        let regs = &config_buffer[1..]; // Skip byte count

        // Parse USB IDs
        let usb_ids = UsbIds {
            vid: u16::from_le_bytes([regs[0], regs[1]]),
            pid: u16::from_le_bytes([regs[2], regs[3]]),
            did: u16::from_le_bytes([regs[4], regs[5]]),
        };
        defmt::debug!(
            "USB HUB: IDs - VID=0x{:04X}, PID=0x{:04X}, DID=0x{:04X}",
            usb_ids.vid,
            usb_ids.pid,
            usb_ids.did
        );

        // Parse config bytes with proper bitfields
        let cfg1 = ConfigByte1::from(regs[6]);
        defmt::debug!(
            "USB HUB: Config1(0x{:02X}): self_powered={}, hs_disable={}, multi_tt={}, eop_disable={}",
            regs[6],
            cfg1.self_powered(),
            cfg1.high_speed_disable(),
            cfg1.multi_tt(),
            cfg1.eop_disable()
        );

        let cfg2 = ConfigByte2::from(regs[7]);
        defmt::debug!(
            "USB HUB: Config2(0x{:02X}): dynamic_power={}, compound_device={}",
            regs[7],
            cfg2.dynamic_power(),
            cfg2.compound_device()
        );

        // Parse power settings
        let power = PowerSettings {
            max_power_self: regs[0x0C] as u16 * 2,
            max_power_bus: regs[0x0D] as u16 * 2,
            hub_current_self: regs[0x0E] as u16 * 2,
            hub_current_bus: regs[0x0F] as u16 * 2,
            power_on_time: regs[0x10] as u16 * 2,
        };
        defmt::debug!(
            "USB HUB: Power - Self={}mA, Bus={}mA, HubSelf={}mA, HubBus={}mA, PowerOn={}ms",
            power.max_power_self,
            power.max_power_bus,
            power.hub_current_self,
            power.hub_current_bus,
            power.power_on_time
        );

        Ok(())
    }

    /// Configure the USB hub if needed, with config dump for debugging
    pub fn configure_if_needed(&self, i2c_dev: &mut i2c::I2c<I2cRegs>) -> Result<(), Error> {
        defmt::debug!("USB HUB: Starting configuration check and setup...");

        // Dump current config for debugging
        if let Err(e) = self.dump_config(i2c_dev) {
            defmt::warn!("USB HUB: Failed to dump initial config: {:?}", e);
        }

        // Check if already configured once
        if self.is_configured(i2c_dev)? {
            defmt::info!("USB HUB: Already configured and finalized, skipping configuration");
            return Ok(());
        }

        // Configure with standard self-powered config
        defmt::debug!("USB HUB: Writing configuration...");
        let mut ops = [Operation::Write(&SELF_POWERED_CONFIG)];
        i2c_dev.transaction(self.address, &mut ops)?;
        defmt::debug!("USB HUB: Configuration written successfully");

        // Apply port swap to fix hardware DP/DM reversal
        defmt::debug!("USB HUB: Swapping downstream port 2 DP/DM...");
        let mut ops = [Operation::Write(&PORT_SWAP_CONFIG)];
        i2c_dev.transaction(self.address, &mut ops)?;
        defmt::debug!("USB HUB: Downstream port 2 DP/DM swapped");

        // Finalize with USB_ATTACH
        defmt::debug!("USB HUB: Finalizing with USB_ATTACH...");
        let attach_payload = [UsbHubRegister::StatusCmd as u8, 0x01, 0x01];
        let mut ops = [Operation::Write(&attach_payload)];
        i2c_dev.transaction(self.address, &mut ops)?;
        defmt::debug!("USB HUB: Hub now visible to USB host, registers write-protected");
        Ok(())
    }
}
