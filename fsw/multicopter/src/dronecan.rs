#![allow(dead_code)]

use modular_bitfield::prelude::*;

use crate::{can, monotonic};

use dsdl::*;

type Duration = fugit::MillisDuration<u32>;

pub const MIN_BROADCASTING_PERIOD: Duration = Duration::millis(2);
pub const MAX_BROADCASTING_PERIOD: Duration = Duration::millis(1000);

pub const OFFLINE_TIMEOUT: Duration = Duration::millis(3000);

const TRANSFER_CRC: crc::Crc<u16> = crc::Crc::<u16>::new(&crc::CRC_16_IBM_3740);

// Supported DroneCAN message types:

// Basic:
// uavcan.protocol.NodeStatus
// uavcan.protocol.debug.LogMessage

// ESC:
// uavcan.equipment.esc.Status
// uavcan.equipment.esc.RawCommand
// uavcan.equipment.safety.ArmingStatus

#[derive(Copy, Clone, Debug, defmt::Format)]
pub enum Error {
    StandardId,
    MessageTooLong,
    CorruptMessage,
    InvalidUtf8,
    UnknownDataType,
    CrcMismatch,
}

impl From<core::str::Utf8Error> for Error {
    fn from(_: core::str::Utf8Error) -> Self {
        Self::InvalidUtf8
    }
}

#[bitfield(bits = 29)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Id {
    source_node_id: NodeId,
    service_not_message: bool,
    frame_type_bits: u16,
    priority_bits: Priority,
}

#[bitfield(bits = 7)]
#[derive(BitfieldSpecifier, Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct NodeId {
    bits: B7,
}

#[derive(Copy, Clone, Debug, BitfieldSpecifier, PartialEq, Eq)]
#[bits = 5]
pub enum Priority {
    Exceptional = 0, // Highest priority
    Immediate = 1,
    Fast = 2,
    High = 3,
    Nominal = 4,
    Low = 5,
    Slow = 6,
    Optional = 7, // Lowest priority
}

pub struct RawMessage<'a> {
    id: Id,
    crc: Option<u16>,
    buf: &'a [u8],
}

#[derive(defmt::Format)]
pub struct Message<'a> {
    priority: Priority,
    source_node_id: NodeId,
    destination_node_id: Option<NodeId>,
    frame_type: FrameType,
    message_type: MessageType<'a>,
}

impl<'a> TryFrom<RawMessage<'a>> for Message<'a> {
    type Error = Error;

    fn try_from(raw_message: RawMessage<'a>) -> Result<Self, Self::Error> {
        let id = raw_message.id;
        let data_type_id = id.data_type_id();
        let message_type = MessageType::parse(data_type_id, raw_message.buf)?;

        // Validate checksum if present
        if let Some(expected_crc) = raw_message.crc {
            let mut digest = TRANSFER_CRC.digest();
            digest.update(&message_type.data_type_signature().to_le_bytes());
            digest.update(raw_message.buf);
            let computed_crc = digest.finalize();
            if computed_crc != expected_crc {
                return Err(Error::CrcMismatch);
            }
        }

        Ok(Self {
            priority: id.priority(),
            source_node_id: id.source_node_id(),
            destination_node_id: id.destination_node_id(),
            frame_type: id.frame_type(),
            message_type,
        })
    }
}

#[derive(defmt::Format)]
pub enum MessageType<'a> {
    NodeStatus(NodeStatusType),
    LogMessage(LogMessageType<'a>),
}

impl<'a> MessageType<'a> {
    pub fn parse(data_type_id: u16, buf: &'a [u8]) -> Result<Self, Error> {
        let message_type = match data_type_id {
            NodeStatusType::ID => MessageType::NodeStatus(NodeStatusType::parse(buf)?),
            LogMessageType::ID => MessageType::LogMessage(LogMessageType::parse(buf)?),
            _ => return Err(Error::UnknownDataType),
        };
        Ok(message_type)
    }

    pub fn data_type_signature(&self) -> u64 {
        match self {
            MessageType::NodeStatus(_) => NodeStatusType::SIGNATURE,
            MessageType::LogMessage(_) => LogMessageType::SIGNATURE,
        }
    }
}

pub struct DroneCan {
    can: fdcan::FdCan<can::Can, fdcan::NormalOperationMode>,
    in_flight_transfers: heapless::LinearMap<Id, InFlightTransfer, 32>,
    buf: [u8; 128],
}

struct InFlightTransfer {
    buf: [u8; 128],
    len: usize,
    toggle: bool,
    transfer_id: u8,
    last_appended: monotonic::Instant,
}

impl Default for InFlightTransfer {
    fn default() -> Self {
        Self {
            buf: [0u8; 128],
            len: 0,
            transfer_id: 0,
            toggle: false,
            last_appended: monotonic::Instant::from_ticks(0),
        }
    }
}

impl DroneCan {
    pub fn new(can: fdcan::FdCan<can::Can, fdcan::NormalOperationMode>) -> Self {
        Self {
            can,
            in_flight_transfers: Default::default(),
            buf: [0u8; 128],
        }
    }

    pub fn read(&mut self, now: monotonic::Instant) -> Option<RawMessage> {
        let frame_info = match self.can.receive0(&mut self.buf).ok()? {
            fdcan::ReceiveOverrun::Overrun(frame_info) => {
                self.can
                    .clear_interrupt(fdcan::interrupt::Interrupt::RxFifo0MsgLost);
                defmt::warn!("CAN bus overrun");
                frame_info
            }
            fdcan::ReceiveOverrun::NoOverrun(frame_info) => frame_info,
        };
        let len = frame_info.len as usize - 1;
        let id = match Id::try_from(frame_info.id) {
            Ok(id) => id,
            Err(err) => {
                defmt::warn!("invalid CAN frame ID: {:?}", err);
                return None;
            }
        };
        let (payload, tail_byte) = self.buf.split_at(len);
        let tail_byte = TailByte::from_bytes([tail_byte[0]]);
        defmt::trace!(
            "Received CAN frame: {}, transfer_id={}, toggle={}, end_of_transfer={}, start_of_transfer={}, len={}",
            id,
            tail_byte.transfer_id(),
            tail_byte.toggle(),
            tail_byte.end_of_transfer(),
            tail_byte.start_of_transfer(),
            payload.len(),
        );

        let message = if tail_byte.start_of_transfer() {
            if tail_byte.end_of_transfer() {
                Some(RawMessage {
                    id,
                    crc: None,
                    buf: &self.buf[..len],
                })
            } else {
                if tail_byte.toggle() {
                    defmt::warn!("Expected toggle bit of 0 on start of transfer");
                    return None;
                }
                let mut message = InFlightTransfer {
                    transfer_id: tail_byte.transfer_id(),
                    last_appended: now,
                    ..Default::default()
                };
                message.buf[..len].copy_from_slice(payload);
                message.len = len;
                match self.in_flight_transfers.insert(id, message) {
                    Ok(Some(_)) => defmt::warn!("In-flight transfer ID collision"),
                    Ok(None) => {}
                    Err(_) => defmt::warn!("Max number of in-flight transfers exceeded"),
                }
                None
            }
        } else {
            let Some(message) = self.in_flight_transfers.get_mut(&id) else {
                defmt::warn!("Received unknown transfer ID");
                return None;
            };
            if message.transfer_id != tail_byte.transfer_id() {
                defmt::warn!("Transfer ID mismatch");
                return None;
            }
            let next_toggle = !message.toggle;
            if tail_byte.toggle() != next_toggle {
                defmt::warn!("Toggle bit mismatch");
                return None;
            }

            message.buf[message.len..message.len + payload.len()].copy_from_slice(payload);
            message.len += payload.len();
            message.last_appended = now;
            message.toggle = next_toggle;

            if tail_byte.end_of_transfer() {
                let message_len = message.len;
                self.buf[..message_len].copy_from_slice(&message.buf[..message_len]);
                self.in_flight_transfers.remove(&id);
                let crc = u16::from_le_bytes(self.buf[..2].try_into().unwrap());
                Some(RawMessage {
                    id,
                    crc: Some(crc),
                    buf: &self.buf[2..message_len],
                })
            } else {
                None
            }
        };

        message.inspect(|m| defmt::trace!("Received DroneCAN message: {}, len={}", m.buf.len(), id))
    }
}

#[bitfield(bits = 8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct TailByte {
    transfer_id: B5,
    toggle: bool,
    end_of_transfer: bool,
    start_of_transfer: bool,
}

impl Priority {
    const MAX_TRANSFER_ID: u8 = 0b1_1111;

    pub const fn high_priority(&self) -> bool {
        matches!(
            self,
            Self::Exceptional | Self::Immediate | Self::Fast | Self::High
        )
    }
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Exceptional => "Exceptional",
            Self::Immediate => "Immediate",
            Self::Fast => "Fast",
            Self::High => "High",
            Self::Nominal => "Nominal",
            Self::Low => "Low",
            Self::Slow => "Slow",
            Self::Optional => "Optional",
        }
    }
}

impl defmt::Format for Priority {
    fn format(&self, fmt: defmt::Formatter) {
        defmt::write!(fmt, "{}", self.as_str())
    }
}

impl Default for Priority {
    fn default() -> Self {
        Self::Optional
    }
}

#[bitfield(bits = 16)]
#[derive(Copy, Clone, Debug, defmt::Format, Default)]
pub struct AnonymousFrameType {
    lower_message_type_id: B2,
    discriminator: B14,
}

#[bitfield(bits = 16)]
#[derive(Copy, Clone, Debug, defmt::Format, Default)]
pub struct ServiceFrameType {
    destination_node_id: NodeId,
    request_not_response: bool,
    service_type_id: u8,
}

#[derive(Copy, Clone, Debug, defmt::Format)]
pub enum FrameType {
    Broadcast,
    Anonymous,
    ServiceRequest,
    ServiceResponse,
}

impl defmt::Format for NodeId {
    fn format(&self, fmt: defmt::Formatter) {
        defmt::write!(fmt, "{}", self.bits())
    }
}

impl NodeId {
    pub const ZERO: Self = Self::from_bytes([0u8]);

    pub fn is_zero(&self) -> bool {
        self.bits() == 0
    }
}

impl defmt::Format for Id {
    fn format(&self, fmt: defmt::Formatter) {
        let priority_str = match self.priority().high_priority() {
            true => " (priority)",
            false => "",
        };
        let source_node_id = self.source_node_id();
        let destination_node_id = self.destination_node_id();
        let data_type_id = self.data_type_id();
        match self.frame_type() {
            FrameType::Broadcast => {
                defmt::write!(
                    fmt,
                    "{} -> *{}, message_type_id={}",
                    source_node_id,
                    priority_str,
                    data_type_id,
                )
            }
            FrameType::Anonymous => {
                defmt::write!(fmt, "anon -> * (alloc) {}", priority_str)
            }
            FrameType::ServiceRequest => {
                defmt::write!(
                    fmt,
                    "{} -> {}{}, service_type_id={}",
                    source_node_id,
                    destination_node_id,
                    priority_str,
                    data_type_id,
                )
            }
            FrameType::ServiceResponse => {
                defmt::write!(
                    fmt,
                    "{} <- {}{}, service_type_id={}",
                    source_node_id,
                    destination_node_id,
                    priority_str,
                    data_type_id,
                )
            }
        }
    }
}

impl TryFrom<fdcan::id::Id> for Id {
    type Error = Error;

    fn try_from(id: fdcan::id::Id) -> Result<Self, Self::Error> {
        match id {
            fdcan::id::Id::Standard(_) => Err(Error::StandardId),
            fdcan::id::Id::Extended(id) => Ok(Self::from_bytes(id.as_raw().to_le_bytes())),
        }
    }
}

impl Id {
    pub fn priority(&self) -> Priority {
        self.priority_bits_or_err().unwrap_or(Priority::Optional)
    }

    pub fn destination_node_id(&self) -> Option<NodeId> {
        match self.frame_type() {
            FrameType::ServiceRequest | FrameType::ServiceResponse => {
                let frame_type = ServiceFrameType::from_bytes(self.frame_type_bits().to_le_bytes());
                Some(frame_type.destination_node_id())
            }
            _ => None,
        }
    }

    pub fn data_type_id(&self) -> u16 {
        let mut data_type_id = self.frame_type_bits();
        match self.frame_type() {
            FrameType::Anonymous => {
                data_type_id &= 0b0000_0000_0000_0011;
            }
            FrameType::ServiceRequest | FrameType::ServiceResponse => {
                data_type_id >>= 8;
            }
            _ => {}
        }
        data_type_id
    }

    pub fn frame_type(&self) -> FrameType {
        if self.service_not_message() {
            let service_frame_type =
                ServiceFrameType::from_bytes(self.frame_type_bits().to_le_bytes());
            if service_frame_type.request_not_response() {
                FrameType::ServiceRequest
            } else {
                FrameType::ServiceResponse
            }
        } else if self.source_node_id().is_zero() {
            FrameType::Anonymous
        } else {
            FrameType::Broadcast
        }
    }
}

mod dsdl {
    use super::*;

    pub trait DataType {
        const FULL_NAME: &'static str;
        const ID: u16;
        // Signatures are obtained using the `show_data_type_info.py` script
        // in the libcanard repository: https://github.com/dronecan/libcanard/tree/master.
        const SIGNATURE: u64;
        const MAX_BIT_LEN: usize;
        const MAX_BYTE_LEN: usize = (Self::MAX_BIT_LEN + 7) / 8;
    }

    #[derive(BitfieldSpecifier, Copy, Clone, Debug, PartialEq, Eq, defmt::Format)]
    #[bits = 2]
    pub enum Health {
        Ok,
        Warning,
        Error,
        Unknown,
    }

    #[derive(BitfieldSpecifier, Copy, Clone, Debug, PartialEq, Eq, defmt::Format)]
    #[bits = 3]
    pub enum Mode {
        Operational,
        Initialization,
        Maintenance,
        SoftwareUpdate,
        Offline = 7,
    }

    #[bitfield(bits = 56)]
    #[derive(Default)]
    pub struct NodeStatusType {
        uptime_sec: u32,
        sub_mode: B3,
        mode: Mode,
        health: Health,
        vendor_specific_status_word: u16,
    }

    impl NodeStatusType {
        pub fn parse(buf: &[u8]) -> Result<Self, Error> {
            let buf = buf.try_into().map_err(|_| Error::CorruptMessage)?;
            Ok(Self::from_bytes(buf))
        }
    }

    impl defmt::Format for NodeStatusType {
        fn format(&self, fmt: defmt::Formatter) {
            defmt::write!(
                fmt,
                "NodeStatusType {{ uptime_sec: {}, health: {}, mode: {} }}",
                self.uptime_sec(),
                self.health(),
                self.mode(),
            )
        }
    }

    impl DataType for NodeStatusType {
        const FULL_NAME: &'static str = "uavcan.protocol.NodeStatus";
        const ID: u16 = 341;
        const SIGNATURE: u64 = 0x0f0868d0c1a7c6f1;
        const MAX_BIT_LEN: usize = 56;
    }

    #[derive(defmt::Format)]
    pub enum LogLevel {
        Debug,
        Info,
        Warning,
        Error,
    }

    #[derive(defmt::Format)]
    pub struct LogMessageType<'a> {
        log_level: LogLevel,
        source: &'a str,
        text: &'a str,
    }

    impl<'a> LogMessageType<'a> {
        pub fn parse(mut buf: &'a [u8]) -> Result<Self, Error> {
            let log_level = match (buf[0] & 0b1110_0000) >> 5 {
                0 => LogLevel::Debug,
                1 => LogLevel::Info,
                2 => LogLevel::Warning,
                3 => LogLevel::Error,
                _ => return Err(Error::CorruptMessage),
            };
            let source_len = (buf[0] & 0b0001_1111) as usize;
            buf = &buf[1..];
            let source = core::str::from_utf8(&buf[..source_len])?;
            let text = core::str::from_utf8(&buf[source_len..])?;
            Ok(Self {
                log_level,
                source,
                text,
            })
        }
    }

    impl DataType for LogMessageType<'_> {
        const FULL_NAME: &'static str = "uavcan.protocol.debug.LogMessage";
        const ID: u16 = 16383;
        const SIGNATURE: u64 = 0xd654a48e0c049d75;
        const MAX_BIT_LEN: usize = 983;
    }
}
