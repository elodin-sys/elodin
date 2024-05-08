use paste::paste;
use std::sync::Arc;
use std::{fmt::Debug, mem::MaybeUninit};
use thingbuf::recycling::DefaultRecycle;
use thingbuf::Recycle;

use crate::sys::*;

// Create a pair of `Tx` and `Rx` channels
pub fn pair<T: Clone + Default>() -> (Tx<T>, Rx<T>) {
    let (tx, rx) = thingbuf::mpsc::channel(2);
    (
        Tx {
            inner: Arc::new(tx),
        },
        Rx {
            inner: Arc::new(rx),
            last_msg: None,
        },
    )
}

/// The send half of a channel
pub struct Tx<T> {
    inner: std::sync::Arc<thingbuf::mpsc::Sender<T>>,
}

impl<T> Clone for Tx<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T: Default> Tx<T>
where
    DefaultRecycle: Recycle<T>,
{
    fn write(&self, msg: T) {
        if self.inner.try_send(msg).is_err() {
            tracing::warn!("bsk channel closed")
        }
    }
}

/// The receive half of a channel
pub struct Rx<T> {
    inner: Arc<thingbuf::mpsc::Receiver<T>>,
    last_msg: Option<T>,
}

impl<T> Clone for Rx<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            last_msg: None,
        }
    }
}

impl<T: Clone + Debug + Default> Rx<T> {
    /// Internal function used by macro to generate basilisk compat interface, hence the strange naming
    fn is_linked(&self) -> bool {
        !self.inner.is_closed()
    }

    fn read(&mut self) -> Option<T> {
        if let Ok(new_msg) = self.inner.try_recv() {
            self.last_msg = Some(new_msg.clone());
            Some(new_msg)
        } else {
            self.last_msg.clone()
        }
    }
}

const CHANNEL_MSG_HEADER: MsgHeader = MsgHeader {
    isLinked: 0x2241,
    isWritten: 0x2241,
    timeWritten: 0x2241,
    moduleID: 0x2241,
};

/// A struct that is designed to be byte compatible with the `FooMsg_C` structs in Basilisk
#[repr(C)]
pub struct BskChannel<T> {
    header: MsgHeader,
    _payload: MaybeUninit<T>,
    rx_ptr: *mut Rx<T>,
    tx_ptr: *mut Tx<T>,
}

impl<T: Clone + Debug + Default> Clone for BskChannel<T> {
    fn clone(&self) -> Self {
        let tx = Box::new(self.tx().unwrap().clone());
        let rx = Box::new(self.rx().unwrap().clone());
        Self {
            header: self.header,
            _payload: MaybeUninit::zeroed(),
            rx_ptr: Box::into_raw(rx),
            tx_ptr: Box::into_raw(tx),
        }
    }
}

impl<T: Clone + Debug + Default> BskChannel<T> {
    pub fn pair() -> Self {
        let (tx, rx) = crate::channel::pair();
        BskChannel::new(tx, rx)
    }

    pub fn new(tx: Tx<T>, rx: Rx<T>) -> Self {
        let tx = Box::new(tx);
        let rx = Box::new(rx);
        BskChannel {
            header: CHANNEL_MSG_HEADER,
            _payload: MaybeUninit::zeroed(),
            rx_ptr: Box::into_raw(rx),
            tx_ptr: Box::into_raw(tx),
        }
    }

    fn validate_header(&self) -> Option<()> {
        if self.header != CHANNEL_MSG_HEADER {
            tracing::warn!("invalid bsk channel");
            return None;
        }
        Some(())
    }

    fn rx_mut(&mut self) -> Option<&mut Rx<T>> {
        self.validate_header()?;
        let rx = unsafe { &mut *self.rx_ptr };
        Some(rx)
    }

    pub fn rx(&self) -> Option<&Rx<T>> {
        self.validate_header()?;
        let rx = unsafe { &*self.rx_ptr };
        Some(rx)
    }

    pub fn tx(&self) -> Option<&Tx<T>> {
        self.validate_header()?;
        let tx = unsafe { &*self.tx_ptr };
        Some(tx)
    }

    pub fn read(&mut self) -> Option<T> {
        let rx = self.rx_mut()?;
        rx.read()
    }

    pub fn write(&self, msg: T) -> Option<()> {
        let tx = self.tx()?;
        tx.write(msg);
        Some(())
    }

    fn is_linked(&self) -> bool {
        self.rx().map(|rx| rx.is_linked()).unwrap_or_default()
    }
}

macro_rules! impl_basilisk_channel {
    ($msg_name:tt, $payload_name:tt) => {
        impl From<BskChannel<$payload_name>> for $msg_name {
            fn from(val: BskChannel<$payload_name>) -> $msg_name {
                unsafe { std::mem::transmute(val) }
            }
        }

        paste! {
            /// Basilisk function to write msg to channel
            ///
            /// # Safety
            /// Don't call this yourself, Basilisk will call it for you
            #[no_mangle]
            pub unsafe extern "C" fn [<$msg_name _write>](
                data: *const $payload_name,
                channel: *mut $msg_name,
                _module_id: i64,
                _call_time: u64,
            ) {
                let channel: *mut BskChannel<$payload_name> =
                    unsafe { std::mem::transmute(channel) };
                let channel: &mut BskChannel<$payload_name> = unsafe { &mut *channel };
                if data.is_null() {
                    tracing::warn!("watcha doin passing null ptrs to write, you know better than that");
                    return;
                }
                let data = unsafe { &*data };
                channel.write(data.clone());
            }
        }

        paste! {
            /// Basilisk function to read msg from channel
            ///
            /// # Safety
            /// Don't call this yourself, Basilisk will call it for you
            #[no_mangle]
            pub unsafe extern "C" fn [<$msg_name _read>](channel: *mut $msg_name) -> $payload_name {
                let channel: *mut BskChannel<$payload_name> =
                    unsafe { std::mem::transmute(channel) };
                let channel: &mut BskChannel<$payload_name> = unsafe { &mut *channel };
                channel.read().unwrap_or_default()
            }
        }

        paste! {
            /// Basilisk function to check if channel is open
            ///
            /// # Safety
            /// Don't call this yourself, Basilisk will call it for you
            #[no_mangle]
            pub unsafe extern "C" fn [<$msg_name _isLinked>](channel: *mut $msg_name) -> bool {
                let channel: *mut BskChannel<$payload_name> =
                    unsafe { std::mem::transmute(channel) };
                let channel: &mut BskChannel<$payload_name> = unsafe { &mut *channel };
                channel.is_linked()
            }
        }


        paste! {
            #[no_mangle]
            pub extern "C" fn [<$msg_name _zeroMsgPayload>]() -> $payload_name {
                $payload_name::default()
            }
        }
    };
}

impl_basilisk_channel!(AttGuidMsg_C, AttGuidMsgPayload);
impl_basilisk_channel!(RateCmdMsg_C, RateCmdMsgPayload);
impl_basilisk_channel!(RWSpeedMsg_C, RWSpeedMsgPayload);
impl_basilisk_channel!(RWAvailabilityMsg_C, RWAvailabilityMsgPayload);
impl_basilisk_channel!(RWArrayConfigMsg_C, RWArrayConfigMsgPayload);
impl_basilisk_channel!(CmdTorqueBodyMsg_C, CmdTorqueBodyMsgPayload);
impl_basilisk_channel!(VehicleConfigMsg_C, VehicleConfigMsgPayload);
