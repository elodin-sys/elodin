use paste::paste;
use std::sync::Arc;

use crate::sys::*;

/// `Mailbox` is a thread-safe container for a single message.
///
/// A spin lock is appropriate here because we can effectively guarantee that
/// the lock will be held for a very short period of time (the duration of
/// copying a message).
#[derive(Default)]
pub struct Mailbox<T>(spin::Mutex<T>);

impl<T: Default + Clone> Mailbox<T> {
    pub fn write(&self, val: T) {
        *self.0.lock() = val;
    }

    pub fn read(&self) -> T {
        self.0.lock().clone()
    }
}

const CHANNEL_MSG_HEADER: MsgHeader = MsgHeader {
    isLinked: 0x2241,
    isWritten: 0x2241,
    timeWritten: 0x2241,
    moduleID: 0x2241,
};

#[derive(Clone, Default)]
pub struct BskChannel<T>(Arc<Mailbox<T>>);

impl<T> std::ops::Deref for BskChannel<T> {
    type Target = Mailbox<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Default + Clone> BskChannel<T> {
    pub fn pair() -> Self {
        Self::default()
    }

    fn into_raw(self) -> *const Mailbox<T> {
        Arc::into_raw(self.0)
    }

    /// # Safety
    ///
    /// This function is safe to call as long as the pointer is valid.
    /// The raw pointer must have been previously returned by a call to `into_raw`.
    pub unsafe fn from_raw(ptr: *mut Mailbox<T>) -> Self {
        Self(Arc::from_raw(ptr))
    }
}

macro_rules! impl_basilisk_channel {
    ($msg_name:tt, $payload_name:tt) => {
        impl From<BskChannel<$payload_name>> for $msg_name {
            fn from(val: BskChannel<$payload_name>) -> $msg_name {
                $msg_name {
                    header: CHANNEL_MSG_HEADER,
                    payload: $payload_name::default(),
                    payloadPointer: val.into_raw() as _,
                    headerPointer: std::ptr::null_mut(),
                }
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
                let channel = channel.as_ref().unwrap();
                if channel.header != CHANNEL_MSG_HEADER {
                    return;
                }
                let Some(data) = data.as_ref() else {
                    tracing::warn!("watcha doin passing null ptrs to write, you know better than that");
                    return;
                };
                let mailbox = channel.payloadPointer as *const Mailbox<$payload_name>;
                mailbox.as_ref().unwrap().write(data.clone());
            }
        }

        paste! {
            /// Basilisk function to read msg from channel
            ///
            /// # Safety
            /// Don't call this yourself, Basilisk will call it for you
            #[no_mangle]
            pub unsafe extern "C" fn [<$msg_name _read>](channel: *mut $msg_name) -> $payload_name {
                let channel = channel.as_ref().unwrap();
                if channel.header != CHANNEL_MSG_HEADER {
                    return $payload_name::default();
                }
                let mailbox = channel.payloadPointer as *const Mailbox<$payload_name>;
                mailbox.as_ref().unwrap().read()
            }
        }

        paste! {
            /// Basilisk function to check if channel is open
            ///
            /// # Safety
            /// Don't call this yourself, Basilisk will call it for you
            #[no_mangle]
            pub unsafe extern "C" fn [<$msg_name _isLinked>](channel: *mut $msg_name) -> bool {
                channel.as_ref().is_some_and(|channel| channel.header == CHANNEL_MSG_HEADER)
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
