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
    pub fn read(&self) -> T {
        self.0.lock().clone()
    }
}

impl<T: Default + Clone> Mailbox<(T, u64)> {
    pub fn write(&self, val: T, time: u64) {
        *self.0.lock() = (val, time);
    }

    pub fn read_msg(&self) -> T {
        self.0.lock().clone().0
    }
}

const CHANNEL_MSG_HEADER: MsgHeader = MsgHeader {
    isLinked: 0x2241,
    isWritten: 0x2241,
    timeWritten: 0x2241,
    moduleID: 0x2241,
};

#[derive(Clone, Default)]
pub struct BskChannel<T>(Arc<Mailbox<(T, u64)>>);

impl<T> std::ops::Deref for BskChannel<T> {
    type Target = Mailbox<(T, u64)>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Default + Clone> BskChannel<T> {
    pub fn pair() -> Self {
        Self::default()
    }

    fn into_raw(self) -> *const Mailbox<(T, u64)> {
        Arc::into_raw(self.0)
    }

    /// # Safety
    ///
    /// This function is safe to call as long as the pointer is valid.
    /// The raw pointer must have been previously returned by a call to `into_raw`.
    pub unsafe fn from_raw(ptr: *mut Mailbox<(T, u64)>) -> Self {
        Self(unsafe { Arc::from_raw(ptr) })
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
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<$msg_name _write>](
                data: *const $payload_name,
                channel: *mut $msg_name,
                _module_id: i64,
                call_time: u64
            ) {
                let channel = unsafe { channel.as_ref() }.unwrap();
                if channel.header != CHANNEL_MSG_HEADER {
                    return;
                }
                let Some(data) = (unsafe { data.as_ref() }) else {
                    tracing::warn!("watcha doing passing null ptrs to write, you know better than that");
                    return;
                };
                let mailbox = channel.payloadPointer as *const Mailbox<($payload_name, u64)>;
                unsafe { mailbox.as_ref() }.unwrap().write(data.clone(), call_time);
            }
        }

        paste! {
            /// Basilisk function to read msg from channel
            ///
            /// # Safety
            /// Don't call this yourself, Basilisk will call it for you
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<$msg_name _read>](channel: *mut $msg_name) -> $payload_name {
                let channel = unsafe { channel.as_ref() }.unwrap();
                if channel.header != CHANNEL_MSG_HEADER {
                    return $payload_name::default();
                }
                let mailbox = channel.payloadPointer as *const Mailbox<($payload_name, u64)>;
                unsafe { mailbox.as_ref() }.unwrap().read().0
            }
        }

        paste! {
            /// Basilisk function to read msg from channel
            ///
            /// # Safety
            /// Don't call this yourself, Basilisk will call it for you
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<$msg_name _timeWritten>](channel: *mut $msg_name) -> u64 {
                let channel = unsafe { channel.as_ref() }.unwrap();
                if channel.header != CHANNEL_MSG_HEADER {
                    return u64::MAX;
                }
                let mailbox = channel.payloadPointer as *const Mailbox<($payload_name, u64)>;
                unsafe { mailbox.as_ref() }.unwrap().read().1
            }
        }

        paste! {
            /// Basilisk function to check if channel is open
            ///
            /// # Safety
            /// Don't call this yourself, Basilisk will call it for you
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<$msg_name _isLinked>](channel: *mut $msg_name) -> bool {
                unsafe { channel.as_ref() }.is_some_and(|channel| channel.header == CHANNEL_MSG_HEADER)
            }
        }

        paste! {
            /// Basilisk function to check if channel is written
            ///
            /// # Safety
            /// Don't call this yourself, Basilisk will call it for you
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<$msg_name _isWritten>](channel: *mut $msg_name) -> bool {
                unsafe { channel.as_ref() }.is_some_and(|channel| channel.header == CHANNEL_MSG_HEADER)
            }
        }




        paste! {
            #[unsafe(no_mangle)]
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
impl_basilisk_channel!(NavAttMsg_C, NavAttMsgPayload);
impl_basilisk_channel!(SunlineFilterMsg_C, SunlineFilterMsgPayload);
impl_basilisk_channel!(CSSArraySensorMsg_C, CSSArraySensorMsgPayload);
impl_basilisk_channel!(CSSConfigMsg_C, CSSConfigMsgPayload);
impl_basilisk_channel!(AttRefMsg_C, AttRefMsgPayload);
impl_basilisk_channel!(NavTransMsg_C, NavTransMsgPayload);
impl_basilisk_channel!(EphemerisMsg_C, EphemerisMsgPayload);
impl_basilisk_channel!(ArrayMotorTorqueMsg_C, ArrayMotorTorqueMsgPayload);
impl_basilisk_channel!(ArrayMotorVoltageMsg_C, ArrayMotorVoltageMsgPayload);
