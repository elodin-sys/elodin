use std::mem::MaybeUninit;

use socket2::SockAddr;

mod tcp;
pub use tcp::*;
mod udp;
pub use udp::*;

#[cfg(target_os = "windows")]
type SockAddrStorage = windows_sys::Win32::Networking::WinSock::SOCKADDR_STORAGE;
#[cfg(not(target_os = "windows"))]
type SockAddrStorage = libc::sockaddr_storage;

#[cfg(target_os = "windows")]
pub struct SockAddrRaw {
    pub(crate) storage: SockAddrStorage,
    pub(crate) len: i32,
}

#[cfg(not(target_os = "windows"))]
pub struct SockAddrRaw {
    pub(crate) storage: SockAddrStorage,
    pub(crate) len: u32,
}

impl SockAddrRaw {
    pub(crate) fn zeroed() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

impl From<SockAddrRaw> for SockAddr {
    fn from(val: SockAddrRaw) -> Self {
        unsafe { SockAddr::new(val.storage, val.len) }
    }
}

impl From<SockAddr> for SockAddrRaw {
    fn from(val: SockAddr) -> Self {
        let addr = val.as_ptr();
        let len = val.len();
        let mut storage = MaybeUninit::<SockAddrStorage>::zeroed();
        unsafe {
            std::ptr::copy_nonoverlapping(
                addr as *const _ as *const u8,
                storage.as_mut_ptr() as *mut u8,
                len as usize,
            )
        };

        SockAddrRaw {
            // This is safe as we written the address to `storage` above.
            storage: unsafe { storage.assume_init() },
            len,
        }
    }
}
