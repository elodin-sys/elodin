use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicBool, Ordering};

#[link_section = ".axisram.buffers"]
static mut DMA_BUF: MaybeUninit<[u8; 4096]> = MaybeUninit::uninit();
static TAKEN: AtomicBool = AtomicBool::new(false);

/// A simple bump allocator for DMA buffers backed by memory that's been reserved in a DMA-accessible region.
/// This allocator does not support deallocation. So, it just leaks memory which is actually fine for DMA buffers.
pub struct ArenaAlloc<B> {
    buf: B,
    pos: usize,
}

impl ArenaAlloc<&'static mut [u8; 4096]> {
    pub fn take() -> Self {
        let buf = critical_section::with(|_| {
            if TAKEN.swap(true, Ordering::AcqRel) {
                defmt::panic!("DMA alloc already taken");
            }
            // SAFETY: A mutable reference to the static is safe to create because it cannot be aliased.
            unsafe { DMA_BUF.write([0u8; 4096]) }
        });
        Self::new(buf)
    }
}

impl<'a, B: AsMut<[u8]> + 'a> ArenaAlloc<B> {
    const fn new(buf: B) -> Self {
        Self { buf, pos: 0 }
    }

    pub fn leak<T>(&mut self, val: T) -> &'a mut T {
        let buf = self.buf.as_mut();
        let size = core::mem::size_of::<T>();
        let align = core::mem::align_of::<T>();
        let addr = buf.as_ptr() as usize + self.pos;
        let align_padding = (align - (addr % align)) % align;
        let start = self.pos + align_padding;
        if start + size > buf.len() {
            defmt::panic!("arena overflow");
        }
        self.pos = start + size;
        let ptr = core::ptr::addr_of_mut!(buf[start]) as *mut T;
        // SAFETY: The memory is being leaked, so it's fine to return a mutable reference to it with an independent lifetime.
        unsafe {
            ptr.write(val);
            &mut *ptr
        }
    }
}
