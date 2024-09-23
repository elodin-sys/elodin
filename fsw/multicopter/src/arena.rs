use core::mem::MaybeUninit;

#[link_section = ".axisram.buffers"]
static mut DMA_BUF: MaybeUninit<[u8; 4096]> = MaybeUninit::uninit();
static mut TAKEN: bool = false;

/// A simple bump allocator for DMA buffers backed by memory that's been reserved in a DMA-accessible region.
/// This allocator does not support deallocation. So, it just leaks memory which is actually fine for DMA buffers.
pub struct DmaAlloc<B> {
    buf: B,
    pos: usize,
}

impl DmaAlloc<&'static mut [u8; 4096]> {
    pub fn take() -> Self {
        // SAFETY: A mutable reference to the static is safe to create because it cannot be aliased.
        let buf = critical_section::with(|_| unsafe {
            if TAKEN {
                defmt::panic!("DMA alloc already taken");
            }
            TAKEN = true;
            let buf = &mut *(core::ptr::addr_of_mut!(DMA_BUF) as *mut [MaybeUninit<u8>; 4096]);
            // zero out the buffer
            for value in buf.iter_mut() {
                value.as_mut_ptr().write(0);
            }
            DMA_BUF.assume_init_mut()
        });
        Self::new(buf)
    }
}

impl<B: AsMut<[u8]>> DmaAlloc<B> {
    const fn new(buf: B) -> Self {
        Self { buf, pos: 0 }
    }

    pub fn leak<'a, T>(&mut self, val: T) -> &'a mut T {
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
