#![no_main]
#![no_std]

extern crate alloc;

use cortex_m_semihosting::debug;

use defmt_rtt as _;
use hal as _;
use panic_probe as _;

pub mod blackbox;
pub mod bmm350;
pub mod bsp;
pub mod can;
pub mod crsf;
pub mod dma;
pub mod dronecan;
pub mod dshot;
pub mod healing_usart;
pub mod i2c_dma;
pub mod led;
pub mod monotonic;
pub mod peripheral;
pub mod rtc;
pub mod sdmmc;

#[global_allocator]
static HEAP: embedded_alloc::TlsfHeap = embedded_alloc::TlsfHeap::empty();

pub fn init_heap() {
    {
        use core::mem::MaybeUninit;
        const HEAP_SIZE: usize = 64 * 1024;
        #[link_section = ".axisram.buffers"]
        static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];
        unsafe { HEAP.init(HEAP_MEM.as_ptr() as usize, HEAP_SIZE) };
        defmt::info!("Configured heap with {} bytes", HEAP_SIZE);
    }
}

// same panicking *behavior* as `panic-probe` but doesn't print a panic message
// this prevents the panic message being printed *twice* when `defmt::panic` is invoked
#[defmt::panic_handler]
fn panic() -> ! {
    cortex_m::asm::udf()
}

pub fn exit() -> ! {
    loop {
        debug::exit(debug::EXIT_SUCCESS);
    }
}

#[allow(non_snake_case)]
#[cortex_m_rt::exception]
unsafe fn HardFault(_frame: &cortex_m_rt::ExceptionFrame) -> ! {
    loop {
        debug::exit(debug::EXIT_FAILURE);
    }
}
