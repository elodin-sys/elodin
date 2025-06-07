#![no_main]
#![no_std]

extern crate alloc;

use core::ptr::addr_of_mut;
use cortex_m_semihosting::debug;

use defmt_rtt as _;
use hal as _;

pub mod adc;
pub mod blackbox;
pub mod bmi270;
pub mod bmm350;
pub mod bmp581;
pub mod bsp;
pub mod can;
pub mod command;
pub mod crsf;
pub mod dma;
pub mod dronecan;
pub mod dshot;
pub mod dwt;
pub mod fm24cl16b;
pub mod healing_usart;
pub mod i2c_dma;
pub mod led;
pub mod monitor;
pub mod monotonic;
pub mod peripheral;
pub mod sdmmc;
pub mod usb2513b;
pub mod usb_serial;

#[global_allocator]
static HEAP: embedded_alloc::TlsfHeap = embedded_alloc::TlsfHeap::empty();

pub fn init_heap() {
    {
        use core::mem::MaybeUninit;
        const HEAP_SIZE: usize = 128 * 1024;
        #[unsafe(link_section = ".axisram.buffers")]
        static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];
        unsafe { HEAP.init(addr_of_mut!(HEAP_MEM) as usize, HEAP_SIZE) };
        defmt::info!("Configured heap with {} bytes", HEAP_SIZE);
    }
}

use core::panic::PanicInfo;
use core::sync::atomic::{AtomicBool, Ordering};

static mut PANIC_LED: Option<hal::gpio::Pin> = None;

/// Initialize the LED that will be turned on during panic
pub fn set_panic_led(red_led: hal::gpio::Pin) {
    unsafe {
        PANIC_LED = Some(red_led);
    }
    defmt::info!("Panic LED initialized");
}

// The function to actually set the LED high during a panic
pub fn set_led_on_panic() {
    unsafe {
        if let Some(ref mut led) = PANIC_LED {
            led.set_high();
        }
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    static PANICKED: AtomicBool = AtomicBool::new(false);
    cortex_m::interrupt::disable();

    // Guard against infinite recursion
    if !PANICKED.load(Ordering::Relaxed) {
        PANICKED.store(true, Ordering::Relaxed);
        set_led_on_panic();
        defmt::error!("PANIC: {}", defmt::Display2Format(info));
    }
    cortex_m::asm::udf()
}

/// Handler for defmt panic to avoid double-prints
#[defmt::panic_handler]
fn defmt_panic() -> ! {
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
