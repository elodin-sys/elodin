#![no_main]
#![no_std]

use cortex_m_semihosting::debug;

use defmt_rtt as _;
use hal as _;
use panic_probe as _;

pub mod arena;
pub mod bsp;
pub mod dma;
pub mod dshot;
pub mod peripheral;
pub mod pin;

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
