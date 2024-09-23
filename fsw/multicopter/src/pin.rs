use hal::gpio;

use crate::peripheral::*;

pub trait Pin {
    const PORT: gpio::Port;
    const PIN: u8;

    fn set<T: PinFunction<Pin = Self>>(_pf: &T) {
        let _ = gpio::Pin::new(Self::PORT, Self::PIN, T::MODE);
    }
}

pub trait PinFunction {
    type Pin: Pin;
    const MODE: gpio::PinMode;
}

pub struct PA8 {}
pub struct PA9 {}
pub struct PA10 {}
pub struct PA11 {}

pub struct PC6 {}
pub struct PC7 {}
pub struct PC8 {}
pub struct PC9 {}

macro_rules! impl_pin {
    ($port:ident, $pin_num:literal) => {
        paste::paste! {
        impl Pin for [<P $port $pin_num>] {
            const PORT: gpio::Port = gpio::Port::$port;
            const PIN: u8 = $pin_num;
        }
        }
    };
}

impl_pin!(A, 8);
impl_pin!(A, 9);
impl_pin!(A, 10);
impl_pin!(A, 11);

impl_pin!(C, 6);
impl_pin!(C, 7);
impl_pin!(C, 8);
impl_pin!(C, 9);

macro_rules! impl_af {
    ($af:ident, $pin:ident, $mode:literal) => {
        impl<'a> PinFunction for $af<'a> {
            type Pin = $pin;
            const MODE: gpio::PinMode = gpio::PinMode::Alt($mode);
        }
    };
}

impl_af!(Tim1Ch1, PA8, 1);
impl_af!(Tim1Ch2, PA9, 1);
impl_af!(Tim1Ch3, PA10, 1);
impl_af!(Tim1Ch4, PA11, 1);

impl_af!(Tim3Ch1, PC6, 2);
impl_af!(Tim3Ch2, PC7, 2);
impl_af!(Tim3Ch3, PC8, 2);
impl_af!(Tim3Ch4, PC9, 2);
