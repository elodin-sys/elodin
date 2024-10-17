use hal::{gpio, pac};

pub trait Pin: Sized {
    const PORT: gpio::Port;
    const PIN: u8;

    fn set<T: PinFunction<Self>>(_pf: &T) -> gpio::Pin {
        gpio::Pin::new(Self::PORT, Self::PIN, T::MODE)
    }
}

pub trait PinFunction<P: Pin> {
    const MODE: gpio::PinMode;
}

pub struct PA8 {}
pub struct PA9 {}
pub struct PA10 {}
pub struct PA11 {}

pub struct PB6 {}
pub struct PB7 {}

pub struct PC6 {}
pub struct PC7 {}
pub struct PC8 {}
pub struct PC9 {}

pub struct PE11 {}
pub struct PE13 {}
pub struct PE14 {}

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

impl_pin!(B, 6);
impl_pin!(B, 7);

impl_pin!(C, 6);
impl_pin!(C, 7);
impl_pin!(C, 8);
impl_pin!(C, 9);

impl_pin!(E, 11);
impl_pin!(E, 13);
impl_pin!(E, 14);

macro_rules! impl_af {
    ($af:ty, $pin:ident, $mode:literal) => {
        impl PinFunction<$pin> for $af {
            const MODE: gpio::PinMode = gpio::PinMode::Alt($mode);
        }
    };
}

impl_af!(pac::TIM1, PA8, 1);
impl_af!(pac::TIM1, PA9, 1);
impl_af!(pac::TIM1, PA10, 1);
impl_af!(pac::TIM1, PA11, 1);

impl_af!(pac::TIM1, PE11, 1);
impl_af!(pac::TIM1, PE13, 1);
impl_af!(pac::TIM1, PE14, 1);

impl_af!(pac::TIM3, PC6, 2);
impl_af!(pac::TIM3, PC7, 2);
impl_af!(pac::TIM3, PC8, 2);
impl_af!(pac::TIM3, PC9, 2);

impl_af!(pac::I2C4, PB6, 6);
impl_af!(pac::I2C4, PB7, 6);
