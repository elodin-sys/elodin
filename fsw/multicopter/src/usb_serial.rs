use core::mem::MaybeUninit;
use hal::{
    clocks, gpio, pac,
    usb_otg::{Usb2, UsbBus},
};
use usb_device::bus::UsbBusAllocator;
use usb_device::{
    device::{UsbDeviceBuilder, UsbVidPid},
    prelude::UsbDevice,
};
use usbd_serial::SerialPort;

static mut EP_MEMORY: MaybeUninit<[u32; 1024]> = MaybeUninit::uninit();

pub struct UsbSerial<'a> {
    serial: SerialPort<'a, UsbBus<Usb2>>,
    device: UsbDevice<'a, UsbBus<Usb2>>,
}

pub fn usb_bus(
    otg_hs: pac::OTG2_HS_GLOBAL,
    otg_hs_dev: pac::OTG2_HS_DEVICE,
    otg2_hs_pwclk: pac::OTG2_HS_PWRCLK,
    clocks: &clocks::Clocks,
    mut pin_dm: gpio::Pin,
    mut pin_dp: gpio::Pin,
) -> UsbBusAllocator<UsbBus<Usb2>> {
    pin_dm.mode(gpio::PinMode::Alt(10));
    pin_dp.mode(gpio::PinMode::Alt(10));

    // Initialise EP_MEMORY to zero
    unsafe {
        let buf: &mut [MaybeUninit<u32>; 1024] =
            &mut *(core::ptr::addr_of_mut!(EP_MEMORY) as *mut _);
        for value in buf.iter_mut() {
            value.as_mut_ptr().write(0);
        }
    }
    // Create USB bus allocator using high-speed peripheral
    UsbBus::new(
        Usb2::new(otg_hs, otg_hs_dev, otg2_hs_pwclk, clocks.hclk()),
        #[allow(static_mut_refs)]
        unsafe {
            EP_MEMORY.assume_init_mut()
        },
    )
}

impl<'a> UsbSerial<'a> {
    pub fn new(usb_bus: &'a UsbBusAllocator<UsbBus<Usb2>>) -> Self {
        // Create serial port device
        let serial = SerialPort::new(usb_bus);
        defmt::info!("created serial port");

        // Create USB device with VID/PID identifying as a CDC ACM serial port
        let device = UsbDeviceBuilder::new(usb_bus, UsbVidPid(0x16c0, 0x27dd))
            .manufacturer("Elodin")
            .product("Aleph Carrier Serial Port")
            .serial_number("224C1")
            .device_class(usbd_serial::USB_CLASS_CDC)
            .build();
        defmt::info!("created usb device");
        Self { serial, device }
    }

    pub fn poll(&mut self) -> bool {
        self.device.poll(&mut [&mut self.serial])
    }

    pub fn read(&mut self, buf: &mut [u8]) -> Result<usize, usb_device::UsbError> {
        self.serial.read(buf)
    }

    pub fn write(&mut self, buf: &[u8]) -> Result<usize, usb_device::UsbError> {
        self.serial.write(buf)
    }

    pub fn flush(&mut self) -> Result<(), usb_device::UsbError> {
        self.serial.flush()
    }
}
