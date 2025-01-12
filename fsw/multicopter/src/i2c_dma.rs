use alloc::boxed::Box;
use core::ops::{Deref, DerefMut};

use hal::{clocks, dma, i2c, pac};

use crate::dma::DmaChannel;

const MAX_ITERS: u32 = 300_000;

#[derive(Copy, Clone)]
enum I2cPperipheral {
    I2C1,
    I2C2,
    I2C3,
    I2C4,
}

// Type-erased I2C peripheral
#[derive(Copy, Clone)]
pub struct I2cRegs {
    peripheral: I2cPperipheral,
    regs: *const pac::i2c1::RegisterBlock,
}

impl Deref for I2cRegs {
    type Target = pac::i2c1::RegisterBlock;
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.regs }
    }
}

impl From<pac::I2C1> for I2cRegs {
    fn from(_: pac::I2C1) -> Self {
        I2cRegs {
            peripheral: I2cPperipheral::I2C1,
            regs: pac::I2C1::ptr(),
        }
    }
}

impl From<pac::I2C2> for I2cRegs {
    fn from(_: pac::I2C2) -> Self {
        I2cRegs {
            peripheral: I2cPperipheral::I2C2,
            regs: pac::I2C2::ptr(),
        }
    }
}

impl From<pac::I2C3> for I2cRegs {
    fn from(_: pac::I2C3) -> Self {
        I2cRegs {
            peripheral: I2cPperipheral::I2C3,
            regs: pac::I2C3::ptr(),
        }
    }
}

impl From<pac::I2C4> for I2cRegs {
    fn from(_: pac::I2C4) -> Self {
        I2cRegs {
            peripheral: I2cPperipheral::I2C4,
            regs: pac::I2C4::ptr(),
        }
    }
}

#[derive(Debug, defmt::Format, Copy, Clone, PartialEq, Eq)]
pub enum State {
    Idle,
    Reading,
    Done,
}

pub struct I2cDma {
    i2c: i2c::I2c<I2cRegs>,
    dma: DmaChannel,
    read_buf: Box<[u8]>,
    read_len: usize,
}

impl Deref for I2cDma {
    type Target = i2c::I2c<I2cRegs>;
    fn deref(&self) -> &Self::Target {
        &self.i2c
    }
}

impl DerefMut for I2cDma {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.i2c
    }
}

#[derive(Debug, defmt::Format)]
pub enum Error {
    I2c(i2c::Error),
    ReadNotStarted,
    Busy,
}

impl From<i2c::Error> for Error {
    fn from(e: i2c::Error) -> Self {
        Error::I2c(e)
    }
}

impl I2cDma {
    pub fn new<R: Into<I2cRegs> + hal::RccPeriph + Deref<Target = pac::i2c1::RegisterBlock>>(
        i2c: R,
        config: i2c::I2cConfig,
        mut dma_channel: DmaChannel,
        clocks: &clocks::Clocks,
        mux1: &mut pac::DMAMUX1,
        mux2: &mut pac::DMAMUX2,
    ) -> Self {
        let i2c = i2c::I2c::new(i2c, config.clone(), clocks);
        let i2c::I2c { regs, cfg } = i2c;
        let i2c = i2c::I2c {
            regs: regs.into(),
            cfg,
        };
        match i2c.regs.peripheral {
            I2cPperipheral::I2C1 => dma_channel.mux_dma1(dma::DmaInput::I2c1Rx, mux1),
            I2cPperipheral::I2C2 => dma_channel.mux_dma1(dma::DmaInput::I2c2Rx, mux1),
            I2cPperipheral::I2C3 => dma_channel.mux_dma1(dma::DmaInput::I2c3Rx, mux1),
            I2cPperipheral::I2C4 => dma_channel.mux_dma2(dma::DmaInput2::I2c4Rx, mux2),
        };
        let read_buf = alloc::vec![0u8; 256].into_boxed_slice();
        I2cDma {
            i2c,
            dma: dma_channel,
            read_buf,
            read_len: 0,
        }
    }

    pub fn enable(&mut self) {
        // Enable DMA requests
        self.i2c.regs.cr1.modify(|_, w| w.rxdmaen().set_bit());
        // Enable DMA channel
        self.dma.enable();
        // Enable I2C peripheral
        self.i2c.regs.cr1.modify(|_, w| w.pe().set_bit());
        while self.i2c.regs.cr1.read().pe().bit_is_clear() {}
    }

    pub fn disable(&mut self) {
        // Disable DMA requests
        self.i2c.regs.cr1.modify(|_, w| w.rxdmaen().clear_bit());
        // Disable DMA channel
        self.dma.disable();
        self.dma.clear_interrupt();
        // Disable I2C peripheral
        self.i2c.regs.cr1.modify(|_, w| w.pe().clear_bit());
        while self.i2c.regs.cr1.read().pe().bit_is_set() {}
    }

    fn wait_txis(&mut self) -> Result<(), Error> {
        let mut i = 0;
        while self.i2c.regs.isr.read().txis().bit_is_clear() {
            i += 1;
            if i >= MAX_ITERS {
                return Err(Error::I2c(i2c::Error::Hardware));
            }
        }
        Ok(())
    }

    pub fn state(&mut self) -> Result<State, Error> {
        // First, check for errors
        let isr = self.i2c.regs.isr.read();
        if isr.berr().bit_is_set() {
            self.i2c.regs.icr.write(|w| w.berrcf().set_bit());
            return Err(Error::I2c(i2c::Error::Bus));
        } else if isr.arlo().bit_is_set() {
            self.i2c.regs.icr.write(|w| w.arlocf().set_bit());
            return Err(Error::I2c(i2c::Error::Arbitration));
        } else if isr.nackf().bit_is_set() {
            self.i2c
                .regs
                .icr
                .write(|w| w.stopcf().set_bit().nackcf().set_bit());
            // If a pending TXIS flag is set, write dummy data to TXDR
            if self.i2c.regs.isr.read().txis().bit_is_set() {
                self.i2c.regs.txdr.write(|w| w.txdata().bits(0));
            }
            // If TXDR is not flagged as empty, write 1 to flush it
            if self.i2c.regs.isr.read().txe().bit_is_clear() {
                self.i2c.regs.isr.write(|w| w.txe().set_bit());
            }
            return Err(Error::I2c(i2c::Error::Nack));
        }

        // Then, check for completion
        let state = if self.read_len == 0 {
            State::Idle
        } else if self.dma.busy() {
            State::Reading
        } else {
            State::Done
        };
        Ok(state)
    }

    pub fn begin_read(&mut self, addr: u8, register: u8, read_len: usize) -> Result<(), Error> {
        if self.state()? == State::Reading {
            return Err(Error::Busy);
        }
        self.read_len = read_len;
        defmt::trace!("Setting I2C CR2 to write");
        self.i2c.regs.cr2.write(|w| {
            w.add10().bit(self.i2c.cfg.address_bits as u8 != 0);
            w.sadd().bits((addr << 1) as u16);
            w.rd_wrn().write();
            w.nbytes().bits(1);
            w.autoend().bit(false);
            w.pecbyte().bit(self.i2c.cfg.smbus);
            w.start().set_bit()
        });

        defmt::trace!("Waiting for I2C TXIS");
        self.wait_txis()?;
        defmt::trace!("Writing register to I2C TXDR");
        self.i2c.regs.txdr.write(|w| w.txdata().bits(register));

        defmt::trace!("Initiating DMA read");
        unsafe {
            self.i2c.read_dma(
                addr,
                &mut self.read_buf[..self.read_len],
                self.dma.channel,
                Default::default(),
                self.dma.peripheral,
            )
        }

        Ok(())
    }

    pub fn finish_read(&mut self) -> Result<&[u8], Error> {
        match self.state()? {
            State::Idle => Err(Error::ReadNotStarted),
            State::Reading => Err(Error::Busy),
            State::Done => {
                let buf = &self.read_buf[..self.read_len];
                self.read_len = 0;
                self.dma.clear_interrupt();
                Ok(buf)
            }
        }
    }
}

impl Drop for I2cDma {
    fn drop(&mut self) {
        self.disable();
    }
}
