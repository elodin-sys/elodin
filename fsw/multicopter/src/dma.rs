use core::ops::Deref;

use hal::{dma, pac};

#[derive(Copy, Clone)]
pub struct DmaRegs {
    peripheral: dma::DmaPeriph,
    pub regs: *const pac::dma1::RegisterBlock,
}

impl Deref for DmaRegs {
    type Target = pac::dma1::RegisterBlock;
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.regs }
    }
}

impl From<pac::DMA1> for DmaRegs {
    fn from(_: pac::DMA1) -> Self {
        DmaRegs {
            peripheral: dma::DmaPeriph::Dma1,
            regs: pac::DMA1::ptr(),
        }
    }
}

pub trait DmaSplit {
    fn split(self) -> [DmaChannel; 8];
}

impl DmaSplit for pac::DMA1 {
    fn split(self) -> [DmaChannel; 8] {
        let dma_channels = [
            dma::DmaChannel::C0,
            dma::DmaChannel::C1,
            dma::DmaChannel::C2,
            dma::DmaChannel::C3,
            dma::DmaChannel::C4,
            dma::DmaChannel::C5,
            dma::DmaChannel::C6,
            dma::DmaChannel::C7,
        ];
        let dma = dma::Dma::new(DmaRegs::from(self));
        dma_channels.map(|channel| DmaChannel {
            dma: dma::Dma { regs: dma.regs },
            peripheral: dma.regs.peripheral,
            channel,
        })
    }
}

pub struct DmaChannel {
    dma: dma::Dma<DmaRegs>,
    pub peripheral: dma::DmaPeriph,
    pub channel: dma::DmaChannel,
}

impl DmaChannel {
    pub fn mux_dma1(&mut self, input: dma::DmaInput, _mux: &mut pac::DMAMUX1) {
        dma::mux(self.peripheral, self.channel, input);
    }

    pub fn mux_dma2(&mut self, input: dma::DmaInput2, mux2: &mut pac::DMAMUX2) {
        dma::mux2(self.peripheral, self.channel, input, mux2);
    }

    pub fn clear_interrupt(&mut self) {
        self.dma
            .clear_interrupt(self.channel, dma::DmaInterrupt::TransferComplete);
    }

    pub fn transfer_complete(&mut self) -> bool {
        self.dma.transfer_is_complete(self.channel)
    }

    pub fn busy(&mut self) -> bool {
        self.remaining() > 0
    }

    fn channel_index(&self) -> usize {
        self.channel as u8 as usize
    }

    fn regs(&mut self) -> &pac::dma1::RegisterBlock {
        &self.dma.regs
    }

    fn st(&mut self) -> &pac::dma1::ST {
        let i = self.channel_index();
        &self.regs().st[i]
    }

    fn remaining(&mut self) -> usize {
        self.st().ndtr.read().ndt().bits() as usize
    }

    pub fn enable(&mut self) {
        self.st().cr.modify(|_, w| w.en().set_bit());
        while self.st().cr.read().en().bit_is_clear() {}
    }

    pub fn disable(&mut self) {
        self.st().cr.modify(|_, w| w.en().clear_bit());
        while self.st().cr.read().en().bit_is_set() {}
    }
}
