use dshot_frame::{Command, Frame};
use embedded_hal::delay::DelayNs;
use hal::timer;

use crate::{arena::DmaAlloc, dma::DmaBuf, peripheral::*};

const DSHOT_FRAME_SIZE: usize = 16;
// Size the DMA buffer to hold a DSHOT frame + 0 padding (as a gap between frames) for 4 motors.
const DMA_BUF_SIZE: usize = (DSHOT_FRAME_SIZE + 1) * 4;

#[derive(Clone, Copy, Debug, Default)]
pub struct Throttle(u16);

impl From<f32> for Throttle {
    fn from(throttle: f32) -> Self {
        Self((throttle.clamp(0.0, 1.0) * 1999.0) as u16)
    }
}

pub struct Driver<T, D>
where
    D: HalDmaRegExt,
{
    pub pwm_timer: timer::Timer<T>,
    pub dma_buf: DmaBuf<DMA_BUF_SIZE, D>,
    max_duty_cycle: u16,
}

impl<T, D> Driver<T, D>
where
    T: HalTimerRegExt,
    D: HalDmaRegExt,
    timer::Timer<T>: HalTimerExt<HalTimerReg = T> + DmaMuxInput,
{
    pub fn new<B: AsMut<[u8]> + 'static>(
        mut pwm_timer: timer::Timer<T>,
        dma: DmaChannel<D>,
        alloc: &mut DmaAlloc<B>,
    ) -> Self {
        pwm_timer.enable_pwm();
        pwm_timer.enable_dma_interrupt();
        dma.mux(&mut pwm_timer);
        let max_duty_cycle = pwm_timer.max_duty_cycle() as u16;
        Self {
            pwm_timer,
            dma_buf: DmaBuf::new(dma, alloc),
            max_duty_cycle,
        }
    }

    fn write_frame(&mut self, motor_index: usize, frame: Frame) {
        let duty_cycles = frame.duty_cycles(self.max_duty_cycle);
        for (duty_cycle_index, duty_cycle) in duty_cycles.into_iter().enumerate() {
            self.dma_buf.staging_buf[4 * duty_cycle_index + motor_index] = duty_cycle;
        }
    }

    pub fn arm_motors<Delay: DelayNs>(&mut self, delay: &mut Delay) {
        defmt::info!("Arming motors");
        // Just send 0 throttle for a while, which seems to work with both AM32 and BLHeli_32 ESCs.
        // Unfortunately, the DSHOT protocol doesn't provide a way to detect when the ESC is ready.
        // TODO(Akhil): Replace this with a state machine, but that requires time-keeping which I'll add later.
        for _ in 0..(5 * 1000) {
            self.write_throttle([Throttle(0); 4]);
            delay.delay_ms(1);
        }
        defmt::info!("Motors armed");
    }

    pub fn write_throttle(&mut self, throttle: [Throttle; 4]) {
        for (i, throttle) in throttle.into_iter().enumerate() {
            let frame = Frame::new(throttle.0, false).unwrap();
            self.write_frame(i, frame);
        }
        self.pwm_timer.write(&mut self.dma_buf);
    }

    pub fn beep<Delay: DelayNs>(&mut self, delay: &mut Delay) {
        defmt::info!("Beeping motors");
        let frame = Frame::command(Command::Beep1, false);
        (0..4).for_each(|motor_index| self.write_frame(motor_index, frame));
        self.pwm_timer.write(&mut self.dma_buf);
        delay.delay_ms(260);
        defmt::info!("Motors beeped");
    }
}
