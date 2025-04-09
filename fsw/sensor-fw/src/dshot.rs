use alloc::boxed::Box;

use dshot_frame::{Command, Frame};
use embedded_hal::delay::DelayNs;
use fugit::ExtU64 as _;
use hal::{pac, timer};

use crate::monotonic::Instant;
use crate::{dma::DmaChannel, peripheral::*};

const DSHOT_FRAME_SIZE: usize = 16;
// Size the DMA buffer to hold a DSHOT frame + 0 padding (as a gap between frames) for 4 motors.
const DMA_BUF_SIZE: usize = (DSHOT_FRAME_SIZE + 1) * 4;

pub const FRAME_TIME: fugit::MicrosDuration<u32> = fugit::MicrosDuration::<u32>::micros(27);
pub const INTER_FRAME_DELAY: fugit::MicrosDuration<u32> = fugit::MicrosDuration::<u32>::micros(40);

pub const UPDATE_RATE: fugit::Hertz<u32> = fugit::Hertz::<u32>::Hz(8000);
pub const UPDATE_PERIOD: fugit::MicrosDuration<u32> = UPDATE_RATE.into_duration();

#[derive(Clone, Copy, Debug, Default)]
pub struct Throttle(u16);

impl From<u16> for Throttle {
    fn from(throttle: u16) -> Self {
        Self(throttle.clamp(0, 1999))
    }
}

impl From<f32> for Throttle {
    fn from(throttle: f32) -> Self {
        Self::from((throttle * 1999.0) as u16)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArmState {
    Disarmed,
    Arming { end: Instant },
    Armed,
}

pub struct Driver<T> {
    pub pwm_timer: timer::Timer<T>,
    dma: DmaChannel,
    max_duty_cycle: u16,
    arm_state: ArmState,
    shared_buf: Box<[u16]>,
    staging_buf: Box<[u16]>,
}

impl<T> Driver<T>
where
    T: HalTimerRegExt,
    timer::Timer<T>: HalTimerExt<HalTimerReg = T> + DmaMuxInput,
{
    pub fn new(
        mut pwm_timer: timer::Timer<T>,
        mut dma: DmaChannel,
        mux1: &mut pac::DMAMUX1,
    ) -> Self {
        pwm_timer.enable_pwm();
        pwm_timer.enable_dma_interrupt();
        dma.mux_dma1(timer::Timer::<T>::DMA_INPUT, mux1);
        let max_duty_cycle = pwm_timer.max_duty_cycle() as u16;
        let shared_buf = alloc::vec![0u16; DMA_BUF_SIZE].into_boxed_slice();
        let staging_buf = alloc::vec![0u16; DMA_BUF_SIZE].into_boxed_slice();

        Self {
            pwm_timer,
            dma,
            max_duty_cycle,
            arm_state: ArmState::Disarmed,
            shared_buf,
            staging_buf,
        }
    }

    fn update_state(&mut self, armed: bool, now: Instant) {
        match (armed, self.arm_state) {
            (false, _) if self.arm_state != ArmState::Disarmed => {
                defmt::debug!("Disarming motors");
                self.arm_state = ArmState::Disarmed;
            }
            (true, ArmState::Disarmed) => {
                defmt::debug!("Arming motors");
                self.arm_state = ArmState::Arming {
                    end: now + 500.millis(),
                };
            }
            (true, ArmState::Arming { end }) => {
                if now > end {
                    self.arm_state = ArmState::Armed;
                    defmt::debug!("Motors armed");
                }
            }
            _ => {}
        }
    }

    fn write_frame(&mut self, motor_index: usize, frame: Frame) {
        let duty_cycles = frame.duty_cycles(self.max_duty_cycle);
        for (duty_cycle_index, duty_cycle) in duty_cycles.into_iter().enumerate() {
            self.staging_buf[4 * duty_cycle_index + motor_index] = duty_cycle;
        }
    }

    fn write(&mut self) -> bool {
        if self.dma.busy() {
            defmt::warn!("DMA transfer in progress");
            return false;
        }
        self.dma.clear_interrupt();

        let base_addr = TIMX_CCR1_OFFSET / 4;
        let burst_len = 4u8;
        assert_eq!(self.staging_buf.len() % burst_len as usize, 0);

        self.shared_buf.copy_from_slice(&self.staging_buf);
        unsafe {
            self.pwm_timer.hal_write_dma_burst(
                &self.shared_buf,
                base_addr,
                burst_len,
                self.dma.channel,
                Default::default(),
                false,
                self.dma.peripheral,
            );
        }
        true
    }

    pub fn write_throttle(&mut self, throttle: [Throttle; 4], armed: bool, now: Instant) {
        self.update_state(armed, now);
        for (i, throttle) in throttle.into_iter().enumerate() {
            let frame = if self.arm_state != ArmState::Armed {
                Frame::command(Command::MotorStop, false)
            } else {
                Frame::new(throttle.0, false).unwrap()
            };
            self.write_frame(i, frame);
        }
        self.write();
    }

    pub fn armed(&self) -> bool {
        matches!(self.arm_state, ArmState::Armed)
    }

    pub fn beep<Delay: DelayNs>(&mut self, delay: &mut Delay) {
        defmt::debug!("Beeping motors");
        let frame = Frame::command(Command::Beep1, false);
        (0..4).for_each(|motor_index| self.write_frame(motor_index, frame));
        self.write();
        delay.delay_ms(260);
        defmt::debug!("Motors beeped");
    }
}

impl<T> Drop for Driver<T> {
    fn drop(&mut self) {
        self.dma.disable();
    }
}
