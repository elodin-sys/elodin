use dshot_frame::{Command, Frame};
use embedded_hal::delay::DelayNs;
use fugit::ExtU64 as _;
use hal::timer;

use crate::{arena::ArenaAlloc, dma::DmaBuf, peripheral::*};

type Instant = fugit::TimerInstant<u64, 1_000_000>;

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

pub struct Driver<T, D>
where
    D: HalDmaRegExt,
{
    pub pwm_timer: timer::Timer<T>,
    pub dma_buf: DmaBuf<DMA_BUF_SIZE, D>,
    max_duty_cycle: u16,
    arm_state: ArmState,
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
        alloc: &mut ArenaAlloc<B>,
    ) -> Self {
        pwm_timer.enable_pwm();
        pwm_timer.enable_dma_interrupt();
        dma.mux(&mut pwm_timer);
        let max_duty_cycle = pwm_timer.max_duty_cycle() as u16;
        Self {
            pwm_timer,
            dma_buf: DmaBuf::new(dma, alloc),
            max_duty_cycle,
            arm_state: ArmState::Disarmed,
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
            self.dma_buf.staging_buf[4 * duty_cycle_index + motor_index] = duty_cycle;
        }
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
        self.pwm_timer.write(&mut self.dma_buf);
    }

    pub fn armed(&self) -> bool {
        matches!(self.arm_state, ArmState::Armed)
    }

    pub fn beep<Delay: DelayNs>(&mut self, delay: &mut Delay) {
        defmt::debug!("Beeping motors");
        let frame = Frame::command(Command::Beep1, false);
        (0..4).for_each(|motor_index| self.write_frame(motor_index, frame));
        self.pwm_timer.write(&mut self.dma_buf);
        delay.delay_ms(260);
        defmt::debug!("Motors beeped");
    }
}
