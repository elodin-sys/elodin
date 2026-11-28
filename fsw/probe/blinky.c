// blinky.c — first-light firmware for the Aleph Flight Controller debug probe.
//
// What it does:
//   - Runs the STM32G0B1 at 64 MHz with a 1 kHz SysTick timebase.
//   - Releases the UCPD2 dead-battery pull-downs on the UART6 console pins.
//   - Establishes safe target-interface states before blinking any LEDs.
//   - Initialises SEGGER RTT and logs a compact heartbeat once per second.
//   - Blinks the indicator LED at 1 Hz and steps the RGB LED through red,
//     green and blue once per second.
//
// What it does not do:
//   - Enumerate USB.
//   - Clock or communicate with the STM32H747.
//   - Assert target nRESET or BOOT0.

#include "core.h"

#include "arch.h"
#include "bsp.h"
#include "hal.h"
#include "rgb_led.h"
#include "rtt.h"

extern U32 _estack;
extern U32 _sidata;
extern U32 _sdata;
extern U32 _edata;
extern U32 _sbss;
extern U32 _ebss;

enum {
    IndicatorOnMs = 100,
    IndicatorOffMs = 900,
    RgbLevel = 24,
};

static Bsp_Signal indicator;
static RgbLed rgb;
static U32 faulting;

static void log_hex_field(char const *name, U32 value) {
    rtt_write_string(name);
    rtt_write_hex_u32(value);
}

static void fault_halt(char const *reason) {
    arch_irq_save();
    bsp_fault_safe_outputs();

    if (!faulting) {
        faulting = 1;
        rtt_write_string("\nblinky fault: ");
        rtt_write_string(reason);
        log_hex_field("\n  exception: ", arch_exception_number());
        log_hex_field("\n  msp:       ", arch_main_stack_pointer());
        log_hex_field("\n  psp:       ", arch_process_stack_pointer());
        rtt_write_string("\n");
    }

    if (arch_debugger_attached()) {
        arch_breakpoint();
    }

    while (1) {
    }
}

void Fault_Handler(void) {
    fault_halt("fault-handler");
}

void Default_Handler(void) {
    fault_halt("default-handler");
}

static void board_check(bool ok) {
    if (!ok) Fault_Handler();
}

int main(void) {
    hal_core_init(&(Hal_CoreConfig){ .clock = HalClock_Hsi16Pll_64MHz });
    board_check(bsp_init());

    rtt_init();
    rtt_write_string("Aleph FC probe blinky boot\n");
    log_hex_field("core_hz=", hal_core()->sys_hz);
    rtt_write_string(" target_swd=safe target_nreset=released target_boot0=released\n");

    board_check(bsp_signal_prepare(&indicator, BspSignal_LedIndicator, "blinky-indicator"));
    Bsp_RgbLed rgb_bsp = {0};
    board_check(bsp_rgb_led_prepare(&rgb_bsp, "blinky-rgb"));
    board_check(rgb_led_init(&rgb, &rgb_bsp));

    U32 cycles = 0;
    U32 phase = 0;
    while (1) {
        hal_delay_us(RgbLedResetUs);
        board_check(rgb_led_set(&rgb, phase == 0u ? RgbLevel : 0u,
                                phase == 1u ? RgbLevel : 0u,
                                phase == 2u ? RgbLevel : 0u));
        board_check(bsp_signal_write(&indicator, true));
        hal_delay_us(IndicatorOnMs * 1000u);
        board_check(bsp_signal_write(&indicator, false));
        hal_delay_us(IndicatorOffMs * 1000u);

        rtt_write_string("blink ");
        rtt_write_hex_u32(cycles++);
        phase = phase == 2u ? 0u : phase + 1u;
        log_hex_field(" us=", hal_now_us());
        rtt_write_string("\n");
    }
}

extern const UPtr vectors[];

void Reset_Handler(void) {
    arch_startup_state((U32)vectors);

    U32 *src = &_sidata;
    U32 *dst = &_sdata;

    while (dst < &_edata) {
        *dst++ = *src++;
    }

    dst = &_sbss;
    while (dst < &_ebss) {
        *dst++ = 0;
    }

    main();

    while (1) {
    }
}

enum {
    VectorInitialStack = 0,
    VectorReset = 1,
    VectorNmi = 2,
    VectorHardFault = 3,
    VectorSvCall = 11,
    VectorPendSv = 14,
    VectorSysTick = 15,
    VectorLast = 16 + 30,
    VectorCount = VectorLast + 1,
};

__attribute__((section(".isr_vector"), used))
const UPtr vectors[VectorCount] = {
    [VectorInitialStack] = (UPtr)&_estack,
    [VectorReset] = (UPtr)Reset_Handler,
    [VectorNmi] = (UPtr)Fault_Handler,
    [VectorHardFault] = (UPtr)Fault_Handler,
    [VectorSvCall] = (UPtr)Default_Handler,
    [VectorPendSv] = (UPtr)Default_Handler,
    [VectorSysTick] = (UPtr)SysTick_Handler,

    [16 + 0] = (UPtr)Default_Handler,  // WWDG
    [16 + 1] = (UPtr)Default_Handler,  // PVD/VDDIO2
    [16 + 2] = (UPtr)Default_Handler,  // RTC/TAMP
    [16 + 3] = (UPtr)Default_Handler,  // FLASH
    [16 + 4] = (UPtr)Default_Handler,  // RCC/CRS
    [16 + 5] = (UPtr)Default_Handler,  // EXTI0/1
    [16 + 6] = (UPtr)Default_Handler,  // EXTI2/3
    [16 + 7] = (UPtr)Default_Handler,  // EXTI4/15
    [16 + HalIrq_UsbUcpd1_2] = (UPtr)Default_Handler,
    [16 + HalIrq_Dma1Channel1] = (UPtr)Default_Handler,
    [16 + HalIrq_Dma1Channel2_3] = (UPtr)Default_Handler,
    [16 + 11] = (UPtr)Default_Handler, // DMA1 Ch4-7/DMA2 Ch1-5/DMAMUX
    [16 + 12] = (UPtr)Default_Handler, // ADC1/COMP
    [16 + 13] = (UPtr)Default_Handler, // TIM1 BRK/UP/TRG/COM
    [16 + 14] = (UPtr)Default_Handler, // TIM1 CC
    [16 + 15] = (UPtr)Default_Handler, // TIM2
    [16 + 16] = (UPtr)Default_Handler, // TIM3/TIM4
    [16 + 17] = (UPtr)Default_Handler, // TIM6/DAC/LPTIM1
    [16 + 18] = (UPtr)Default_Handler, // TIM7/LPTIM2
    [16 + 19] = (UPtr)Default_Handler, // TIM14
    [16 + 20] = (UPtr)Default_Handler, // TIM15
    [16 + 21] = (UPtr)Default_Handler, // TIM16/FDCAN IT0
    [16 + 22] = (UPtr)Default_Handler, // TIM17/FDCAN IT1
    [16 + HalIrq_I2c1] = (UPtr)Default_Handler,
    [16 + 24] = (UPtr)Default_Handler, // I2C2/I2C3
    [16 + HalIrq_Spi1] = (UPtr)Default_Handler,
    [16 + 26] = (UPtr)Default_Handler, // SPI2/SPI3
    [16 + HalIrq_Usart1] = (UPtr)Default_Handler,
    [16 + HalIrq_Usart2Lpuart2] = (UPtr)Default_Handler,
    [16 + HalIrq_Usart3_4_5_6Lpuart1] = (UPtr)Default_Handler,
    [16 + 30] = (UPtr)Default_Handler, // CEC
};
