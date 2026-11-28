#ifndef RGB_LED_H
#define RGB_LED_H

// One WL-ICLED (WS2812-class, GRB) on TIM2_CH1. Each bit is one PWM period;
// the next bit's duty is loaded into the preloaded CCR1 while the current one
// plays, so edge timing comes from the timer, not from instruction counts.
// Datasheet: bit 1.2 us, T0H 0.15-0.45 us, T1H 0.75-1.05 us, reset >= 200 us.

#include "arch.h"
#include "bsp.h"
#include "core.h"
#include "hal.h"

enum {
    RgbLedBitNs = 1200u,
    RgbLedT0hNs = 300u,
    RgbLedT1hNs = 900u,
    RgbLedResetUs = 250u,
};

typedef struct {
    UPtr tim;
    U32 period;
    U32 t0h;
    U32 t1h;
    U32 last_us;
    U32 grb;
    bool valid;
} RgbLed;

#define RGB_LED__REG(base_, off_) (*(volatile U32 *)((base_) + (off_)))
#define RGB_LED__CR1   0x00u
#define RGB_LED__SR    0x10u
#define RGB_LED__EGR   0x14u
#define RGB_LED__CCMR1 0x18u
#define RGB_LED__CCER  0x20u
#define RGB_LED__CNT   0x24u
#define RGB_LED__PSC   0x28u
#define RGB_LED__ARR   0x2Cu
#define RGB_LED__CCR1  0x34u

#define RGB_LED__CR1_CEN   (1u << 0)
#define RGB_LED__CR1_ARPE  (1u << 7)
#define RGB_LED__SR_UIF    (1u << 0)
#define RGB_LED__EGR_UG    (1u << 0)
#define RGB_LED__OC1PE     (1u << 3)
#define RGB_LED__OC1M_PWM1 (6u << 4)
#define RGB_LED__CC1E      (1u << 0)

static U32 rgb_led__ticks(U32 hz, U32 ns) {
    return udiv32_round(udiv32(hz, 1000000u) * ns, 1000u);
}

static bool rgb_led_init(RgbLed *led, Bsp_RgbLed const *hw) {
    if (!led || !hw || !hw->tim_base || !hw->tim_hz) return false;
    *led = (RgbLed){
        .tim = hw->tim_base,
        .period = rgb_led__ticks(hw->tim_hz, RgbLedBitNs),
        .t0h = rgb_led__ticks(hw->tim_hz, RgbLedT0hNs),
        .t1h = rgb_led__ticks(hw->tim_hz, RgbLedT1hNs),
        .last_us = hal_now_us(),
    };

    UPtr t = led->tim;
    RGB_LED__REG(t, RGB_LED__CR1) = RGB_LED__CR1_ARPE;
    RGB_LED__REG(t, RGB_LED__PSC) = 0u;
    RGB_LED__REG(t, RGB_LED__ARR) = led->period - 1u;
    RGB_LED__REG(t, RGB_LED__CCR1) = 0u;
    RGB_LED__REG(t, RGB_LED__CCMR1) = RGB_LED__OC1M_PWM1 | RGB_LED__OC1PE;
    RGB_LED__REG(t, RGB_LED__CCER) = RGB_LED__CC1E;
    RGB_LED__REG(t, RGB_LED__EGR) = RGB_LED__EGR_UG;
    RGB_LED__REG(t, RGB_LED__SR) = 0u;
    return true;
}

static void rgb_led__wait_update(UPtr t) {
    while ((RGB_LED__REG(t, RGB_LED__SR) & RGB_LED__SR_UIF) == 0u) {
    }
    RGB_LED__REG(t, RGB_LED__SR) = 0u;
}

// Sends one frame if the colour changed and the previous frame's latch time
// has passed. Returns false only when it skipped a change, to be retried.
static bool rgb_led_set(RgbLed *led, U8 r, U8 g, U8 b) {
    U32 grb = ((U32)g << 16) | ((U32)r << 8) | b;
    if (led->valid && grb == led->grb) return true;
    if (!hal_time_expired_us(led->last_us, RgbLedResetUs)) return false;

    UPtr t = led->tim;
    U32 lock = arch_irq_save();
    RGB_LED__REG(t, RGB_LED__CNT) = 0u;
    RGB_LED__REG(t, RGB_LED__CCR1) = (grb & 0x800000u) ? led->t1h : led->t0h;
    RGB_LED__REG(t, RGB_LED__EGR) = RGB_LED__EGR_UG;
    RGB_LED__REG(t, RGB_LED__SR) = 0u;
    RGB_LED__REG(t, RGB_LED__CR1) = RGB_LED__CR1_ARPE | RGB_LED__CR1_CEN;
    for (S32 bit = 22; bit >= 0; bit--) {
        RGB_LED__REG(t, RGB_LED__CCR1) = (grb & (1u << bit)) ? led->t1h : led->t0h;
        rgb_led__wait_update(t);
    }
    RGB_LED__REG(t, RGB_LED__CCR1) = 0u;
    rgb_led__wait_update(t);  // last data bit plays out
    rgb_led__wait_update(t);  // CCR1 = 0 takes effect: line low
    RGB_LED__REG(t, RGB_LED__CR1) = RGB_LED__CR1_ARPE;
    arch_irq_restore(lock);

    led->grb = grb;
    led->valid = true;
    led->last_us = hal_now_us();
    return true;
}

#endif
