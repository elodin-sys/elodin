#ifndef SWD_H
#define SWD_H

#include "arch.h"
#include "bsp.h"
#include "core.h"
#include "hal.h"

enum {
    SwdAck_Ok = 1u,
    SwdAck_Wait = 2u,
    SwdAck_Fault = 4u,
    SwdAck_NoAck = 7u,
    SwdAck_Error = 8u,

    SwdDefaultClockHz = 1000000u,
    SwdMaximumClockHz = 2000000u,
    SwdMinimumClockHz = 10000u,

    // swd_dir_pin value for SWDIO wired straight to the target (no translator).
    SwdNoDirPin = 0xFFu,
};

typedef struct {
    Bsp_SwdPort port;
    // BSRR words; bit_words[b] sets SWDIO to b and the clock low in one store.
    U32 swclk_high;
    U32 swclk_low;
    U32 swdio_high;
    U32 swdio_low;
    U32 bit_words[2];
    U32 requested_clock_hz;
    U32 clock_hz;
    U32 delay_iterations;
    U32 transfer_count;
    U32 wait_count;
    U32 fault_count;
    U32 protocol_error_count;
    U8 turnaround;
    U8 idle_cycles;
    bool data_phase;
    bool connected;
    bool swdio_output;
    bool nreset_released;
} Swd;

// -Os leaves these out of line otherwise, which alone halves the bit rate.
#define SWD__INLINE static inline __attribute__((always_inline))

// Runs from RAM: taken branches would otherwise refetch from wait-state flash.
#ifndef SWD__RAMFUNC
#define SWD__RAMFUNC __attribute__((section(".ramfunc"), noinline))
#endif

#define SWD__REG32(addr) (*(volatile U32 *)(addr))
#define SWD__GPIO_MODER(base) SWD__REG32((base) + 0x00u)
#define SWD__GPIO_IDR(base)   SWD__REG32((base) + 0x10u)
#define SWD__GPIO_BSRR(base)  SWD__REG32((base) + 0x18u)

#ifndef SWD__PIN_WRITE
#define SWD__PIN_WRITE(base_, pin_, high_) \
    (SWD__GPIO_BSRR(base_) = 1u << ((U32)(pin_) + ((high_) ? 0u : 16u)))
#endif

#ifndef SWD__PIN_READ
#define SWD__PIN_READ(base_, pin_) \
    ((SWD__GPIO_IDR(base_) & (1u << (pin_))) != 0u)
#endif

#ifndef SWD__BSRR_WRITE
#define SWD__BSRR_WRITE(base_, word_) (SWD__GPIO_BSRR(base_) = (word_))
#endif

#ifndef SWD__IDR_READ
#define SWD__IDR_READ(base_) SWD__GPIO_IDR(base_)
#endif

#ifndef SWD__PIN_MODE
#define SWD__PIN_MODE(base_, pin_, mode_) do { \
    U32 swd__shift = (U32)(pin_) * 2u; \
    U32 swd__mask = 3u << swd__shift; \
    SWD__GPIO_MODER(base_) = (SWD__GPIO_MODER(base_) & ~swd__mask) | \
                             ((mode_) << swd__shift); \
} while (0)
#endif

SWD__INLINE void swd__pin_write(UPtr base, U8 pin, bool high) {
    SWD__PIN_WRITE(base, pin, high);
}

SWD__INLINE bool swd__pin_read(UPtr base, U8 pin) {
    return SWD__PIN_READ(base, pin);
}

SWD__INLINE void swd__pin_mode(UPtr base, U8 pin, U32 mode) {
    SWD__PIN_MODE(base, pin, mode);
}

#ifndef SWD__DELAY
SWD__INLINE void swd__delay(U32 iterations) {
    __asm volatile (
        ".syntax unified\n"
        "1:\n"
        "subs %0, #1\n"
        "bne 1b\n"
        : "+l"(iterations)
        :
        : "cc"
    );
}
#else
SWD__INLINE void swd__delay(U32 iterations) {
    SWD__DELAY(iterations);
}
#endif

// Copied out of Swd so the bit loops keep these in registers.
typedef struct {
    UPtr base;
    U32 swclk_high;
    U32 swclk_low;
    U32 swdio_high;
    U32 swdio_low;
    U32 bit_words[2];
    U32 delay;
} SwdWire;

SWD__INLINE SwdWire swd__wire(Swd const *swd) {
    return (SwdWire){
        .base = swd->port.data_gpio_base,
        .swclk_high = swd->swclk_high,
        .swclk_low = swd->swclk_low,
        .swdio_high = swd->swdio_high,
        .swdio_low = swd->swdio_low,
        .bit_words = { swd->bit_words[0], swd->bit_words[1] },
        .delay = swd->delay_iterations,
    };
}

SWD__INLINE void swd__swclk(SwdWire const *w, bool high) {
    SWD__BSRR_WRITE(w->base, high ? w->swclk_high : w->swclk_low);
}

SWD__INLINE void swd__swdio_latch(SwdWire const *w, bool high) {
    SWD__BSRR_WRITE(w->base, high ? w->swdio_high : w->swdio_low);
}

SWD__INLINE void swd__swdio_input(Swd *swd) {
    if (!swd->swdio_output) return;

    // Release the MCU pin before changing the translator from A->B to B->A.
    // This prevents the MCU output from contending with the translator A port.
    swd__pin_mode(swd->port.data_gpio_base, swd->port.swdio_pin, 0u);
    arch_barrier();
    if (swd->port.swd_dir_pin != SwdNoDirPin) {
        swd__pin_write(swd->port.data_gpio_base, swd->port.swd_dir_pin, false);
        arch_barrier();
    }
    swd->swdio_output = false;
}

SWD__INLINE void swd__swdio_output(Swd *swd) {
    if (swd->swdio_output) return;

    // The output latch is preloaded high. Disconnect the target-driven B port
    // before enabling the MCU output, again avoiding A-port contention.
    SwdWire w = swd__wire(swd);
    swd__swdio_latch(&w, true);
    if (swd->port.swd_dir_pin != SwdNoDirPin) {
        swd__pin_write(swd->port.data_gpio_base, swd->port.swd_dir_pin, true);
        arch_barrier();
    }
    swd__pin_mode(swd->port.data_gpio_base, swd->port.swdio_pin, 1u);
    arch_barrier();
    swd->swdio_output = true;
}

// Bit loops, LSB first, each written with and without the delay so the top
// rate has nothing but port stores between edges.
SWD__INLINE void swd__out(SwdWire const *w, U32 value, U32 count) {
    UPtr base = w->base;
    U32 high = w->swclk_high;
    U32 word0 = w->bit_words[0];
    U32 word1 = w->bit_words[1];
    U32 delay = w->delay;
    if (delay == 0u) {
        while (count--) {
            SWD__BSRR_WRITE(base, (value & 1u) ? word1 : word0);
            value >>= 1;
            SWD__BSRR_WRITE(base, high);
        }
        return;
    }
    while (count--) {
        SWD__BSRR_WRITE(base, (value & 1u) ? word1 : word0);
        value >>= 1;
        swd__delay(delay);
        SWD__BSRR_WRITE(base, high);
        swd__delay(delay);
    }
}

SWD__INLINE U32 swd__in(SwdWire const *w, U32 count) {
    UPtr base = w->base;
    U32 low = w->swclk_low;
    U32 high = w->swclk_high;
    U32 mask = w->swdio_high;
    U32 delay = w->delay;
    U32 value = 0u;
    U32 bit = 1u;
    if (delay == 0u) {
        while (count--) {
            SWD__BSRR_WRITE(base, low);
            U32 sample = SWD__IDR_READ(base);
            SWD__BSRR_WRITE(base, high);
            if ((sample & mask) != 0u) value |= bit;
            bit <<= 1;
        }
        return value;
    }
    while (count--) {
        SWD__BSRR_WRITE(base, low);
        swd__delay(delay);
        U32 sample = SWD__IDR_READ(base);
        SWD__BSRR_WRITE(base, high);
        swd__delay(delay);
        if ((sample & mask) != 0u) value |= bit;
        bit <<= 1;
    }
    return value;
}

SWD__INLINE void swd__clocks(SwdWire const *w, U32 count) {
    UPtr base = w->base;
    U32 low = w->swclk_low;
    U32 high = w->swclk_high;
    U32 delay = w->delay;
    if (delay == 0u) {
        while (count--) {
            SWD__BSRR_WRITE(base, low);
            SWD__BSRR_WRITE(base, high);
        }
        return;
    }
    while (count--) {
        SWD__BSRR_WRITE(base, low);
        swd__delay(delay);
        SWD__BSRR_WRITE(base, high);
        swd__delay(delay);
    }
}

SWD__INLINE U32 swd__parity(U32 value) {
    value ^= value >> 16;
    value ^= value >> 8;
    value ^= value >> 4;
    value ^= value >> 2;
    value ^= value >> 1;
    return value & 1u;
}

static bool swd_set_clock(Swd *swd, U32 requested_hz) {
    if (!swd || requested_hz < SwdMinimumClockHz) return false;

    // From a measured sweep at 64 MHz: each rate at or below its label, within
    // 6 % (one iteration is 9 % of a bit at 1 MHz). 2 MHz is no delay at all.
    U32 actual_hz;
    U32 iterations;
    if (requested_hz >= 2000000u) {
        actual_hz = 2000000u;
        iterations = 0u;
    } else if (requested_hz >= 1000000u) {
        actual_hz = 1000000u;
        iterations = 6u;
    } else if (requested_hz >= 500000u) {
        actual_hz = 500000u;
        iterations = 15u;
    } else if (requested_hz >= 250000u) {
        actual_hz = 250000u;
        iterations = 37u;
    } else if (requested_hz >= 100000u) {
        actual_hz = 100000u;
        iterations = 100u;
    } else if (requested_hz >= 50000u) {
        actual_hz = 50000u;
        iterations = 206u;
    } else {
        actual_hz = SwdMinimumClockHz;
        iterations = 1056u;
    }

    swd->requested_clock_hz = requested_hz;
    swd->clock_hz = actual_hz;
    swd->delay_iterations = iterations;
    return true;
}

static bool swd_init(Swd *swd, Bsp_SwdPort const *port) {
    if (!swd || !port || !port->data_gpio_base || !port->reset_gpio_base) return false;

    *swd = (Swd){
        .port = *port,
        .swclk_high = 1u << port->swclk_pin,
        .swclk_low = 1u << (port->swclk_pin + 16u),
        .swdio_high = 1u << port->swdio_pin,
        .swdio_low = 1u << (port->swdio_pin + 16u),
        .bit_words = {
            (1u << (port->swdio_pin + 16u)) | (1u << (port->swclk_pin + 16u)),
            (1u << port->swdio_pin) | (1u << (port->swclk_pin + 16u)),
        },
        .turnaround = 1u,
        .idle_cycles = 0u,
        .data_phase = false,
        .swdio_output = false,
        .nreset_released = true,
    };
    if (!swd_set_clock(swd, SwdDefaultClockHz)) return false;

    SwdWire w = swd__wire(swd);
    swd__swdio_input(swd);
    swd__swclk(&w, false);
    swd__pin_write(swd->port.reset_gpio_base, swd->port.nreset_pin, true);
    return true;
}

static void swd_idle(Swd *swd) {
    swd__swdio_output(swd);
    SwdWire w = swd__wire(swd);
    swd__swdio_latch(&w, true);
}

static void swd_connect(Swd *swd) {
    if (!swd) return;
    swd_idle(swd);
    SwdWire w = swd__wire(swd);
    swd__swclk(&w, true);
    swd__pin_write(swd->port.reset_gpio_base, swd->port.nreset_pin, true);
    swd->nreset_released = true;
    swd->connected = true;
}

static void swd_disconnect(Swd *swd) {
    if (!swd) return;
    SwdWire w = swd__wire(swd);
    swd__swdio_input(swd);
    swd__swclk(&w, false);
    swd__pin_write(swd->port.reset_gpio_base, swd->port.nreset_pin, true);
    swd->nreset_released = true;
    swd->connected = false;
}

static void swd_nreset_write(Swd *swd, bool released) {
    if (!swd) return;
    swd__pin_write(swd->port.reset_gpio_base, swd->port.nreset_pin, released);
    swd->nreset_released = released;
}

static U8 swd_pins_read(Swd const *swd) {
    if (!swd) return 0u;

    SwdWire w = swd__wire(swd);
    U8 value = 0u;
    if (swd__pin_read(swd->port.data_gpio_base, swd->port.swclk_pin)) value |= 1u << 0;
    if ((SWD__IDR_READ(w.base) & w.swdio_high) != 0u) value |= 1u << 1;
    if (swd->nreset_released) value |= 1u << 7;
    return value;
}

static void swd_pins_write(Swd *swd, U8 value, U8 select) {
    if (!swd) return;

    SwdWire w = swd__wire(swd);
    if ((select & (1u << 0)) != 0u) {
        swd__swclk(&w, (value & (1u << 0)) != 0u);
    }
    if ((select & (1u << 1)) != 0u) {
        swd__swdio_output(swd);
        swd__swdio_latch(&w, (value & (1u << 1)) != 0u);
    }
    if ((select & (1u << 7)) != 0u) {
        swd_nreset_write(swd, (value & (1u << 7)) != 0u);
    }
}

static SWD__RAMFUNC void swd_sequence_out(Swd *swd, U32 count, U8 const *data) {
    if (!swd || !data) return;
    swd__swdio_output(swd);

    SwdWire w = swd__wire(swd);
    for (U32 at = 0u; at < count; at += 8u) {
        swd__out(&w, data[at >> 3], Min(8u, count - at));
    }
}

static SWD__RAMFUNC void swd_sequence_in(Swd *swd, U32 count, U8 *data) {
    if (!swd || !data) return;
    swd__swdio_input(swd);

    SwdWire w = swd__wire(swd);
    for (U32 at = 0u; at < count; at += 8u) {
        data[at >> 3] = (U8)swd__in(&w, Min(8u, count - at));
    }
}

static SWD__RAMFUNC U8 swd_transfer(Swd *swd, U8 request, U32 *data) {
    if (!swd || !swd->connected) return 0u;

    swd->transfer_count++;
    SwdWire w = swd__wire(swd);
    U32 turnaround = swd->turnaround;
    bool read = (request & (1u << 1)) != 0u;

    U32 header = 1u | ((request & 0xFu) << 1) | (swd__parity(request & 0xFu) << 5) | (1u << 7);
    swd__swdio_output(swd);
    swd__out(&w, header, 8u);

    swd__swdio_input(swd);
    swd__clocks(&w, turnaround);
    U8 ack = (U8)swd__in(&w, 3u);

    if (ack == SwdAck_Ok) {
        if (read) {
            U32 value = swd__in(&w, 32u);
            U32 parity = swd__in(&w, 1u);
            swd__clocks(&w, turnaround);
            swd__swdio_output(swd);

            if (swd__parity(value) != parity) {
                ack = SwdAck_Error;
                swd->protocol_error_count++;
            }
            if (data) *data = value;
        } else {
            swd__clocks(&w, turnaround);
            swd__swdio_output(swd);

            U32 value = data ? *data : 0u;
            swd__out(&w, value, 32u);
            swd__out(&w, swd__parity(value), 1u);
        }

        if (swd->idle_cycles != 0u) {
            swd__swdio_latch(&w, false);
            swd__clocks(&w, swd->idle_cycles);
        }
        swd__swdio_latch(&w, true);
        return ack;
    }

    if (ack == SwdAck_Wait || ack == SwdAck_Fault) {
        if (ack == SwdAck_Wait) swd->wait_count++;
        else swd->fault_count++;

        if (swd->data_phase && read) swd__clocks(&w, 33u);
        swd__clocks(&w, turnaround);
        swd__swdio_output(swd);
        if (swd->data_phase && !read) {
            swd__swdio_latch(&w, false);
            swd__clocks(&w, 33u);
        }
        swd__swdio_latch(&w, true);
        return ack;
    }

    swd__clocks(&w, turnaround + 33u);
    swd__swdio_output(swd);
    swd__swdio_latch(&w, true);
    swd->protocol_error_count++;
    return ack;
}

#endif
