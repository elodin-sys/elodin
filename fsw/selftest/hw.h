// hw.h — minimal STM32H747 (Cortex-M7 core) helpers for the self-test:
// GPIO, a 1 ms SysTick timebase on the 64 MHz HSI, a polled console UART,
// a tiny formatter, and bit-banged I2C and SPI.
#ifndef HW_H
#define HW_H

#define STM32H747xx
#define CORE_CM7
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include "stm32h747xx.h"

typedef struct {
    GPIO_TypeDef *port;
    uint8_t pin;
} Pin;

typedef struct {
    const char *name;
    Pin scl;
    Pin sda;
} I2cBus;

typedef struct {
    Pin sck;
    Pin mosi;
    Pin miso;
    Pin cs;
} SpiBus;

enum { ModeInput = 0, ModeOutput = 1, ModeAf = 2, ModeAnalog = 3 };
enum { PullNone = 0, PullUp = 1, PullDown = 2 };

// ---- GPIO ----

static inline void pin_config(Pin p, uint32_t mode, bool open_drain, uint32_t pull, uint32_t af) {
    uint32_t n = p.pin;
    p.port->OTYPER = (p.port->OTYPER & ~(1u << n)) | ((open_drain ? 1u : 0u) << n);
    p.port->OSPEEDR &= ~(3u << (2u * n));  // low speed
    p.port->PUPDR = (p.port->PUPDR & ~(3u << (2u * n))) | (pull << (2u * n));
    volatile uint32_t *afr = &p.port->AFR[n >> 3];
    *afr = (*afr & ~(0xFu << (4u * (n & 7u)))) | (af << (4u * (n & 7u)));
    p.port->MODER = (p.port->MODER & ~(3u << (2u * n))) | (mode << (2u * n));
}

static inline void pin_write(Pin p, bool high) {
    p.port->BSRR = high ? (1u << p.pin) : (1u << (p.pin + 16u));
}

static inline bool pin_read(Pin p) {
    return (p.port->IDR & (1u << p.pin)) != 0u;
}

// Preloads the level, then switches to output, so the pin never glitches.
static inline void pin_output(Pin p, bool high, bool open_drain) {
    pin_write(p, high);
    pin_config(p, ModeOutput, open_drain, PullNone, 0);
}

static inline void pin_input(Pin p, uint32_t pull) {
    pin_config(p, ModeInput, false, pull, 0);
}

// ---- Time: SysTick at 1 kHz from the 64 MHz core clock ----

#define CORE_HZ 64000000u
extern volatile uint32_t tick_ms;  // incremented by SysTick_Handler in selftest.c
void SysTick_Handler(void);

static inline void time_init(void) {
    SysTick->LOAD = CORE_HZ / 1000u - 1u;
    SysTick->VAL = 0;
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk | SysTick_CTRL_TICKINT_Msk | SysTick_CTRL_ENABLE_Msk;
}

static inline uint32_t now_ms(void) {
    return tick_ms;
}

static inline uint32_t now_us(void) {
    uint32_t ms, val;
    do {
        ms = tick_ms;
        val = SysTick->VAL;
    } while (ms != tick_ms);
    return ms * 1000u + (SysTick->LOAD - val) / (CORE_HZ / 1000000u);
}

static inline void delay_us(uint32_t us) {
    uint32_t start = now_us();
    while ((uint32_t)(now_us() - start) < us) {
    }
}

static inline void delay_ms(uint32_t ms) {
    delay_us(ms * 1000u);
}

// ---- Polled UART (USART3/UART4 kernel clock = PCLK1 = 64 MHz after reset) ----

static inline void uart_init(USART_TypeDef *u, Pin tx, Pin rx, uint32_t af, uint32_t baud) {
    pin_write(tx, true);
    pin_config(tx, ModeAf, false, PullNone, af);
    pin_config(rx, ModeAf, false, PullUp, af);
    u->CR1 = 0;
    u->BRR = (CORE_HZ + baud / 2u) / baud;
    u->CR1 = USART_CR1_TE | USART_CR1_RE | USART_CR1_UE;
}

static inline void uart_putc(USART_TypeDef *u, char c) {
    while ((u->ISR & USART_ISR_TXE_TXFNF) == 0u) {
    }
    u->TDR = (uint8_t)c;
}

static inline int uart_getc(USART_TypeDef *u) {
    if ((u->ISR & (USART_ISR_ORE | USART_ISR_FE | USART_ISR_NE)) != 0u) {
        u->ICR = USART_ICR_ORECF | USART_ICR_FECF | USART_ICR_NECF;
    }
    if ((u->ISR & USART_ISR_RXNE_RXFNE) == 0u) return -1;
    return (int)(u->RDR & 0xFFu);
}

static inline void uart_drain(USART_TypeDef *u) {
    while (uart_getc(u) >= 0) {
    }
}

// ---- Console formatting: %s %c %d %u %x %02x %08x %% ----

#define CONSOLE USART3

static inline void con_putc(char c) {
    if (c == '\n') uart_putc(CONSOLE, '\r');
    uart_putc(CONSOLE, c);
}

static inline void con_puts(const char *s) {
    while (*s) con_putc(*s++);
}

static inline void con_number(uint32_t value, uint32_t base, int width, char pad, bool negative) {
    char buf[12];
    int n = 0;
    do {
        uint32_t digit = value % base;
        buf[n++] = (char)(digit < 10u ? '0' + digit : 'a' + digit - 10u);
        value /= base;
    } while (value != 0u);
    if (negative) buf[n++] = '-';
    while (n < width) buf[n++] = pad;
    while (n > 0) con_putc(buf[--n]);
}

static inline void con_vprintf(const char *fmt, va_list ap) {
    for (; *fmt; fmt++) {
        if (*fmt != '%') {
            con_putc(*fmt);
            continue;
        }
        fmt++;
        char pad = ' ';
        int width = 0;
        if (*fmt == '0') {
            pad = '0';
            fmt++;
        }
        while (*fmt >= '0' && *fmt <= '9') width = width * 10 + (*fmt++ - '0');
        switch (*fmt) {
        case 's': con_puts(va_arg(ap, const char *)); break;
        case 'c': con_putc((char)va_arg(ap, int)); break;
        case 'u': con_number(va_arg(ap, uint32_t), 10u, width, pad, false); break;
        case 'x': con_number(va_arg(ap, uint32_t), 16u, width, pad, false); break;
        case 'd': {
            int32_t v = va_arg(ap, int32_t);
            con_number(v < 0 ? (uint32_t)-v : (uint32_t)v, 10u, width, pad, v < 0);
            break;
        }
        case '%': con_putc('%'); break;
        default: con_putc('?'); break;
        }
    }
}

static inline void con_printf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    con_vprintf(fmt, ap);
    va_end(ap);
}

// ---- Bit-banged I2C (open drain, board pull-ups, about 100 kHz) ----

#define I2C_HALF_US 5u

static inline void i2c_release(Pin p) {
    pin_write(p, true);
}

static inline void i2c_pull(Pin p) {
    pin_write(p, false);
}

// Both lines high-idle; returns false if either is held low (no pull-up,
// unpowered rail, or a stuck device) even after nine recovery clocks.
static inline bool i2c_init(const I2cBus *b) {
    pin_output(b->scl, true, true);
    pin_output(b->sda, true, true);
    delay_us(I2C_HALF_US);
    for (int i = 0; i < 9 && !pin_read(b->sda); i++) {
        i2c_pull(b->scl);
        delay_us(I2C_HALF_US);
        i2c_release(b->scl);
        delay_us(I2C_HALF_US);
    }
    return pin_read(b->scl) && pin_read(b->sda);
}

static inline void i2c_start(const I2cBus *b) {
    i2c_release(b->sda);
    i2c_release(b->scl);
    delay_us(I2C_HALF_US);
    i2c_pull(b->sda);
    delay_us(I2C_HALF_US);
    i2c_pull(b->scl);
}

static inline void i2c_stop(const I2cBus *b) {
    i2c_pull(b->sda);
    delay_us(I2C_HALF_US);
    i2c_release(b->scl);
    delay_us(I2C_HALF_US);
    i2c_release(b->sda);
    delay_us(I2C_HALF_US);
}

static inline bool i2c_clock_high(const I2cBus *b) {
    i2c_release(b->scl);
    uint32_t start = now_us();
    while (!pin_read(b->scl)) {  // clock stretching
        if ((uint32_t)(now_us() - start) > 1000u) return false;
    }
    delay_us(I2C_HALF_US);
    return true;
}

// Returns true on ACK.
static inline bool i2c_write_byte(const I2cBus *b, uint8_t byte) {
    for (int i = 7; i >= 0; i--) {
        pin_write(b->sda, (byte >> i) & 1u);
        delay_us(I2C_HALF_US);
        if (!i2c_clock_high(b)) return false;
        i2c_pull(b->scl);
    }
    i2c_release(b->sda);
    delay_us(I2C_HALF_US);
    if (!i2c_clock_high(b)) return false;
    bool ack = !pin_read(b->sda);
    i2c_pull(b->scl);
    return ack;
}

static inline uint8_t i2c_read_byte(const I2cBus *b, bool ack) {
    uint8_t byte = 0;
    i2c_release(b->sda);
    for (int i = 7; i >= 0; i--) {
        delay_us(I2C_HALF_US);
        i2c_clock_high(b);
        byte = (uint8_t)((byte << 1) | (pin_read(b->sda) ? 1u : 0u));
        i2c_pull(b->scl);
    }
    pin_write(b->sda, !ack);
    delay_us(I2C_HALF_US);
    i2c_clock_high(b);
    i2c_pull(b->scl);
    i2c_release(b->sda);
    return byte;
}

static inline bool i2c_probe(const I2cBus *b, uint8_t addr) {
    i2c_start(b);
    bool ack = i2c_write_byte(b, (uint8_t)(addr << 1));
    i2c_stop(b);
    return ack;
}

// Register read with a repeated start. `skip` leading bytes (dummy bytes) are
// clocked in and dropped before `n` data bytes are stored.
static inline bool i2c_read_reg(const I2cBus *b, uint8_t addr, uint8_t reg, uint8_t *out, int n, int skip) {
    i2c_start(b);
    bool ok = i2c_write_byte(b, (uint8_t)(addr << 1)) && i2c_write_byte(b, reg);
    if (ok) {
        i2c_start(b);
        ok = i2c_write_byte(b, (uint8_t)((addr << 1) | 1u));
    }
    if (ok) {
        int total = n + skip;
        for (int i = 0; i < total; i++) {
            uint8_t v = i2c_read_byte(b, i + 1 < total);
            if (i >= skip) out[i - skip] = v;
        }
    }
    i2c_stop(b);
    return ok;
}

// ---- Bit-banged SPI, mode 0, MSB first, about 250 kHz ----

#define SPI_HALF_US 2u

static inline void spi_init(const SpiBus *s) {
    pin_output(s->cs, true, false);
    pin_output(s->sck, false, false);
    pin_output(s->mosi, false, false);
    pin_input(s->miso, PullNone);
}

static inline void spi_select(const SpiBus *s, bool on) {
    delay_us(SPI_HALF_US);
    pin_write(s->cs, !on);
    delay_us(SPI_HALF_US);
}

static inline uint8_t spi_byte(const SpiBus *s, uint8_t out) {
    uint8_t in = 0;
    for (int i = 7; i >= 0; i--) {
        pin_write(s->mosi, (out >> i) & 1u);
        delay_us(SPI_HALF_US);
        pin_write(s->sck, true);
        in = (uint8_t)((in << 1) | (pin_read(s->miso) ? 1u : 0u));
        delay_us(SPI_HALF_US);
        pin_write(s->sck, false);
    }
    return in;
}

#endif
