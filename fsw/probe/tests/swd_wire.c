#include "test.h"

#define ARCH_H
static void arch_barrier(void);

#define BSP_H
typedef struct {
    char const *owner;
    UPtr data_gpio_base;
    UPtr reset_gpio_base;
    U8 swdio_pin;
    U8 swd_dir_pin;
    U8 swclk_pin;
    U8 nreset_pin;
} Bsp_SwdPort;

#define HAL_H

enum {
    DataBase = 1u,
    ResetBase = 2u,
    SwdioPin = 6u,
    DirectionPin = 7u,
    SwclkPin = 8u,
    NresetPin = 0u,
    BitCapacity = 128u,
};

typedef struct {
    U8 target_bits[BitCapacity];
    U8 driven_bits[BitCapacity];
    U32 target_count;
    U32 target_at;
    U32 driven_count;
    U32 order;
    U32 release_order;
    U32 direction_in_order;
    U32 direction_out_order;
    U32 drive_order;
    U32 modes[3][16];
    bool levels[3][16];
} Wire;

static Wire wire;
static bool direct_wire;  // SWDIO without a direction-controlled translator

static void wire_pin_write(UPtr base, U8 pin, bool high) {
    CHECK(base < 3u && pin < 16u);
    wire.order++;
    wire.levels[base][pin] = high;
    if (base == DataBase && pin == DirectionPin) {
        if (high) wire.direction_out_order = wire.order;
        else wire.direction_in_order = wire.order;
    }
    if (base == DataBase && pin == SwclkPin && high &&
        wire.modes[DataBase][SwdioPin] == 1u) {
        CHECK(wire.driven_count < BitCapacity);
        wire.driven_bits[wire.driven_count++] =
            wire.levels[DataBase][SwdioPin] ? 1u : 0u;
    }
}

static bool wire_pin_read(UPtr base, U8 pin) {
    if (base == DataBase && pin == SwdioPin &&
        wire.modes[DataBase][SwdioPin] == 0u) {
        CHECK(wire.target_at < wire.target_count);
        return wire.target_bits[wire.target_at++] != 0u;
    }
    return wire.levels[base][pin];
}

static void wire_pin_mode(UPtr base, U8 pin, U32 mode) {
    wire.order++;
    wire.modes[base][pin] = mode;
    if (base == DataBase && pin == SwdioPin) {
        if (mode == 0u) wire.release_order = wire.order;
        if (mode == 1u) wire.drive_order = wire.order;
    }
}

static void wire_delay(U32 iterations) {
    CHECK(iterations != 0u);
}

static void arch_barrier(void) {
    wire.order++;
}

static void wire_bsrr_write(UPtr base, U32 word) {
    CHECK(word != 0u && (word & (word >> 16)) == 0u);
    for (U8 pin = 0u; pin < 16u; pin++) {
        if (word & (1u << (pin + 16u))) wire_pin_write(base, pin, false);
    }
    for (U8 pin = 0u; pin < 16u; pin++) {
        if (word & (1u << pin)) wire_pin_write(base, pin, true);
    }
}

static U32 wire_idr_read(UPtr base) {
    U32 value = 0u;
    for (U8 pin = 0u; pin < 16u; pin++) {
        if (wire_pin_read(base, pin)) value |= 1u << pin;
    }
    return value;
}

#define SWD__BSRR_WRITE(base_, word_) wire_bsrr_write((base_), (word_))
#define SWD__IDR_READ(base_) wire_idr_read(base_)
#define SWD__PIN_WRITE(base_, pin_, high_) \
    wire_pin_write((base_), (pin_), (high_))
#define SWD__PIN_READ(base_, pin_) wire_pin_read((base_), (pin_))
#define SWD__PIN_MODE(base_, pin_, mode_) \
    wire_pin_mode((base_), (pin_), (mode_))
#define SWD__DELAY(iterations_) wire_delay(iterations_)
#define SWD__RAMFUNC
#include "../swd.h"

static void reset_transaction(void) {
    wire.target_count = 0u;
    wire.target_at = 0u;
    wire.driven_count = 0u;
    wire.release_order = 0u;
    wire.direction_in_order = 0u;
    wire.direction_out_order = 0u;
    wire.drive_order = 0u;
}

static void target_bit(U32 bit) {
    CHECK(wire.target_count < BitCapacity);
    wire.target_bits[wire.target_count++] = (U8)(bit & 1u);
}

static void target_ack(U8 ack) {
    target_bit(ack >> 0);
    target_bit(ack >> 1);
    target_bit(ack >> 2);
}

static void target_read(U32 value, bool valid_parity) {
    U32 parity = 0u;
    for (U32 bit = 0u; bit < 32u; bit++) {
        U32 next = (value >> bit) & 1u;
        target_bit(next);
        parity ^= next;
    }
    target_bit(valid_parity ? parity : parity ^ 1u);
}

static void check_request(U8 request) {
    U8 expected[8] = {1u};
    U8 parity = 0u;
    for (U32 bit = 0u; bit < 4u; bit++) {
        expected[1u + bit] = (U8)((request >> bit) & 1u);
        parity ^= expected[1u + bit];
    }
    expected[5] = parity;
    expected[6] = 0u;
    expected[7] = 1u;
    CHECK(wire.driven_count >= ArrayCount(expected));
    CHECK(memcmp(wire.driven_bits, expected, sizeof(expected)) == 0);
}

static void check_turnaround_order(void) {
    if (direct_wire) {
        CHECK(wire.direction_in_order == 0u && wire.direction_out_order == 0u);
        CHECK(wire.release_order != 0u && wire.drive_order != 0u);
        return;
    }
    CHECK(wire.release_order != 0u);
    CHECK(wire.direction_in_order != 0u);
    CHECK(wire.direction_out_order != 0u);
    CHECK(wire.drive_order != 0u);
    CHECK(wire.release_order < wire.direction_in_order);
    CHECK(wire.direction_out_order < wire.drive_order);
}

static void test_read_transfer(Swd *swd) {
    U8 request = 0x0Fu;
    U32 expected = 0xA53CC35Au;
    reset_transaction();
    target_ack(SwdAck_Ok);
    target_read(expected, true);

    U32 value = 0u;
    CHECK(swd_transfer(swd, request, &value) == SwdAck_Ok);
    CHECK(value == expected);
    CHECK(wire.target_at == wire.target_count);
    check_request(request);
    check_turnaround_order();
}

static void test_write_transfer(Swd *swd) {
    U8 request = 0x04u;
    U32 value = 0xD34DB33Fu;
    reset_transaction();
    target_ack(SwdAck_Ok);

    CHECK(swd_transfer(swd, request, &value) == SwdAck_Ok);
    CHECK(wire.target_at == wire.target_count);
    check_request(request);
    CHECK(wire.driven_count == 41u);

    U32 parity = 0u;
    for (U32 bit = 0u; bit < 32u; bit++) {
        U8 expected = (U8)((value >> bit) & 1u);
        CHECK(wire.driven_bits[8u + bit] == expected);
        parity ^= expected;
    }
    CHECK(wire.driven_bits[40] == parity);
    check_turnaround_order();
}

static void test_error_paths(Swd *swd) {
    reset_transaction();
    target_ack(SwdAck_Ok);
    target_read(0x12345678u, false);
    U32 value = 0u;
    CHECK(swd_transfer(swd, 0x02u, &value) == SwdAck_Error);

    swd->data_phase = true;
    reset_transaction();
    target_ack(SwdAck_Wait);
    CHECK(swd_transfer(swd, 0x02u, &value) == SwdAck_Wait);

    reset_transaction();
    target_ack(SwdAck_Fault);
    CHECK(swd_transfer(swd, 0x00u, &value) == SwdAck_Fault);
}

static void test_clock_selection(Swd *swd) {
    CHECK(!swd_set_clock(swd, 9999u));
    CHECK(swd_set_clock(swd, 10000u) && swd->clock_hz == 10000u);
    CHECK(swd_set_clock(swd, 750000u) && swd->clock_hz == 500000u);
    CHECK(swd_set_clock(swd, 1200000u) && swd->clock_hz == 1000000u);
    CHECK(swd_set_clock(swd, 5000000u) && swd->clock_hz == 2000000u);
}

int main(void) {
    Bsp_SwdPort port = {
        .owner = "test",
        .data_gpio_base = DataBase,
        .reset_gpio_base = ResetBase,
        .swdio_pin = SwdioPin,
        .swd_dir_pin = DirectionPin,
        .swclk_pin = SwclkPin,
        .nreset_pin = NresetPin,
    };
    Swd swd;
    CHECK(swd_init(&swd, &port));
    swd_connect(&swd);

    test_read_transfer(&swd);
    test_write_transfer(&swd);
    test_error_paths(&swd);
    test_clock_selection(&swd);

    swd_disconnect(&swd);
    CHECK(!swd.connected);
    CHECK(wire.modes[DataBase][SwdioPin] == 0u);
    CHECK(!wire.levels[DataBase][DirectionPin]);
    CHECK(!wire.levels[DataBase][SwclkPin]);
    CHECK(wire.levels[ResetBase][NresetPin]);

    // Flight Controller: SWDIO straight to the target; the direction pin is never driven.
    memset(&wire, 0, sizeof(wire));
    direct_wire = true;
    port.swd_dir_pin = SwdNoDirPin;
    CHECK(swd_init(&swd, &port));
    swd_connect(&swd);
    test_read_transfer(&swd);
    test_write_transfer(&swd);
    swd_disconnect(&swd);
    CHECK(wire.modes[DataBase][SwdioPin] == 0u);
    CHECK(wire.levels[ResetBase][NresetPin]);

    puts("PASS swd-wire: framing, parity, ACKs, direction, direct wire, safe idle");
    return 0;
}
