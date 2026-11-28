#ifndef RTT_H
#define RTT_H

#include "core.h"
#include "arch.h"

#ifndef RTT_UP_BUFFER_SIZE
#define RTT_UP_BUFFER_SIZE 1024u
#endif

#ifndef RTT_DOWN_BUFFER_SIZE
#define RTT_DOWN_BUFFER_SIZE 16u
#endif

enum {
    RTT_MODE_NO_BLOCK_SKIP = 0,
};

typedef struct {
    char const *name;
    char *buffer;
    U32 size;
    U32 write_offset;
    volatile U32 read_offset;
    U32 flags;
} Rtt_UpBuffer;

typedef struct {
    char const *name;
    char *buffer;
    U32 size;
    volatile U32 write_offset;
    U32 read_offset;
    U32 flags;
} Rtt_DownBuffer;

typedef struct {
    char id[16];
    S32 up_buffer_count;
    S32 down_buffer_count;
    Rtt_UpBuffer up[1];
    Rtt_DownBuffer down[1];
} Rtt_ControlBlock;

__attribute__((used, aligned(4))) Rtt_ControlBlock _SEGGER_RTT;

static U32 rtt_initialized;
static char rtt_up_buffer[RTT_UP_BUFFER_SIZE] __attribute__((used, aligned(4)));
static char rtt_down_buffer[RTT_DOWN_BUFFER_SIZE] __attribute__((used, aligned(4)));

static void rtt__zero(void *data, U32 size) {
    U8 *at = (U8 *)data;
    for (U32 i = 0; i < size; i++) at[i] = 0;
}

static void rtt__copy(void *dest, void const *source, U32 size) {
    U8 *d = (U8 *)dest;
    U8 const *s = (U8 const *)source;
    for (U32 i = 0; i < size; i++) d[i] = s[i];
}

static U32 rtt__length(char const *string) {
    U32 result = 0;
    while (string[result]) result++;
    return result;
}

static U32 rtt__available(Rtt_UpBuffer *ring) {
    U32 r = ring->read_offset;
    U32 w = ring->write_offset;
    return (r <= w) ? (ring->size - 1u - w + r) : (r - w - 1u);
}

static void rtt__write_unchecked(Rtt_UpBuffer *ring, char const *data, U32 size) {
    U32 w = ring->write_offset;
    U32 first_size = size;
    U32 until_wrap = ring->size - w;

    if (first_size > until_wrap) first_size = until_wrap;

    rtt__copy(ring->buffer + w, data, first_size);
    rtt__copy(ring->buffer, data + first_size, size - first_size);

    w += size;
    if (w >= ring->size) w -= ring->size;

    arch_barrier();
    ring->write_offset = w;
}

void rtt_init(void) {
    static char const id_reversed[] = "\0\0\0\0\0\0TTR REGGES";
    U32 lock = arch_irq_save();

    rtt__zero(&_SEGGER_RTT, sizeof(_SEGGER_RTT));

    _SEGGER_RTT.up_buffer_count = 1;
    _SEGGER_RTT.down_buffer_count = 1;

    _SEGGER_RTT.up[0].name = "Terminal";
    _SEGGER_RTT.up[0].buffer = rtt_up_buffer;
    _SEGGER_RTT.up[0].size = RTT_UP_BUFFER_SIZE;
    _SEGGER_RTT.up[0].flags = RTT_MODE_NO_BLOCK_SKIP;

    _SEGGER_RTT.down[0].name = "Terminal";
    _SEGGER_RTT.down[0].buffer = rtt_down_buffer;
    _SEGGER_RTT.down[0].size = RTT_DOWN_BUFFER_SIZE;
    _SEGGER_RTT.down[0].flags = RTT_MODE_NO_BLOCK_SKIP;

    arch_barrier();
    for (U32 i = 0; i < sizeof(_SEGGER_RTT.id); i++) {
        _SEGGER_RTT.id[i] = id_reversed[sizeof(_SEGGER_RTT.id) - 1u - i];
    }
    arch_barrier();

    rtt_initialized = 1;
    arch_irq_restore(lock);
}

U32 rtt_write(void const *data, U32 size) {
    U32 result = 0;
    if (!rtt_initialized) rtt_init();

    U32 lock = arch_irq_save();
    if (rtt__available(&_SEGGER_RTT.up[0]) >= size) {
        rtt__write_unchecked(&_SEGGER_RTT.up[0], data, size);
        result = size;
    }
    arch_irq_restore(lock);

    return result;
}

U32 rtt_write_string(char const *string) {
    return rtt_write(string, rtt__length(string));
}

U32 rtt_write_char(char c) {
    return rtt_write(&c, 1u);
}

U32 rtt_write_hex_u32(U32 value) {
    static char const digits[] = "0123456789ABCDEF";
    U32 result = rtt_write_string("0x");

    for (S32 shift = 28; shift >= 0; shift -= 4) {
        result += rtt_write_char(digits[(value >> (U32)shift) & 0xFu]);
    }

    return result;
}

U32 rtt_available_write_space(void) {
    if (!rtt_initialized) rtt_init();

    U32 lock = arch_irq_save();
    U32 result = rtt__available(&_SEGGER_RTT.up[0]);
    arch_irq_restore(lock);

    return result;
}

#endif
