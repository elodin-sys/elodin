#ifndef PROBE_FIRMWARE_TEST_H
#define PROBE_FIRMWARE_TEST_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Host-side definitions for the freestanding firmware headers.
#define CORE_H
typedef int8_t S8;
typedef uint8_t U8;
typedef int16_t S16;
typedef uint16_t U16;
typedef int32_t S32;
typedef uint32_t U32;
typedef int64_t S64;
typedef uint64_t U64;
typedef intptr_t SPtr;
typedef uintptr_t UPtr;
typedef size_t Size;

#define ArrayCount(array_) ((Size)(sizeof(array_) / sizeof((array_)[0])))
#define Min(a_, b_) ((a_) < (b_) ? (a_) : (b_))
#define Max(a_, b_) ((a_) > (b_) ? (a_) : (b_))

static inline U16 read_le16(U8 const *data) {
    return (U16)(data[0] | ((U16)data[1] << 8));
}

static inline void write_le16(U8 *data, U16 value) {
    data[0] = (U8)value;
    data[1] = (U8)(value >> 8);
}

static inline U32 udiv32(U32 n, U32 d) {
    return d ? n / d : 0xFFFFFFFFu;
}

static inline U32 udiv32_round(U32 n, U32 d) {
    return udiv32(n + (d >> 1), d);
}

static inline U32 read_le32(U8 const *data) {
    return (U32)data[0] |
           ((U32)data[1] << 8) |
           ((U32)data[2] << 16) |
           ((U32)data[3] << 24);
}

static inline void write_le32(U8 *data, U32 value) {
    data[0] = (U8)value;
    data[1] = (U8)(value >> 8);
    data[2] = (U8)(value >> 16);
    data[3] = (U8)(value >> 24);
}

static void test_fail(char const *expression, char const *file, int line) {
    fprintf(stderr, "FAIL %s:%d: %s\n", file, line, expression);
    exit(1);
}

#define CHECK(expression_) \
    do { \
        if (!(expression_)) test_fail(#expression_, __FILE__, __LINE__); \
    } while (0)

#endif
