#ifndef CORE_H
#define CORE_H

typedef signed char S8;
typedef unsigned char U8;
typedef signed short S16;
typedef unsigned short U16;
typedef signed int S32;
typedef unsigned int U32;
typedef signed long long S64;
typedef unsigned long long U64;
typedef signed long SPtr;
typedef unsigned long UPtr;
typedef unsigned int Size;

static_assert(sizeof(S8) == 1);
static_assert(sizeof(U8) == 1);
static_assert(sizeof(S16) == 2);
static_assert(sizeof(U16) == 2);
static_assert(sizeof(S32) == 4);
static_assert(sizeof(U32) == 4);
static_assert(sizeof(S64) == 8);
static_assert(sizeof(U64) == 8);
static_assert(sizeof(UPtr) == sizeof(void *));
static_assert(sizeof(Size) == sizeof(void *));

#define ArrayCount(array_) ((Size)(sizeof(array_) / sizeof((array_)[0])))
#define Min(a_, b_) ((a_) < (b_) ? (a_) : (b_))
#define Max(a_, b_) ((a_) > (b_) ? (a_) : (b_))

__attribute__((used, externally_visible, noinline))
void *memset(void *dest, int value, Size size) {
    U8 *at = (U8 *)dest;
    for (Size i = 0; i < size; i++) at[i] = (U8)value;
    return dest;
}

__attribute__((used, externally_visible, noinline))
void *memcpy(void *dest, void const *src, Size size) {
    U8 *d = (U8 *)dest;
    U8 const *s = (U8 const *)src;
    for (Size i = 0; i < size; i++) d[i] = s[i];
    return dest;
}

static inline U16 read_le16(U8 const *data) {
    return (U16)(data[0] | ((U16)data[1] << 8));
}

static inline void write_le16(U8 *data, U16 value) {
    data[0] = (U8)value;
    data[1] = (U8)(value >> 8);
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

// Cortex-M0+ has no divider and libgcc is not linked: for setup-time math only.
static inline U32 udiv32(U32 n, U32 d) {
    if (d == 0u) return 0xFFFFFFFFu;
    U32 q = 0u;
    U32 r = 0u;
    for (S32 bit = 31; bit >= 0; bit--) {
        r = (r << 1) | ((n >> bit) & 1u);
        if (r >= d) {
            r -= d;
            q |= 1u << bit;
        }
    }
    return q;
}

// Rounded n / d.
static inline U32 udiv32_round(U32 n, U32 d) {
    return udiv32(n + (d >> 1), d);
}

#endif
