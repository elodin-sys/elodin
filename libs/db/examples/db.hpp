#ifndef ELODIN_DB_H
#define ELODIN_DB_H

#include <stdio.h>
#include <span>
#include <variant>
#include <type_traits>
#include <vector>
#include <optional>
#include <unordered_map>


#if __has_include("postcard.h")
#include "postcard.h"
#endif


#ifndef POSTCARD_H
#define POSTCARD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/// A growable view into a buffer
typedef struct {
    // Pointer to the underlying data buffer
    uint8_t* data;
    /// The total written length of data in the buffer
    size_t len; // Current length of data
    /// The maximum capacity of the buffer
    size_t capacity;
} postcard_slice_t;

/// postcard_c error codes
typedef enum {
    POSTCARD_SUCCESS = 0,
    POSTCARD_ERROR_BUFFER_TOO_SMALL,
    POSTCARD_ERROR_INVALID_INPUT,
    POSTCARD_ERROR_INCOMPLETE_DATA,
    POSTCARD_ERROR_OVERFLOW
} postcard_error_t;

/// Initializes a postcard_slice_t a growable view into a buffer
///
/// *Arguments*
///
/// - slice - a pointer to an uninitialized `postcard_slice_t`
/// - buffer - a pointer to the underlying buffer
/// - capacity - the total size of the underlying buffer
///
/// Safety / Lifetimes:
///
/// The user must ensure that postcard_slice_t does not outlive buffer, and that capacity is less than or
/// equal to the capacity of the underlying buffer
void postcard_init_slice(postcard_slice_t* slice, uint8_t* buffer, size_t capacity);

/// Encodes a bool to the passed in slice.
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - value - the value to encode
/// *Side Effects / Return*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If encoding was successful, slice.len will be incremented by the number of encoded bytes
postcard_error_t postcard_encode_bool(postcard_slice_t* slice, bool value);
/// Encodes a u8 to the passed in slice.
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - value - the value to encode
/// *Side Effects / Return*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If encoding was successful, slice.len will be incremented by the number of encoded bytes
postcard_error_t postcard_encode_u8(postcard_slice_t* slice, uint8_t value);
/// Encodes a i8 to the passed in slice.
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - value - the value to encode
/// *Side Effects / Return*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If encoding was successful, slice.len will be incremented by the number of encoded bytes
postcard_error_t postcard_encode_i8(postcard_slice_t* slice, int8_t value);
/// Encodes a u16 to the passed in slice.
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - value - the value to encode
/// *Side Effects / Return*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If encoding was successful, slice.len will be incremented by the number of encoded bytes
postcard_error_t postcard_encode_u16(postcard_slice_t* slice, uint16_t value);
/// Encodes a i16 to the passed in slice.
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - value - the value to encode
/// *Side Effects / Return*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If encoding was successful, slice.len will be incremented by the number of encoded bytes
postcard_error_t postcard_encode_i16(postcard_slice_t* slice, int16_t value);
/// Encodes a u32 to the passed in slice.
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - value - the value to encode
/// *Side Effects / Return*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If encoding was successful, slice.len will be incremented by the number of encoded bytes
postcard_error_t postcard_encode_u32(postcard_slice_t* slice, uint32_t value);
/// Encodes a i32 to the passed in slice.
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - value - the value to encode
/// *Side Effects / Return*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If encoding was successful, slice.len will be incremented by the number of encoded bytes
postcard_error_t postcard_encode_i32(postcard_slice_t* slice, int32_t value);
/// Encodes a u64 to the passed in slice.
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - value - the value to encode
/// *Side Effects / Return*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If encoding was successful, slice.len will be incremented by the number of encoded bytes
postcard_error_t postcard_encode_u64(postcard_slice_t* slice, uint64_t value);
/// Encodes a i64 to the passed in slice.
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - value - the value to encode
/// *Side Effects / Return*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If encoding was successful, slice.len will be incremented by the number of encoded bytes
postcard_error_t postcard_encode_i64(postcard_slice_t* slice, int64_t value);
/// Encodes a f32 to the passed in slice.
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - value - the value to encode
/// *Side Effects / Return*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If encoding was successful, slice.len will be incremented by the number of encoded bytes
postcard_error_t postcard_encode_f32(postcard_slice_t* slice, float value);

/// Encodes a f64 to the passed in slice.
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - value - the value to encode
/// *Side Effects / Return*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If encoding was successful, slice.len will be incremented by the number of encoded bytes
postcard_error_t postcard_encode_f64(postcard_slice_t* slice, double value);

/// Encodes a byte array to the passed in slice.
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - bytes - a pointer to the byte array to encode
/// - length - the length of the byte array
/// *Safety*
/// The user must ensure that bytes points to a valid initialized memory, and that length is not greater than the size of the array.
/// *Side Effects / Returns*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If encoding was successful, slice.len will be incremented by the number of encoded bytes
postcard_error_t postcard_encode_byte_array(postcard_slice_t* slice, const uint8_t* bytes, size_t length);

/// Encodes a string to the passed in slice.
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - string - a pointer to the string to encode. This string should be utf8 encoded string. It shouldn't be null-terminated.
/// - length - the length of the string
/// *Safety*
/// The user must ensure that string points to a valid initialized memory, and that length is not greater than the size of the string.
/// *Side Effects / Returns*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If encoding was successful, slice.len will be incremented by the number of encoded bytes
postcard_error_t postcard_encode_string(postcard_slice_t* slice, const char* string, size_t length);

/// Encodes the none tag for an optional
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// *Side Effects / Returns*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If encoding was successful, slice.len will be incremented by the number of encoded bytes
postcard_error_t postcard_encode_option_none(postcard_slice_t* slice);
/// Encodes the some tag for an optional
///
/// The user needs to encode actual value of the optional after this call
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// *Side Effects / Returns*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If encoding was successful, slice.len will be incremented by the number of encoded bytes
postcard_error_t postcard_encode_option_some(postcard_slice_t* slice);

/// Encodes a variant tag for an enum
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - discriminant - the discriminant of the variant
/// *Side Effects / Returns*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If encoding was successful, slice.len will be incremented by the number of encoded bytes
postcard_error_t postcard_encode_variant(postcard_slice_t* slice, uint32_t discriminant);

/// Encodes a length for a sequence of values
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - count - the number of elements in the sequence
/// *Side Effects / Returns*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If encoding was successful, slice.len will be incremented by the number of encoded bytes
postcard_error_t postcard_start_seq(postcard_slice_t* slice, size_t count);

/// Encodes a length for a map of key-value pairs
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - count - the number of key-value pairs in the map
/// *Side Effects / Returns*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If encoding was successful, slice.len will be incremented by the number of encoded bytes
postcard_error_t postcard_start_map(postcard_slice_t* slice, size_t count);

/// Decodes a boolean value from the slice
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - (out) value - an out pointer to the boolean value to be decoded
/// *Side Effects / Returns*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If decoding was successful, slice.len will be incremented by the number of decoded bytes
postcard_error_t postcard_decode_bool(postcard_slice_t* slice, bool* value);
/// Decodes a int8_t from the slice
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - (out) value - an out pointer to the signed 8-bit integer to be decoded
/// *Side Effects / Returns*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If decoding was successful, slice.len will be incremented by the number of decoded bytes
postcard_error_t postcard_decode_i8(postcard_slice_t* slice, int8_t* value);
/// Decodes a uint8_t from the slice
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - (out) value - an out pointer to the unsigned 8-bit integer to be decoded
/// *Side Effects / Returns*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If decoding was successful, slice.len will be incremented by the number of decoded bytes
postcard_error_t postcard_decode_u8(postcard_slice_t* slice, uint8_t* value);
/// Decodes a uint16_t from the slice
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - (out) value - an out pointer to the unsigned 16-bit integer to be decoded
/// *Side Effects / Returns*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If decoding was successful, slice.len will be incremented by the number of decoded bytes
postcard_error_t postcard_decode_u16(postcard_slice_t* slice, uint16_t* value);
/// Decodes a int16_t from the slice
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - (out) value - an out pointer to the signed 16-bit integer to be decoded
/// *Side Effects / Returns*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If decoding was successful, slice.len will be incremented by the number of decoded bytes
postcard_error_t postcard_decode_i16(postcard_slice_t* slice, int16_t* value);
/// Decodes a uint32_t from the slice
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - (out) value - an out pointer to the unsigned 32-bit integer to be decoded
/// *Side Effects / Returns*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If decoding was successful, slice.len will be incremented by the number of decoded bytes
postcard_error_t postcard_decode_u32(postcard_slice_t* slice, uint32_t* value);
/// Decodes a int32_t from the slice
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - (out) value - an out pointer to the signed 32-bit integer to be decoded
/// *Side Effects / Returns*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If decoding was successful, slice.len will be incremented by the number of decoded bytes
postcard_error_t postcard_decode_i32(postcard_slice_t* slice, int32_t* value);
/// Decodes a uint64_t from the slice
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - (out) value - an out pointer to the unsigned 64-bit integer to be decoded
/// *Side Effects / Returns*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If decoding was successful, slice.len will be incremented by the number of decoded bytes
postcard_error_t postcard_decode_u64(postcard_slice_t* slice, uint64_t* value);
/// Decodes a int64_t from the slice
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - (out) value - an out pointer to the signed 64-bit integer to be decoded
/// *Side Effects / Returns*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If decoding was successful, slice.len will be incremented by the number of decoded bytes
postcard_error_t postcard_decode_i64(postcard_slice_t* slice, int64_t* value);
/// Decodes a float32_t from the slice
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - (out) value - an out pointer to the float32_t to be decoded
/// *Side Effects / Returns*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If decoding was successful, slice.len will be incremented by the number of decoded bytes
postcard_error_t postcard_decode_f32(postcard_slice_t* slice, float* value);
/// Decodes a float64_t from the slice
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - (out) value - an out pointer to the float64_t to be decoded
/// *Side Effects / Returns*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If decoding was successful, slice.len will be incremented by the number of decoded bytes
postcard_error_t postcard_decode_f64(postcard_slice_t* slice, double* value);

/// Decodes the length of a byte array from the slice
///
/// This should be called before `postcard_decode_byte_array` to determine the size of the array to be decoded.
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - length - an out pointer to the length of the byte array to be decoded
/// *Side Effects / Returns*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If decoding was successful, slice.len will be incremented by the number of decoded bytes
postcard_error_t postcard_decode_byte_array_len(postcard_slice_t* slice, size_t* length);

/// Decodes a byte array from the slice
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - bytes - a pointer to the buffer to store the decoded bytes
/// - max_length - the maximum number of bytes that can be stored in the buffer
/// - actual_length - the length of the byte array to decode, usually returned by `postcard_decode_byte_array_len`
/// *Side Effects / Returns*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If decoding was successful, slice.len will be incremented by the number of decoded bytes
postcard_error_t postcard_decode_byte_array(postcard_slice_t* slice, uint8_t* bytes, size_t max_length, size_t actual_length);

/// Decodes the length of a string from the slice
///
/// This should be called before `postcard_decode_string` to determine the size of the array to be decoded. This function is a wrapper around `postcard_decode_byte_array_len`.
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - length - an out pointer to the length of the byte array to be decoded
/// *Side Effects / Returns*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If decoding was successful, slice.len will be incremented by the number of decoded bytes
postcard_error_t postcard_decode_string_len(postcard_slice_t* slice, size_t* length);

/// Decodes a string from the slice
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - (out) string - a pointer to the string to store the decoded bytes
/// - max_length - the maximum number of bytes that can be stored in the string
/// - actual_length - the length of the byte array to decode, usually returned by `postcard_decode_byte_array_len`
/// *Side Effects / Returns*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If decoding was successful, slice.len will be incremented by the number of decoded bytes
postcard_error_t postcard_decode_string(postcard_slice_t* slice, char* string, size_t max_length, size_t actual_length);

/// Decodes an option tag from the slice, returning whether it is some or none
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - (out) is_some - a pointer to a boolean to store the decoded option tag
/// *Side Effects / Returns*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If decoding was successful, slice.len will be incremented by the number of decoded bytes
postcard_error_t postcard_decode_option_tag(postcard_slice_t* slice, bool* is_some);

/// Decodes the variant tag from the slice, returning the discriminant
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - (out) discriminant - a pointer to a uint32_t to store the decoded variant tag
/// *Side Effects / Returns*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If decoding was successful, slice.len will be incremented by the number of decoded bytes
postcard_error_t postcard_decode_variant(postcard_slice_t* slice, uint32_t* discriminant);

/// Decodes the length of a sequence from the slice, returning the count
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - (out) count - a pointer to a size_t to store the decoded sequence length
/// *Side Effects / Returns*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If decoding was successful, slice.len will be incremented by the number of decoded bytes
postcard_error_t postcard_decode_seq_len(postcard_slice_t* slice, size_t* count);

/// Decodes the length of a map from the slice, returning the count
///
/// *Arguments*
/// - slice - a pointer to an initialized `postcard_slice_t`
/// - (out) count - a pointer to a size_t to store the decoded map length
/// *Side Effects / Returns*
/// If there is not enough room in the buffer `postcard_error_t` will return a non-zero value
/// If decoding was successful, slice.len will be incremented by the number of decoded bytes
postcard_error_t postcard_decode_map_len(postcard_slice_t* slice, size_t* count);

/// Returns the encoded size of a bool
size_t postcard_size_bool();
/// Returns the encoded size of a uint8_t
size_t postcard_size_u8();
/// Returns the encoded size of a int8_t
size_t postcard_size_i8();
/// Returns the encoded size of a uint16_t based on the value
size_t postcard_size_u16(uint16_t value);
/// Returns the encoded size of a int16_t based on the value
size_t postcard_size_i16(int16_t value);
/// Returns the encoded size of a uint32_t based on the value
size_t postcard_size_u32(uint32_t value);
/// Returns the encoded size of a int32_t based on the value
size_t postcard_size_i32(int32_t value);
/// Returns the encoded size of a uint64_t based on the value
size_t postcard_size_u64(uint64_t value);
/// Returns the encoded size of a int64_t based on the value
size_t postcard_size_i64(int64_t value);
/// Returns the encoded size of a float based on the value
size_t postcard_size_f32(float value);
/// Returns the encoded size of a double based on the value
size_t postcard_size_f64(double value);
/// Returns the encoded size of a string based on the length
size_t postcard_size_string(size_t length);
/// Returns the encoded size of a byte array based on the length
size_t postcard_size_byte_array(size_t length);
/// Returns the encoded size of a none variant
size_t postcard_size_option_none();
/// Returns the encoded size of a some variant including the inner size
size_t postcard_size_option_some(size_t inner_size);
/// Returns the encoded size of a variant based on the discriminant
size_t postcard_size_variant(uint32_t discriminant);
/// Returns the encoded size of a sequence's length
size_t postcard_size_seq_len(size_t count);
/// Returns the encoded size of a maps's length
size_t postcard_size_map_len(size_t count);
/// Returns the size of an unsigned varint based on the value
size_t postcard_size_unsigned_varint(uint64_t value);
/// Returns the size of an signed varint based on the value
size_t postcard_size_signed_varint(int64_t value);

inline void postcard_init_slice(postcard_slice_t* slice, uint8_t* buffer, size_t capacity)
{
    slice->data = buffer;
    slice->len = 0;
    slice->capacity = capacity;
}

inline static postcard_error_t encode_unsigned_varint(postcard_slice_t* slice, uint64_t value, size_t max_bytes)
{
    if (!slice || !slice->data)
        return POSTCARD_ERROR_INVALID_INPUT;

    size_t i = 0;
    while (value >= 0x80) {
        if (slice->len + i >= slice->capacity)
            return POSTCARD_ERROR_BUFFER_TOO_SMALL;
        if (i >= max_bytes)
            return POSTCARD_ERROR_OVERFLOW;

        slice->data[slice->len + i] = (value & 0x7f) | 0x80;
        value >>= 7;
        i++;
    }

    if (slice->len + i >= slice->capacity)
        return POSTCARD_ERROR_BUFFER_TOO_SMALL;
    slice->data[slice->len + i] = value & 0x7f;
    slice->len += i + 1;

    return POSTCARD_SUCCESS;
}

// Encode a signed integer as a zigzag-encoded varint
inline static postcard_error_t encode_signed_varint(postcard_slice_t* slice, int64_t value, size_t max_bytes)
{
    // Zigzag encoding: (n << 1) ^ (n >> 63)
    uint64_t zigzag = (value << 1) ^ (value >> 63);
    return encode_unsigned_varint(slice, zigzag, max_bytes);
}

inline static postcard_error_t decode_unsigned_varint(postcard_slice_t* slice, uint64_t* value, size_t max_bytes)
{
    if (!slice || !slice->data || !value)
        return POSTCARD_ERROR_INVALID_INPUT;

    *value = 0;
    uint64_t shift = 0;
    size_t i = 0;

    while (i < max_bytes) {
        if (slice->len >= slice->capacity)
            return POSTCARD_ERROR_INCOMPLETE_DATA;

        uint8_t byte = slice->data[slice->len];
        slice->len++;
        i++;

        *value |= ((uint64_t)(byte & 0x7F)) << shift;
        if (!(byte & 0x80))
            return POSTCARD_SUCCESS;

        shift += 7;
        if (shift > 63)
            return POSTCARD_ERROR_OVERFLOW;
    }

    return POSTCARD_ERROR_OVERFLOW;
}

// Decode a signed varint (zigzag encoded)
inline static postcard_error_t decode_signed_varint(postcard_slice_t* slice, int64_t* value, size_t max_bytes)
{
    uint64_t zigzag;
    postcard_error_t err = decode_unsigned_varint(slice, &zigzag, max_bytes);
    if (err != POSTCARD_SUCCESS)
        return err;

    // Zigzag decoding: (n >> 1) ^ (-(n & 1))
    *value = (zigzag >> 1) ^ (-(zigzag & 1));
    return POSTCARD_SUCCESS;
}

inline postcard_error_t postcard_encode_bool(postcard_slice_t* slice, bool value)
{
    if (!slice || !slice->data)
        return POSTCARD_ERROR_INVALID_INPUT;
    if (slice->len >= slice->capacity)
        return POSTCARD_ERROR_BUFFER_TOO_SMALL;

    slice->data[slice->len++] = value ? 0x01 : 0x00;
    return POSTCARD_SUCCESS;
}

inline postcard_error_t postcard_encode_u8(postcard_slice_t* slice, uint8_t value)
{
    if (!slice || !slice->data)
        return POSTCARD_ERROR_INVALID_INPUT;
    if (slice->len >= slice->capacity)
        return POSTCARD_ERROR_BUFFER_TOO_SMALL;

    slice->data[slice->len++] = value;
    return POSTCARD_SUCCESS;
}

inline postcard_error_t postcard_encode_i8(postcard_slice_t* slice, int8_t value)
{
    if (!slice || !slice->data)
        return POSTCARD_ERROR_INVALID_INPUT;
    if (slice->len >= slice->capacity)
        return POSTCARD_ERROR_BUFFER_TOO_SMALL;

    slice->data[slice->len++] = (uint8_t)value;
    return POSTCARD_SUCCESS;
}

inline postcard_error_t postcard_encode_u16(postcard_slice_t* slice, uint16_t value)
{
    return encode_unsigned_varint(slice, value, 3);
}

inline postcard_error_t postcard_encode_i16(postcard_slice_t* slice, int16_t value)
{
    return encode_signed_varint(slice, value, 3);
}

inline postcard_error_t postcard_encode_u32(postcard_slice_t* slice, uint32_t value)
{
    return encode_unsigned_varint(slice, value, 5);
}

inline postcard_error_t postcard_encode_i32(postcard_slice_t* slice, int32_t value)
{
    return encode_signed_varint(slice, value, 5);
}

inline postcard_error_t postcard_encode_u64(postcard_slice_t* slice, uint64_t value)
{
    return encode_unsigned_varint(slice, value, 10);
}

inline postcard_error_t postcard_encode_i64(postcard_slice_t* slice, int64_t value)
{
    return encode_signed_varint(slice, value, 10);
}

inline postcard_error_t postcard_encode_f32(postcard_slice_t* slice, float value)
{
    if (!slice || !slice->data)
        return POSTCARD_ERROR_INVALID_INPUT;
    if (slice->len + 4 > slice->capacity)
        return POSTCARD_ERROR_BUFFER_TOO_SMALL;

    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));

    // Little-endian encoding
    slice->data[slice->len++] = (bits >> 0) & 0xFF;
    slice->data[slice->len++] = (bits >> 8) & 0xFF;
    slice->data[slice->len++] = (bits >> 16) & 0xFF;
    slice->data[slice->len++] = (bits >> 24) & 0xFF;

    return POSTCARD_SUCCESS;
}

inline postcard_error_t postcard_encode_f64(postcard_slice_t* slice, double value)
{
    if (!slice || !slice->data)
        return POSTCARD_ERROR_INVALID_INPUT;
    if (slice->len + 8 > slice->capacity)
        return POSTCARD_ERROR_BUFFER_TOO_SMALL;

    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));

    // Little-endian encoding
    slice->data[slice->len++] = (bits >> 0) & 0xFF;
    slice->data[slice->len++] = (bits >> 8) & 0xFF;
    slice->data[slice->len++] = (bits >> 16) & 0xFF;
    slice->data[slice->len++] = (bits >> 24) & 0xFF;
    slice->data[slice->len++] = (bits >> 32) & 0xFF;
    slice->data[slice->len++] = (bits >> 40) & 0xFF;
    slice->data[slice->len++] = (bits >> 48) & 0xFF;
    slice->data[slice->len++] = (bits >> 56) & 0xFF;

    return POSTCARD_SUCCESS;
}

inline postcard_error_t postcard_encode_byte_array(postcard_slice_t* slice, const uint8_t* bytes, size_t length)
{
    if (!slice || !slice->data || (!bytes && length > 0))
        return POSTCARD_ERROR_INVALID_INPUT;

    // encode the length of the byte array
    postcard_error_t err = encode_unsigned_varint(slice, length, 10);
    if (err != POSTCARD_SUCCESS)
        return err;

    // check if we have enough space for the data
    if (slice->len + length > slice->capacity)
        return POSTCARD_ERROR_BUFFER_TOO_SMALL;

    // Copy the data
    if (length > 0) {
        memcpy(slice->data + slice->len, bytes, length);
        slice->len += length;
    }

    return POSTCARD_SUCCESS;
}

inline postcard_error_t postcard_encode_string(postcard_slice_t* slice, const char* string, size_t length)
{
    return postcard_encode_byte_array(slice, (const uint8_t*)string, length);
}

inline postcard_error_t postcard_encode_option_none(postcard_slice_t* slice)
{
    if (!slice || !slice->data)
        return POSTCARD_ERROR_INVALID_INPUT;
    if (slice->len >= slice->capacity)
        return POSTCARD_ERROR_BUFFER_TOO_SMALL;

    slice->data[slice->len++] = 0x00;
    return POSTCARD_SUCCESS;
}

inline postcard_error_t postcard_encode_option_some(postcard_slice_t* slice)
{
    if (!slice || !slice->data)
        return POSTCARD_ERROR_INVALID_INPUT;
    if (slice->len >= slice->capacity)
        return POSTCARD_ERROR_BUFFER_TOO_SMALL;

    slice->data[slice->len++] = 0x01;
    return POSTCARD_SUCCESS;
}

inline postcard_error_t postcard_encode_variant(postcard_slice_t* slice, uint32_t discriminant)
{
    return postcard_encode_u32(slice, discriminant);
}

inline postcard_error_t postcard_start_seq(postcard_slice_t* slice, size_t count)
{
    return encode_unsigned_varint(slice, count, 10);
}

inline postcard_error_t postcard_start_map(postcard_slice_t* slice, size_t count)
{
    return encode_unsigned_varint(slice, count, 10);
}

inline postcard_error_t postcard_decode_bool(postcard_slice_t* slice, bool* value)
{
    if (!slice || !slice->data || !value)
        return POSTCARD_ERROR_INVALID_INPUT;
    if (slice->len >= slice->capacity)
        return POSTCARD_ERROR_INCOMPLETE_DATA;

    uint8_t byte = slice->data[slice->len++];
    if (byte == 0x00) {
        *value = false;
        return POSTCARD_SUCCESS;
    } else if (byte == 0x01) {
        *value = true;
        return POSTCARD_SUCCESS;
    } else {
        return POSTCARD_ERROR_INVALID_INPUT;
    }
}

inline postcard_error_t postcard_decode_u8(postcard_slice_t* slice, uint8_t* value)
{
    if (!slice || !slice->data || !value)
        return POSTCARD_ERROR_INVALID_INPUT;
    if (slice->len >= slice->capacity)
        return POSTCARD_ERROR_INCOMPLETE_DATA;

    *value = slice->data[slice->len++];
    return POSTCARD_SUCCESS;
}

inline postcard_error_t postcard_decode_i8(postcard_slice_t* slice, int8_t* value)
{
    if (!slice || !slice->data || !value)
        return POSTCARD_ERROR_INVALID_INPUT;
    if (slice->len >= slice->capacity)
        return POSTCARD_ERROR_INCOMPLETE_DATA;

    *value = (int8_t)slice->data[slice->len++];
    return POSTCARD_SUCCESS;
}

inline postcard_error_t postcard_decode_u16(postcard_slice_t* slice, uint16_t* value)
{
    uint64_t val;
    postcard_error_t err = decode_unsigned_varint(slice, &val, 3);
    if (err != POSTCARD_SUCCESS)
        return err;

    if (val > UINT16_MAX)
        return POSTCARD_ERROR_OVERFLOW;
    *value = (uint16_t)val;
    return POSTCARD_SUCCESS;
}

inline postcard_error_t postcard_decode_i16(postcard_slice_t* slice, int16_t* value)
{
    int64_t val;
    postcard_error_t err = decode_signed_varint(slice, &val, 3);
    if (err != POSTCARD_SUCCESS)
        return err;

    if (val < INT16_MIN || val > INT16_MAX)
        return POSTCARD_ERROR_OVERFLOW;
    *value = (int16_t)val;
    return POSTCARD_SUCCESS;
}

inline postcard_error_t postcard_decode_u32(postcard_slice_t* slice, uint32_t* value)
{
    uint64_t val;
    postcard_error_t err = decode_unsigned_varint(slice, &val, 5);
    if (err != POSTCARD_SUCCESS)
        return err;

    if (val > UINT32_MAX)
        return POSTCARD_ERROR_OVERFLOW;
    *value = (uint32_t)val;
    return POSTCARD_SUCCESS;
}

inline postcard_error_t postcard_decode_i32(postcard_slice_t* slice, int32_t* value)
{
    int64_t val;
    postcard_error_t err = decode_signed_varint(slice, &val, 5);
    if (err != POSTCARD_SUCCESS)
        return err;

    if (val < INT32_MIN || val > INT32_MAX)
        return POSTCARD_ERROR_OVERFLOW;
    *value = (int32_t)val;
    return POSTCARD_SUCCESS;
}

inline postcard_error_t postcard_decode_u64(postcard_slice_t* slice, uint64_t* value)
{
    return decode_unsigned_varint(slice, value, 10);
}

inline postcard_error_t postcard_decode_i64(postcard_slice_t* slice, int64_t* value)
{
    return decode_signed_varint(slice, value, 10);
}

inline postcard_error_t postcard_decode_f32(postcard_slice_t* slice, float* value)
{
    if (!slice || !slice->data || !value)
        return POSTCARD_ERROR_INVALID_INPUT;
    if (slice->len + 4 > slice->capacity)
        return POSTCARD_ERROR_INCOMPLETE_DATA;

    uint32_t bits = 0;
    bits |= (uint32_t)slice->data[slice->len++] << 0;
    bits |= (uint32_t)slice->data[slice->len++] << 8;
    bits |= (uint32_t)slice->data[slice->len++] << 16;
    bits |= (uint32_t)slice->data[slice->len++] << 24;

    memcpy(value, &bits, sizeof(*value));
    return POSTCARD_SUCCESS;
}

inline postcard_error_t postcard_decode_f64(postcard_slice_t* slice, double* value)
{
    if (!slice || !slice->data || !value)
        return POSTCARD_ERROR_INVALID_INPUT;
    if (slice->len + 8 > slice->capacity)
        return POSTCARD_ERROR_INCOMPLETE_DATA;

    uint64_t bits = 0;
    bits |= (uint64_t)slice->data[slice->len++] << 0;
    bits |= (uint64_t)slice->data[slice->len++] << 8;
    bits |= (uint64_t)slice->data[slice->len++] << 16;
    bits |= (uint64_t)slice->data[slice->len++] << 24;
    bits |= (uint64_t)slice->data[slice->len++] << 32;
    bits |= (uint64_t)slice->data[slice->len++] << 40;
    bits |= (uint64_t)slice->data[slice->len++] << 48;
    bits |= (uint64_t)slice->data[slice->len++] << 56;

    memcpy(value, &bits, sizeof(*value));
    return POSTCARD_SUCCESS;
}

inline postcard_error_t postcard_decode_byte_array_len(postcard_slice_t* slice, size_t* length)
{
    uint64_t len;
    postcard_error_t err = decode_unsigned_varint(slice, &len, 10);
    if (err != POSTCARD_SUCCESS)
        return err;

    if (len > SIZE_MAX)
        return POSTCARD_ERROR_OVERFLOW;

    *length = (size_t)len;
    return POSTCARD_SUCCESS;
}

inline postcard_error_t postcard_decode_byte_array(postcard_slice_t* slice, uint8_t* bytes,
    size_t max_length, size_t actual_length)
{
    if (!slice || !slice->data || !actual_length)
        return POSTCARD_ERROR_INVALID_INPUT;

    if (actual_length + slice->len > slice->capacity)
        return POSTCARD_ERROR_INCOMPLETE_DATA;

    if (actual_length > max_length)
        return POSTCARD_ERROR_BUFFER_TOO_SMALL;

    if (bytes && actual_length > 0) {
        memcpy(bytes, slice->data + slice->len, actual_length);
    }
    slice->len += actual_length;

    return POSTCARD_SUCCESS;
}

inline postcard_error_t postcard_decode_string_len(postcard_slice_t* slice, size_t* length)
{
    return postcard_decode_byte_array_len(slice, length);
}

inline postcard_error_t postcard_decode_string(postcard_slice_t* slice, char* string,
    size_t max_length, size_t actual_length)
{
    return postcard_decode_byte_array(slice, (uint8_t*)string, max_length, actual_length);
}

inline postcard_error_t postcard_decode_option_tag(postcard_slice_t* slice, bool* is_some)
{
    if (!slice || !slice->data || !is_some)
        return POSTCARD_ERROR_INVALID_INPUT;
    if (slice->len >= slice->capacity)
        return POSTCARD_ERROR_INCOMPLETE_DATA;

    uint8_t tag = slice->data[slice->len++];
    if (tag == 0x00) {
        *is_some = false;
        return POSTCARD_SUCCESS;
    } else if (tag == 0x01) {
        *is_some = true;
        return POSTCARD_SUCCESS;
    } else {
        return POSTCARD_ERROR_INVALID_INPUT;
    }
}

inline postcard_error_t postcard_decode_variant(postcard_slice_t* slice, uint32_t* discriminant)
{
    return postcard_decode_u32(slice, discriminant);
}

inline postcard_error_t postcard_decode_seq_len(postcard_slice_t* slice, size_t* count)
{
    uint64_t len;
    postcard_error_t err = decode_unsigned_varint(slice, &len, 10);
    if (err != POSTCARD_SUCCESS)
        return err;

    if (len > SIZE_MAX)
        return POSTCARD_ERROR_OVERFLOW;

    *count = (size_t)len;
    return POSTCARD_SUCCESS;
}

inline postcard_error_t postcard_decode_map_len(postcard_slice_t* slice, size_t* count)
{
    return postcard_decode_seq_len(slice, count);
}

inline size_t postcard_size_unsigned_varint(uint64_t value)
{
    if (value < 0x80) {
        return 1;
    } else if (value < 0x4000) {
        return 2;
    } else if (value < 0x200000) {
        return 3;
    } else if (value < 0x10000000) {
        return 4;
    } else if (value < 0x800000000) {
        return 5;
    } else if (value < 0x40000000000) {
        return 6;
    } else if (value < 0x2000000000000) {
        return 7;
    } else if (value < 0x100000000000000) {
        return 8;
    } else if (value < 0x8000000000000000) {
        return 9;
    } else {
        return 10;
    }
}

// Helper to calculate the size of a signed varint (zigzag encoded)
inline size_t postcard_size_signed_varint(int64_t value)
{
    // Zigzag encoding: (n << 1) ^ (n >> 63)
    uint64_t zigzag = (value << 1) ^ (value >> 63);
    return postcard_size_unsigned_varint(zigzag);
}

// Basic types
inline size_t postcard_size_bool()
{
    return 1;
}

inline size_t postcard_size_u8()
{
    return 1;
}

inline size_t postcard_size_i8()
{
    return 1;
}

inline size_t postcard_size_u16(uint16_t value)
{
    return postcard_size_unsigned_varint(value);
}

inline size_t postcard_size_i16(int16_t value)
{
    return postcard_size_signed_varint(value);
}

inline size_t postcard_size_u32(uint32_t value)
{
    return postcard_size_unsigned_varint(value);
}

inline size_t postcard_size_i32(int32_t value)
{
    return postcard_size_signed_varint(value);
}

inline size_t postcard_size_u64(uint64_t value)
{
    return postcard_size_unsigned_varint(value);
}

inline size_t postcard_size_i64(int64_t value)
{
    return postcard_size_signed_varint(value);
}

inline size_t postcard_size_f32()
{
    return 4;
}

inline size_t postcard_size_f64()
{
    return 8;
}

inline size_t postcard_size_byte_array(size_t length)
{
    return postcard_size_unsigned_varint(length) + length;
}

inline size_t postcard_size_string(size_t length)
{
    return postcard_size_byte_array(length);
}

inline size_t postcard_size_option_none()
{
    return 1;
}

inline size_t postcard_size_option_some(size_t inner_size)
{
    return 1 + inner_size;
}

inline size_t postcard_size_variant(uint32_t discriminant)
{
    return postcard_size_u32(discriminant);
}

inline size_t postcard_size_seq(size_t count)
{
    return postcard_size_unsigned_varint(count);
}

inline size_t postcard_size_map(size_t count)
{
    return postcard_size_unsigned_varint(count);
}

#endif // POSTCARD_H


class InitialTimestamp : public std::variant<std::monostate, std::monostate, int64_t> {
public:
  static constexpr std::string_view TYPE_NAME = "InitialTimestamp";
  // Inherit constructors from std::variant
  using std::variant<std::monostate, std::monostate, int64_t>::variant;

  
  // Static constructor for unit variant Earliest
  static InitialTimestamp Earliest() {
    return InitialTimestamp{std::in_place_index<0>, std::monostate{}};
  }
  

  // Accessor method for variant Earliest
  bool is_earliest() const {
    return this->index() == 0;
  }
  
  const std::monostate* get_earliest() const {
    return std::get_if<0>((const std::variant<std::monostate, std::monostate, int64_t>*)this);
  }

  std::monostate* get_earliest() {
    return std::get_if<0>((std::variant<std::monostate, std::monostate, int64_t>*)this);
  }
  

  
  // Static constructor for unit variant Latest
  static InitialTimestamp Latest() {
    return InitialTimestamp{std::in_place_index<1>, std::monostate{}};
  }
  

  // Accessor method for variant Latest
  bool is_latest() const {
    return this->index() == 1;
  }
  
  const std::monostate* get_latest() const {
    return std::get_if<1>((const std::variant<std::monostate, std::monostate, int64_t>*)this);
  }

  std::monostate* get_latest() {
    return std::get_if<1>((std::variant<std::monostate, std::monostate, int64_t>*)this);
  }
  

  
  // Static constructor for Manual variant
  static InitialTimestamp Manual(const int64_t& value) {
    return InitialTimestamp{std::in_place_index<2>, value};
  }
  

  // Accessor method for variant Manual
  bool is_manual() const {
    return this->index() == 2;
  }
  
  const int64_t* get_manual() const {
    return std::get_if<2>((const std::variant<std::monostate, std::monostate, int64_t>*)this);
  }

  int64_t* get_manual() {
    return std::get_if<2>((std::variant<std::monostate, std::monostate, int64_t>*)this);
  }
  

  

  size_t encoded_size() const {
    size_t size = 0;

    // Tag size (discriminant)
    size += postcard_size_u8(); // Just for the variant tag

    if ([[maybe_unused]] auto val = std::get_if<0>((const std::variant<std::monostate, std::monostate, int64_t>*)this)) {
        
    }
    else if ([[maybe_unused]] auto val = std::get_if<1>((const std::variant<std::monostate, std::monostate, int64_t>*)this)) {
        
    }
    else if ([[maybe_unused]] auto val = std::get_if<2>((const std::variant<std::monostate, std::monostate, int64_t>*)this)) {
        size += postcard_size_i64((*val));
    }
    

    return size;
  }

  postcard_error_t encode(std::span<uint8_t>& output) const {
    postcard_slice_t slice;
    postcard_init_slice(&slice, output.data(), output.size());
    auto res = encode_raw(&slice);
    if(res != POSTCARD_SUCCESS) return res;
    output = output.subspan(0, slice.len);
    return POSTCARD_SUCCESS;
  }

  std::vector<uint8_t> encode_vec() const {
    // Pre-allocate vector with the required size
    std::vector<uint8_t> vec(encoded_size());

    // Create a span from the vector
    auto span = std::span<uint8_t>(vec);

    // Encode into the span
    postcard_slice_t slice;
    postcard_init_slice(&slice, span.data(), span.size());
    auto res = encode_raw(&slice);

    // Resize to actual used length if successful
    if (res == POSTCARD_SUCCESS) {
      vec.resize(slice.len);
    } else {
      vec.clear(); // Clear the vector on error
    }

    return vec;
  }

  postcard_error_t encode_raw(postcard_slice_t* slice) const {
    postcard_error_t result;

    if ([[maybe_unused]] auto val = std::get_if<0>((std::variant<std::monostate, std::monostate, int64_t>*)this)) {
        result = postcard_encode_u8(slice, 0);
        if (result != POSTCARD_SUCCESS) return result;
        
        if (result != POSTCARD_SUCCESS) return result;
    }
    else if ([[maybe_unused]] auto val = std::get_if<1>((std::variant<std::monostate, std::monostate, int64_t>*)this)) {
        result = postcard_encode_u8(slice, 1);
        if (result != POSTCARD_SUCCESS) return result;
        
        if (result != POSTCARD_SUCCESS) return result;
    }
    else if ([[maybe_unused]] auto val = std::get_if<2>((std::variant<std::monostate, std::monostate, int64_t>*)this)) {
        result = postcard_encode_u8(slice, 2);
        if (result != POSTCARD_SUCCESS) return result;
        result = postcard_encode_i64(slice, (*val));
        if (result != POSTCARD_SUCCESS) return result;
    }
    
    else {
        return POSTCARD_ERROR_INVALID_INPUT;
    }

    return POSTCARD_SUCCESS;
  }

  postcard_error_t decode(std::span<const uint8_t>& input) {
    postcard_slice_t slice;
    postcard_init_slice(&slice, const_cast<uint8_t*>(input.data()), input.size());
    postcard_error_t result = decode_raw(&slice);
    if (result == POSTCARD_SUCCESS) {
      // Update the input span to point past the decoded data
      input = input.subspan(slice.len);
    }
    return result;
  }

  postcard_error_t decode_raw(postcard_slice_t* slice) {
    postcard_error_t result;
    uint8_t tag;
    result = postcard_decode_u8(slice, &tag);
    if (result != POSTCARD_SUCCESS) return result;

    switch(tag) {
        case 0: {  // Earliest
            
            this->emplace<0>(std::monostate{});
            
            break;
        }
        case 1: {  // Latest
            
            this->emplace<1>(std::monostate{});
            
            break;
        }
        case 2: {  // Manual
            
            int64_t val;
            result = postcard_decode_i64(slice, &val);
            if (result != POSTCARD_SUCCESS) return result;
            this->emplace<2>(val);
            
            break;
        }
        
        default:
            return POSTCARD_ERROR_INVALID_INPUT;
    }

    return POSTCARD_SUCCESS;
  }
};

struct FixedRateBehavior {
  static constexpr std::string_view TYPE_NAME = "FixedRateBehavior";

  InitialTimestamp initial_timestamp;
  std::optional<uint64_t> timestep;
  std::optional<uint64_t> frequency;
  


  size_t encoded_size() const {
    size_t size = 0;
    size += initial_timestamp.encoded_size();
    if(timestep) {
                        size += postcard_size_option_some(0);
                        size += postcard_size_u64(*timestep);
                    }else{
                        size += postcard_size_option_none();
                    }
                    
    if(frequency) {
                        size += postcard_size_option_some(0);
                        size += postcard_size_u64(*frequency);
                    }else{
                        size += postcard_size_option_none();
                    }
                    
    
    return size;
  }

  postcard_error_t encode(std::span<uint8_t>& output) const {
    postcard_slice_t slice;
    postcard_init_slice(&slice, output.data(), output.size());
    auto res = encode_raw(&slice);
    if(res != POSTCARD_SUCCESS) return res;
    output = output.subspan(0, slice.len);
    return POSTCARD_SUCCESS;
  }

  std::vector<uint8_t> encode_vec() const {
    // Pre-allocate vector with the required size
    std::vector<uint8_t> vec(encoded_size());

    // Create a span from the vector
    auto span = std::span<uint8_t>(vec);

    // Encode into the span
    postcard_slice_t slice;
    postcard_init_slice(&slice, span.data(), span.size());
    auto res = encode_raw(&slice);

    // Resize to actual used length if successful
    if (res == POSTCARD_SUCCESS) {
      vec.resize(slice.len);
    } else {
      vec.clear(); // Clear the vector on error
    }

    return vec;
  }

  postcard_error_t encode_raw(postcard_slice_t* slice) const {
    postcard_error_t result;
    result = initial_timestamp.encode_raw(slice);
        if(result != POSTCARD_SUCCESS) return result;
    if(timestep) {
                        result = postcard_encode_option_some(slice); if(result != POSTCARD_SUCCESS) return result;
                        result = postcard_encode_u64(slice, *timestep);
                    }else{
                        result = postcard_encode_option_none(slice);
                    }
                    
        if(result != POSTCARD_SUCCESS) return result;
    if(frequency) {
                        result = postcard_encode_option_some(slice); if(result != POSTCARD_SUCCESS) return result;
                        result = postcard_encode_u64(slice, *frequency);
                    }else{
                        result = postcard_encode_option_none(slice);
                    }
                    
        if(result != POSTCARD_SUCCESS) return result;
    
    return POSTCARD_SUCCESS;
  }

  postcard_error_t decode(std::span<const uint8_t>& input) {
    postcard_slice_t slice;
    postcard_init_slice(&slice, const_cast<uint8_t*>(input.data()), input.size());
    postcard_error_t result = decode_raw(&slice);
    if (result == POSTCARD_SUCCESS) {
      // Update the input span to point past the decoded data
      input = input.subspan(slice.len);
    }
    return result;
  }

  postcard_error_t decode_raw(postcard_slice_t* slice) {
    postcard_error_t result;
    result = initial_timestamp.decode_raw(slice);
    if(result != POSTCARD_SUCCESS) return result;

    {bool is_some;
    result = postcard_decode_option_tag(slice, &is_some);
    if (result != POSTCARD_SUCCESS) return result;
    if (is_some) {
        uint64_t val;
        result = postcard_decode_u64(slice, &val);
        if (result != POSTCARD_SUCCESS) return result;
        timestep = val;
    } else {
        timestep = std::nullopt;
    }}
    if(result != POSTCARD_SUCCESS) return result;

    {bool is_some;
    result = postcard_decode_option_tag(slice, &is_some);
    if (result != POSTCARD_SUCCESS) return result;
    if (is_some) {
        uint64_t val;
        result = postcard_decode_u64(slice, &val);
        if (result != POSTCARD_SUCCESS) return result;
        frequency = val;
    } else {
        frequency = std::nullopt;
    }}
    if(result != POSTCARD_SUCCESS) return result;

    
    return POSTCARD_SUCCESS;
  }
};

class StreamBehavior : public std::variant<std::monostate, FixedRateBehavior> {
public:
  static constexpr std::string_view TYPE_NAME = "StreamBehavior";
  // Inherit constructors from std::variant
  using std::variant<std::monostate, FixedRateBehavior>::variant;

  
  // Static constructor for unit variant RealTime
  static StreamBehavior RealTime() {
    return StreamBehavior{std::in_place_index<0>, std::monostate{}};
  }
  

  // Accessor method for variant RealTime
  bool is_real_time() const {
    return this->index() == 0;
  }
  
  const std::monostate* get_real_time() const {
    return std::get_if<0>((const std::variant<std::monostate, FixedRateBehavior>*)this);
  }

  std::monostate* get_real_time() {
    return std::get_if<0>((std::variant<std::monostate, FixedRateBehavior>*)this);
  }
  

  
  // Static constructor for FixedRate variant
  static StreamBehavior FixedRate(const FixedRateBehavior& value) {
    return StreamBehavior{std::in_place_index<1>, value};
  }
  

  // Accessor method for variant FixedRate
  bool is_fixed_rate() const {
    return this->index() == 1;
  }
  
  const FixedRateBehavior* get_fixed_rate() const {
    return std::get_if<1>((const std::variant<std::monostate, FixedRateBehavior>*)this);
  }

  FixedRateBehavior* get_fixed_rate() {
    return std::get_if<1>((std::variant<std::monostate, FixedRateBehavior>*)this);
  }
  

  

  size_t encoded_size() const {
    size_t size = 0;

    // Tag size (discriminant)
    size += postcard_size_u8(); // Just for the variant tag

    if ([[maybe_unused]] auto val = std::get_if<0>((const std::variant<std::monostate, FixedRateBehavior>*)this)) {
        
    }
    else if ([[maybe_unused]] auto val = std::get_if<1>((const std::variant<std::monostate, FixedRateBehavior>*)this)) {
        size += (*val).encoded_size();
    }
    

    return size;
  }

  postcard_error_t encode(std::span<uint8_t>& output) const {
    postcard_slice_t slice;
    postcard_init_slice(&slice, output.data(), output.size());
    auto res = encode_raw(&slice);
    if(res != POSTCARD_SUCCESS) return res;
    output = output.subspan(0, slice.len);
    return POSTCARD_SUCCESS;
  }

  std::vector<uint8_t> encode_vec() const {
    // Pre-allocate vector with the required size
    std::vector<uint8_t> vec(encoded_size());

    // Create a span from the vector
    auto span = std::span<uint8_t>(vec);

    // Encode into the span
    postcard_slice_t slice;
    postcard_init_slice(&slice, span.data(), span.size());
    auto res = encode_raw(&slice);

    // Resize to actual used length if successful
    if (res == POSTCARD_SUCCESS) {
      vec.resize(slice.len);
    } else {
      vec.clear(); // Clear the vector on error
    }

    return vec;
  }

  postcard_error_t encode_raw(postcard_slice_t* slice) const {
    postcard_error_t result;

    if ([[maybe_unused]] auto val = std::get_if<0>((std::variant<std::monostate, FixedRateBehavior>*)this)) {
        result = postcard_encode_u8(slice, 0);
        if (result != POSTCARD_SUCCESS) return result;
        
        if (result != POSTCARD_SUCCESS) return result;
    }
    else if ([[maybe_unused]] auto val = std::get_if<1>((std::variant<std::monostate, FixedRateBehavior>*)this)) {
        result = postcard_encode_u8(slice, 1);
        if (result != POSTCARD_SUCCESS) return result;
        result = (*val).encode_raw(slice);
        if (result != POSTCARD_SUCCESS) return result;
    }
    
    else {
        return POSTCARD_ERROR_INVALID_INPUT;
    }

    return POSTCARD_SUCCESS;
  }

  postcard_error_t decode(std::span<const uint8_t>& input) {
    postcard_slice_t slice;
    postcard_init_slice(&slice, const_cast<uint8_t*>(input.data()), input.size());
    postcard_error_t result = decode_raw(&slice);
    if (result == POSTCARD_SUCCESS) {
      // Update the input span to point past the decoded data
      input = input.subspan(slice.len);
    }
    return result;
  }

  postcard_error_t decode_raw(postcard_slice_t* slice) {
    postcard_error_t result;
    uint8_t tag;
    result = postcard_decode_u8(slice, &tag);
    if (result != POSTCARD_SUCCESS) return result;

    switch(tag) {
        case 0: {  // RealTime
            
            this->emplace<0>(std::monostate{});
            
            break;
        }
        case 1: {  // FixedRate
            
            FixedRateBehavior val;
            result = val.decode_raw(slice);
            if (result != POSTCARD_SUCCESS) return result;
            this->emplace<1>(val);
            
            break;
        }
        
        default:
            return POSTCARD_ERROR_INVALID_INPUT;
    }

    return POSTCARD_SUCCESS;
  }
};

struct StreamFilter {
  static constexpr std::string_view TYPE_NAME = "StreamFilter";

  std::optional<uint64_t> component_id;
  std::optional<uint64_t> entity_id;
  


  size_t encoded_size() const {
    size_t size = 0;
    if(component_id) {
                        size += postcard_size_option_some(0);
                        size += postcard_size_u64(*component_id);
                    }else{
                        size += postcard_size_option_none();
                    }
                    
    if(entity_id) {
                        size += postcard_size_option_some(0);
                        size += postcard_size_u64(*entity_id);
                    }else{
                        size += postcard_size_option_none();
                    }
                    
    
    return size;
  }

  postcard_error_t encode(std::span<uint8_t>& output) const {
    postcard_slice_t slice;
    postcard_init_slice(&slice, output.data(), output.size());
    auto res = encode_raw(&slice);
    if(res != POSTCARD_SUCCESS) return res;
    output = output.subspan(0, slice.len);
    return POSTCARD_SUCCESS;
  }

  std::vector<uint8_t> encode_vec() const {
    // Pre-allocate vector with the required size
    std::vector<uint8_t> vec(encoded_size());

    // Create a span from the vector
    auto span = std::span<uint8_t>(vec);

    // Encode into the span
    postcard_slice_t slice;
    postcard_init_slice(&slice, span.data(), span.size());
    auto res = encode_raw(&slice);

    // Resize to actual used length if successful
    if (res == POSTCARD_SUCCESS) {
      vec.resize(slice.len);
    } else {
      vec.clear(); // Clear the vector on error
    }

    return vec;
  }

  postcard_error_t encode_raw(postcard_slice_t* slice) const {
    postcard_error_t result;
    if(component_id) {
                        result = postcard_encode_option_some(slice); if(result != POSTCARD_SUCCESS) return result;
                        result = postcard_encode_u64(slice, *component_id);
                    }else{
                        result = postcard_encode_option_none(slice);
                    }
                    
        if(result != POSTCARD_SUCCESS) return result;
    if(entity_id) {
                        result = postcard_encode_option_some(slice); if(result != POSTCARD_SUCCESS) return result;
                        result = postcard_encode_u64(slice, *entity_id);
                    }else{
                        result = postcard_encode_option_none(slice);
                    }
                    
        if(result != POSTCARD_SUCCESS) return result;
    
    return POSTCARD_SUCCESS;
  }

  postcard_error_t decode(std::span<const uint8_t>& input) {
    postcard_slice_t slice;
    postcard_init_slice(&slice, const_cast<uint8_t*>(input.data()), input.size());
    postcard_error_t result = decode_raw(&slice);
    if (result == POSTCARD_SUCCESS) {
      // Update the input span to point past the decoded data
      input = input.subspan(slice.len);
    }
    return result;
  }

  postcard_error_t decode_raw(postcard_slice_t* slice) {
    postcard_error_t result;
    {bool is_some;
    result = postcard_decode_option_tag(slice, &is_some);
    if (result != POSTCARD_SUCCESS) return result;
    if (is_some) {
        uint64_t val;
        result = postcard_decode_u64(slice, &val);
        if (result != POSTCARD_SUCCESS) return result;
        component_id = val;
    } else {
        component_id = std::nullopt;
    }}
    if(result != POSTCARD_SUCCESS) return result;

    {bool is_some;
    result = postcard_decode_option_tag(slice, &is_some);
    if (result != POSTCARD_SUCCESS) return result;
    if (is_some) {
        uint64_t val;
        result = postcard_decode_u64(slice, &val);
        if (result != POSTCARD_SUCCESS) return result;
        entity_id = val;
    } else {
        entity_id = std::nullopt;
    }}
    if(result != POSTCARD_SUCCESS) return result;

    
    return POSTCARD_SUCCESS;
  }
};

struct Stream {
  static constexpr std::string_view TYPE_NAME = "Stream";

  StreamFilter filter;
  StreamBehavior behavior;
  uint64_t id;
  


  size_t encoded_size() const {
    size_t size = 0;
    size += filter.encoded_size();
    size += behavior.encoded_size();
    size += postcard_size_u64(id);
    
    return size;
  }

  postcard_error_t encode(std::span<uint8_t>& output) const {
    postcard_slice_t slice;
    postcard_init_slice(&slice, output.data(), output.size());
    auto res = encode_raw(&slice);
    if(res != POSTCARD_SUCCESS) return res;
    output = output.subspan(0, slice.len);
    return POSTCARD_SUCCESS;
  }

  std::vector<uint8_t> encode_vec() const {
    // Pre-allocate vector with the required size
    std::vector<uint8_t> vec(encoded_size());

    // Create a span from the vector
    auto span = std::span<uint8_t>(vec);

    // Encode into the span
    postcard_slice_t slice;
    postcard_init_slice(&slice, span.data(), span.size());
    auto res = encode_raw(&slice);

    // Resize to actual used length if successful
    if (res == POSTCARD_SUCCESS) {
      vec.resize(slice.len);
    } else {
      vec.clear(); // Clear the vector on error
    }

    return vec;
  }

  postcard_error_t encode_raw(postcard_slice_t* slice) const {
    postcard_error_t result;
    result = filter.encode_raw(slice);
        if(result != POSTCARD_SUCCESS) return result;
    result = behavior.encode_raw(slice);
        if(result != POSTCARD_SUCCESS) return result;
    result = postcard_encode_u64(slice, id);
        if(result != POSTCARD_SUCCESS) return result;
    
    return POSTCARD_SUCCESS;
  }

  postcard_error_t decode(std::span<const uint8_t>& input) {
    postcard_slice_t slice;
    postcard_init_slice(&slice, const_cast<uint8_t*>(input.data()), input.size());
    postcard_error_t result = decode_raw(&slice);
    if (result == POSTCARD_SUCCESS) {
      // Update the input span to point past the decoded data
      input = input.subspan(slice.len);
    }
    return result;
  }

  postcard_error_t decode_raw(postcard_slice_t* slice) {
    postcard_error_t result;
    result = filter.decode_raw(slice);
    if(result != POSTCARD_SUCCESS) return result;

    result = behavior.decode_raw(slice);
    if(result != POSTCARD_SUCCESS) return result;

    result = postcard_decode_u64(slice, &id);
    if(result != POSTCARD_SUCCESS) return result;

    
    return POSTCARD_SUCCESS;
  }
};

struct MsgStream {
  static constexpr std::string_view TYPE_NAME = "MsgStream";

  std::tuple<uint8_t, uint8_t> msg_id;
  


  size_t encoded_size() const {
    size_t size = 0;
    size += postcard_size_u8();size += postcard_size_u8();
    
    return size;
  }

  postcard_error_t encode(std::span<uint8_t>& output) const {
    postcard_slice_t slice;
    postcard_init_slice(&slice, output.data(), output.size());
    auto res = encode_raw(&slice);
    if(res != POSTCARD_SUCCESS) return res;
    output = output.subspan(0, slice.len);
    return POSTCARD_SUCCESS;
  }

  std::vector<uint8_t> encode_vec() const {
    // Pre-allocate vector with the required size
    std::vector<uint8_t> vec(encoded_size());

    // Create a span from the vector
    auto span = std::span<uint8_t>(vec);

    // Encode into the span
    postcard_slice_t slice;
    postcard_init_slice(&slice, span.data(), span.size());
    auto res = encode_raw(&slice);

    // Resize to actual used length if successful
    if (res == POSTCARD_SUCCESS) {
      vec.resize(slice.len);
    } else {
      vec.clear(); // Clear the vector on error
    }

    return vec;
  }

  postcard_error_t encode_raw(postcard_slice_t* slice) const {
    postcard_error_t result;
    
                        {
                            auto val = std::get<0>(msg_id);
                            result = postcard_encode_u8(slice, val);
                        }
                        
                        {
                            auto val = std::get<1>(msg_id);
                            result = postcard_encode_u8(slice, val);
                        }
                        
        if(result != POSTCARD_SUCCESS) return result;
    
    return POSTCARD_SUCCESS;
  }

  postcard_error_t decode(std::span<const uint8_t>& input) {
    postcard_slice_t slice;
    postcard_init_slice(&slice, const_cast<uint8_t*>(input.data()), input.size());
    postcard_error_t result = decode_raw(&slice);
    if (result == POSTCARD_SUCCESS) {
      // Update the input span to point past the decoded data
      input = input.subspan(slice.len);
    }
    return result;
  }

  postcard_error_t decode_raw(postcard_slice_t* slice) {
    postcard_error_t result;
    uint8_t val0;
    result = postcard_decode_u8(slice, &val0);
    uint8_t val1;
    result = postcard_decode_u8(slice, &val1);
    msg_id =  std::tuple<uint8_t, uint8_t>(val0, val1);
    if(result != POSTCARD_SUCCESS) return result;

    
    return POSTCARD_SUCCESS;
  }
};

struct Field {
  static constexpr std::string_view TYPE_NAME = "Field";

  uint16_t offset;
  uint16_t len;
  uint16_t arg;
  


  size_t encoded_size() const {
    size_t size = 0;
    size += postcard_size_u16(offset);
    size += postcard_size_u16(len);
    size += postcard_size_u16(arg);
    
    return size;
  }

  postcard_error_t encode(std::span<uint8_t>& output) const {
    postcard_slice_t slice;
    postcard_init_slice(&slice, output.data(), output.size());
    auto res = encode_raw(&slice);
    if(res != POSTCARD_SUCCESS) return res;
    output = output.subspan(0, slice.len);
    return POSTCARD_SUCCESS;
  }

  std::vector<uint8_t> encode_vec() const {
    // Pre-allocate vector with the required size
    std::vector<uint8_t> vec(encoded_size());

    // Create a span from the vector
    auto span = std::span<uint8_t>(vec);

    // Encode into the span
    postcard_slice_t slice;
    postcard_init_slice(&slice, span.data(), span.size());
    auto res = encode_raw(&slice);

    // Resize to actual used length if successful
    if (res == POSTCARD_SUCCESS) {
      vec.resize(slice.len);
    } else {
      vec.clear(); // Clear the vector on error
    }

    return vec;
  }

  postcard_error_t encode_raw(postcard_slice_t* slice) const {
    postcard_error_t result;
    result = postcard_encode_u16(slice, offset);
        if(result != POSTCARD_SUCCESS) return result;
    result = postcard_encode_u16(slice, len);
        if(result != POSTCARD_SUCCESS) return result;
    result = postcard_encode_u16(slice, arg);
        if(result != POSTCARD_SUCCESS) return result;
    
    return POSTCARD_SUCCESS;
  }

  postcard_error_t decode(std::span<const uint8_t>& input) {
    postcard_slice_t slice;
    postcard_init_slice(&slice, const_cast<uint8_t*>(input.data()), input.size());
    postcard_error_t result = decode_raw(&slice);
    if (result == POSTCARD_SUCCESS) {
      // Update the input span to point past the decoded data
      input = input.subspan(slice.len);
    }
    return result;
  }

  postcard_error_t decode_raw(postcard_slice_t* slice) {
    postcard_error_t result;
    result = postcard_decode_u16(slice, &offset);
    if(result != POSTCARD_SUCCESS) return result;

    result = postcard_decode_u16(slice, &len);
    if(result != POSTCARD_SUCCESS) return result;

    result = postcard_decode_u16(slice, &arg);
    if(result != POSTCARD_SUCCESS) return result;

    
    return POSTCARD_SUCCESS;
  }
};

struct OpData {
  static constexpr std::string_view TYPE_NAME = "OpData";

  uint16_t offset;
  uint16_t len;
  


  size_t encoded_size() const {
    size_t size = 0;
    size += postcard_size_u16(offset);
    size += postcard_size_u16(len);
    
    return size;
  }

  postcard_error_t encode(std::span<uint8_t>& output) const {
    postcard_slice_t slice;
    postcard_init_slice(&slice, output.data(), output.size());
    auto res = encode_raw(&slice);
    if(res != POSTCARD_SUCCESS) return res;
    output = output.subspan(0, slice.len);
    return POSTCARD_SUCCESS;
  }

  std::vector<uint8_t> encode_vec() const {
    // Pre-allocate vector with the required size
    std::vector<uint8_t> vec(encoded_size());

    // Create a span from the vector
    auto span = std::span<uint8_t>(vec);

    // Encode into the span
    postcard_slice_t slice;
    postcard_init_slice(&slice, span.data(), span.size());
    auto res = encode_raw(&slice);

    // Resize to actual used length if successful
    if (res == POSTCARD_SUCCESS) {
      vec.resize(slice.len);
    } else {
      vec.clear(); // Clear the vector on error
    }

    return vec;
  }

  postcard_error_t encode_raw(postcard_slice_t* slice) const {
    postcard_error_t result;
    result = postcard_encode_u16(slice, offset);
        if(result != POSTCARD_SUCCESS) return result;
    result = postcard_encode_u16(slice, len);
        if(result != POSTCARD_SUCCESS) return result;
    
    return POSTCARD_SUCCESS;
  }

  postcard_error_t decode(std::span<const uint8_t>& input) {
    postcard_slice_t slice;
    postcard_init_slice(&slice, const_cast<uint8_t*>(input.data()), input.size());
    postcard_error_t result = decode_raw(&slice);
    if (result == POSTCARD_SUCCESS) {
      // Update the input span to point past the decoded data
      input = input.subspan(slice.len);
    }
    return result;
  }

  postcard_error_t decode_raw(postcard_slice_t* slice) {
    postcard_error_t result;
    result = postcard_decode_u16(slice, &offset);
    if(result != POSTCARD_SUCCESS) return result;

    result = postcard_decode_u16(slice, &len);
    if(result != POSTCARD_SUCCESS) return result;

    
    return POSTCARD_SUCCESS;
  }
};

struct OpTable {
  static constexpr std::string_view TYPE_NAME = "OpTable";

  uint16_t offset;
  uint16_t len;
  


  size_t encoded_size() const {
    size_t size = 0;
    size += postcard_size_u16(offset);
    size += postcard_size_u16(len);
    
    return size;
  }

  postcard_error_t encode(std::span<uint8_t>& output) const {
    postcard_slice_t slice;
    postcard_init_slice(&slice, output.data(), output.size());
    auto res = encode_raw(&slice);
    if(res != POSTCARD_SUCCESS) return res;
    output = output.subspan(0, slice.len);
    return POSTCARD_SUCCESS;
  }

  std::vector<uint8_t> encode_vec() const {
    // Pre-allocate vector with the required size
    std::vector<uint8_t> vec(encoded_size());

    // Create a span from the vector
    auto span = std::span<uint8_t>(vec);

    // Encode into the span
    postcard_slice_t slice;
    postcard_init_slice(&slice, span.data(), span.size());
    auto res = encode_raw(&slice);

    // Resize to actual used length if successful
    if (res == POSTCARD_SUCCESS) {
      vec.resize(slice.len);
    } else {
      vec.clear(); // Clear the vector on error
    }

    return vec;
  }

  postcard_error_t encode_raw(postcard_slice_t* slice) const {
    postcard_error_t result;
    result = postcard_encode_u16(slice, offset);
        if(result != POSTCARD_SUCCESS) return result;
    result = postcard_encode_u16(slice, len);
        if(result != POSTCARD_SUCCESS) return result;
    
    return POSTCARD_SUCCESS;
  }

  postcard_error_t decode(std::span<const uint8_t>& input) {
    postcard_slice_t slice;
    postcard_init_slice(&slice, const_cast<uint8_t*>(input.data()), input.size());
    postcard_error_t result = decode_raw(&slice);
    if (result == POSTCARD_SUCCESS) {
      // Update the input span to point past the decoded data
      input = input.subspan(slice.len);
    }
    return result;
  }

  postcard_error_t decode_raw(postcard_slice_t* slice) {
    postcard_error_t result;
    result = postcard_decode_u16(slice, &offset);
    if(result != POSTCARD_SUCCESS) return result;

    result = postcard_decode_u16(slice, &len);
    if(result != POSTCARD_SUCCESS) return result;

    
    return POSTCARD_SUCCESS;
  }
};

struct OpPair {
  static constexpr std::string_view TYPE_NAME = "OpPair";

  uint16_t entity_id;
  uint16_t component_id;
  


  size_t encoded_size() const {
    size_t size = 0;
    size += postcard_size_u16(entity_id);
    size += postcard_size_u16(component_id);
    
    return size;
  }

  postcard_error_t encode(std::span<uint8_t>& output) const {
    postcard_slice_t slice;
    postcard_init_slice(&slice, output.data(), output.size());
    auto res = encode_raw(&slice);
    if(res != POSTCARD_SUCCESS) return res;
    output = output.subspan(0, slice.len);
    return POSTCARD_SUCCESS;
  }

  std::vector<uint8_t> encode_vec() const {
    // Pre-allocate vector with the required size
    std::vector<uint8_t> vec(encoded_size());

    // Create a span from the vector
    auto span = std::span<uint8_t>(vec);

    // Encode into the span
    postcard_slice_t slice;
    postcard_init_slice(&slice, span.data(), span.size());
    auto res = encode_raw(&slice);

    // Resize to actual used length if successful
    if (res == POSTCARD_SUCCESS) {
      vec.resize(slice.len);
    } else {
      vec.clear(); // Clear the vector on error
    }

    return vec;
  }

  postcard_error_t encode_raw(postcard_slice_t* slice) const {
    postcard_error_t result;
    result = postcard_encode_u16(slice, entity_id);
        if(result != POSTCARD_SUCCESS) return result;
    result = postcard_encode_u16(slice, component_id);
        if(result != POSTCARD_SUCCESS) return result;
    
    return POSTCARD_SUCCESS;
  }

  postcard_error_t decode(std::span<const uint8_t>& input) {
    postcard_slice_t slice;
    postcard_init_slice(&slice, const_cast<uint8_t*>(input.data()), input.size());
    postcard_error_t result = decode_raw(&slice);
    if (result == POSTCARD_SUCCESS) {
      // Update the input span to point past the decoded data
      input = input.subspan(slice.len);
    }
    return result;
  }

  postcard_error_t decode_raw(postcard_slice_t* slice) {
    postcard_error_t result;
    result = postcard_decode_u16(slice, &entity_id);
    if(result != POSTCARD_SUCCESS) return result;

    result = postcard_decode_u16(slice, &component_id);
    if(result != POSTCARD_SUCCESS) return result;

    
    return POSTCARD_SUCCESS;
  }
};

struct OpSchema {
  static constexpr std::string_view TYPE_NAME = "OpSchema";

  uint16_t ty;
  uint16_t dim;
  uint16_t arg;
  


  size_t encoded_size() const {
    size_t size = 0;
    size += postcard_size_u16(ty);
    size += postcard_size_u16(dim);
    size += postcard_size_u16(arg);
    
    return size;
  }

  postcard_error_t encode(std::span<uint8_t>& output) const {
    postcard_slice_t slice;
    postcard_init_slice(&slice, output.data(), output.size());
    auto res = encode_raw(&slice);
    if(res != POSTCARD_SUCCESS) return res;
    output = output.subspan(0, slice.len);
    return POSTCARD_SUCCESS;
  }

  std::vector<uint8_t> encode_vec() const {
    // Pre-allocate vector with the required size
    std::vector<uint8_t> vec(encoded_size());

    // Create a span from the vector
    auto span = std::span<uint8_t>(vec);

    // Encode into the span
    postcard_slice_t slice;
    postcard_init_slice(&slice, span.data(), span.size());
    auto res = encode_raw(&slice);

    // Resize to actual used length if successful
    if (res == POSTCARD_SUCCESS) {
      vec.resize(slice.len);
    } else {
      vec.clear(); // Clear the vector on error
    }

    return vec;
  }

  postcard_error_t encode_raw(postcard_slice_t* slice) const {
    postcard_error_t result;
    result = postcard_encode_u16(slice, ty);
        if(result != POSTCARD_SUCCESS) return result;
    result = postcard_encode_u16(slice, dim);
        if(result != POSTCARD_SUCCESS) return result;
    result = postcard_encode_u16(slice, arg);
        if(result != POSTCARD_SUCCESS) return result;
    
    return POSTCARD_SUCCESS;
  }

  postcard_error_t decode(std::span<const uint8_t>& input) {
    postcard_slice_t slice;
    postcard_init_slice(&slice, const_cast<uint8_t*>(input.data()), input.size());
    postcard_error_t result = decode_raw(&slice);
    if (result == POSTCARD_SUCCESS) {
      // Update the input span to point past the decoded data
      input = input.subspan(slice.len);
    }
    return result;
  }

  postcard_error_t decode_raw(postcard_slice_t* slice) {
    postcard_error_t result;
    result = postcard_decode_u16(slice, &ty);
    if(result != POSTCARD_SUCCESS) return result;

    result = postcard_decode_u16(slice, &dim);
    if(result != POSTCARD_SUCCESS) return result;

    result = postcard_decode_u16(slice, &arg);
    if(result != POSTCARD_SUCCESS) return result;

    
    return POSTCARD_SUCCESS;
  }
};

struct OpTimestamp {
  static constexpr std::string_view TYPE_NAME = "OpTimestamp";

  uint16_t source;
  uint16_t arg;
  


  size_t encoded_size() const {
    size_t size = 0;
    size += postcard_size_u16(source);
    size += postcard_size_u16(arg);
    
    return size;
  }

  postcard_error_t encode(std::span<uint8_t>& output) const {
    postcard_slice_t slice;
    postcard_init_slice(&slice, output.data(), output.size());
    auto res = encode_raw(&slice);
    if(res != POSTCARD_SUCCESS) return res;
    output = output.subspan(0, slice.len);
    return POSTCARD_SUCCESS;
  }

  std::vector<uint8_t> encode_vec() const {
    // Pre-allocate vector with the required size
    std::vector<uint8_t> vec(encoded_size());

    // Create a span from the vector
    auto span = std::span<uint8_t>(vec);

    // Encode into the span
    postcard_slice_t slice;
    postcard_init_slice(&slice, span.data(), span.size());
    auto res = encode_raw(&slice);

    // Resize to actual used length if successful
    if (res == POSTCARD_SUCCESS) {
      vec.resize(slice.len);
    } else {
      vec.clear(); // Clear the vector on error
    }

    return vec;
  }

  postcard_error_t encode_raw(postcard_slice_t* slice) const {
    postcard_error_t result;
    result = postcard_encode_u16(slice, source);
        if(result != POSTCARD_SUCCESS) return result;
    result = postcard_encode_u16(slice, arg);
        if(result != POSTCARD_SUCCESS) return result;
    
    return POSTCARD_SUCCESS;
  }

  postcard_error_t decode(std::span<const uint8_t>& input) {
    postcard_slice_t slice;
    postcard_init_slice(&slice, const_cast<uint8_t*>(input.data()), input.size());
    postcard_error_t result = decode_raw(&slice);
    if (result == POSTCARD_SUCCESS) {
      // Update the input span to point past the decoded data
      input = input.subspan(slice.len);
    }
    return result;
  }

  postcard_error_t decode_raw(postcard_slice_t* slice) {
    postcard_error_t result;
    result = postcard_decode_u16(slice, &source);
    if(result != POSTCARD_SUCCESS) return result;

    result = postcard_decode_u16(slice, &arg);
    if(result != POSTCARD_SUCCESS) return result;

    
    return POSTCARD_SUCCESS;
  }
};

struct OpExt {
  static constexpr std::string_view TYPE_NAME = "OpExt";

  uint16_t arg;
  std::tuple<uint8_t, uint8_t> id;
  uint16_t data;
  


  size_t encoded_size() const {
    size_t size = 0;
    size += postcard_size_u16(arg);
    size += postcard_size_u8();size += postcard_size_u8();
    size += postcard_size_u16(data);
    
    return size;
  }

  postcard_error_t encode(std::span<uint8_t>& output) const {
    postcard_slice_t slice;
    postcard_init_slice(&slice, output.data(), output.size());
    auto res = encode_raw(&slice);
    if(res != POSTCARD_SUCCESS) return res;
    output = output.subspan(0, slice.len);
    return POSTCARD_SUCCESS;
  }

  std::vector<uint8_t> encode_vec() const {
    // Pre-allocate vector with the required size
    std::vector<uint8_t> vec(encoded_size());

    // Create a span from the vector
    auto span = std::span<uint8_t>(vec);

    // Encode into the span
    postcard_slice_t slice;
    postcard_init_slice(&slice, span.data(), span.size());
    auto res = encode_raw(&slice);

    // Resize to actual used length if successful
    if (res == POSTCARD_SUCCESS) {
      vec.resize(slice.len);
    } else {
      vec.clear(); // Clear the vector on error
    }

    return vec;
  }

  postcard_error_t encode_raw(postcard_slice_t* slice) const {
    postcard_error_t result;
    result = postcard_encode_u16(slice, arg);
        if(result != POSTCARD_SUCCESS) return result;
    
                        {
                            auto val = std::get<0>(id);
                            result = postcard_encode_u8(slice, val);
                        }
                        
                        {
                            auto val = std::get<1>(id);
                            result = postcard_encode_u8(slice, val);
                        }
                        
        if(result != POSTCARD_SUCCESS) return result;
    result = postcard_encode_u16(slice, data);
        if(result != POSTCARD_SUCCESS) return result;
    
    return POSTCARD_SUCCESS;
  }

  postcard_error_t decode(std::span<const uint8_t>& input) {
    postcard_slice_t slice;
    postcard_init_slice(&slice, const_cast<uint8_t*>(input.data()), input.size());
    postcard_error_t result = decode_raw(&slice);
    if (result == POSTCARD_SUCCESS) {
      // Update the input span to point past the decoded data
      input = input.subspan(slice.len);
    }
    return result;
  }

  postcard_error_t decode_raw(postcard_slice_t* slice) {
    postcard_error_t result;
    result = postcard_decode_u16(slice, &arg);
    if(result != POSTCARD_SUCCESS) return result;

    uint8_t val0;
    result = postcard_decode_u8(slice, &val0);
    uint8_t val1;
    result = postcard_decode_u8(slice, &val1);
    id =  std::tuple<uint8_t, uint8_t>(val0, val1);
    if(result != POSTCARD_SUCCESS) return result;

    result = postcard_decode_u16(slice, &data);
    if(result != POSTCARD_SUCCESS) return result;

    
    return POSTCARD_SUCCESS;
  }
};

class Op : public std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt> {
public:
  static constexpr std::string_view TYPE_NAME = "Op";
  // Inherit constructors from std::variant
  using std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>::variant;

  
  // Static constructor for Data variant
  static Op Data(const OpData& value) {
    return Op{std::in_place_index<0>, value};
  }
  

  // Accessor method for variant Data
  bool is_data() const {
    return this->index() == 0;
  }
  
  const OpData* get_data() const {
    return std::get_if<0>((const std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this);
  }

  OpData* get_data() {
    return std::get_if<0>((std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this);
  }
  

  
  // Static constructor for Table variant
  static Op Table(const OpTable& value) {
    return Op{std::in_place_index<1>, value};
  }
  

  // Accessor method for variant Table
  bool is_table() const {
    return this->index() == 1;
  }
  
  const OpTable* get_table() const {
    return std::get_if<1>((const std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this);
  }

  OpTable* get_table() {
    return std::get_if<1>((std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this);
  }
  

  
  // Static constructor for unit variant None
  static Op None() {
    return Op{std::in_place_index<2>, std::monostate{}};
  }
  

  // Accessor method for variant None
  bool is_none() const {
    return this->index() == 2;
  }
  
  const std::monostate* get_none() const {
    return std::get_if<2>((const std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this);
  }

  std::monostate* get_none() {
    return std::get_if<2>((std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this);
  }
  

  
  // Static constructor for Pair variant
  static Op Pair(const OpPair& value) {
    return Op{std::in_place_index<3>, value};
  }
  

  // Accessor method for variant Pair
  bool is_pair() const {
    return this->index() == 3;
  }
  
  const OpPair* get_pair() const {
    return std::get_if<3>((const std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this);
  }

  OpPair* get_pair() {
    return std::get_if<3>((std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this);
  }
  

  
  // Static constructor for Schema variant
  static Op Schema(const OpSchema& value) {
    return Op{std::in_place_index<4>, value};
  }
  

  // Accessor method for variant Schema
  bool is_schema() const {
    return this->index() == 4;
  }
  
  const OpSchema* get_schema() const {
    return std::get_if<4>((const std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this);
  }

  OpSchema* get_schema() {
    return std::get_if<4>((std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this);
  }
  

  
  // Static constructor for Timestamp variant
  static Op Timestamp(const OpTimestamp& value) {
    return Op{std::in_place_index<5>, value};
  }
  

  // Accessor method for variant Timestamp
  bool is_timestamp() const {
    return this->index() == 5;
  }
  
  const OpTimestamp* get_timestamp() const {
    return std::get_if<5>((const std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this);
  }

  OpTimestamp* get_timestamp() {
    return std::get_if<5>((std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this);
  }
  

  
  // Static constructor for Ext variant
  static Op Ext(const OpExt& value) {
    return Op{std::in_place_index<6>, value};
  }
  

  // Accessor method for variant Ext
  bool is_ext() const {
    return this->index() == 6;
  }
  
  const OpExt* get_ext() const {
    return std::get_if<6>((const std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this);
  }

  OpExt* get_ext() {
    return std::get_if<6>((std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this);
  }
  

  

  size_t encoded_size() const {
    size_t size = 0;

    // Tag size (discriminant)
    size += postcard_size_u8(); // Just for the variant tag

    if ([[maybe_unused]] auto val = std::get_if<0>((const std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this)) {
        size += (*val).encoded_size();
    }
    else if ([[maybe_unused]] auto val = std::get_if<1>((const std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this)) {
        size += (*val).encoded_size();
    }
    else if ([[maybe_unused]] auto val = std::get_if<2>((const std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this)) {
        
    }
    else if ([[maybe_unused]] auto val = std::get_if<3>((const std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this)) {
        size += (*val).encoded_size();
    }
    else if ([[maybe_unused]] auto val = std::get_if<4>((const std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this)) {
        size += (*val).encoded_size();
    }
    else if ([[maybe_unused]] auto val = std::get_if<5>((const std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this)) {
        size += (*val).encoded_size();
    }
    else if ([[maybe_unused]] auto val = std::get_if<6>((const std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this)) {
        size += (*val).encoded_size();
    }
    

    return size;
  }

  postcard_error_t encode(std::span<uint8_t>& output) const {
    postcard_slice_t slice;
    postcard_init_slice(&slice, output.data(), output.size());
    auto res = encode_raw(&slice);
    if(res != POSTCARD_SUCCESS) return res;
    output = output.subspan(0, slice.len);
    return POSTCARD_SUCCESS;
  }

  std::vector<uint8_t> encode_vec() const {
    // Pre-allocate vector with the required size
    std::vector<uint8_t> vec(encoded_size());

    // Create a span from the vector
    auto span = std::span<uint8_t>(vec);

    // Encode into the span
    postcard_slice_t slice;
    postcard_init_slice(&slice, span.data(), span.size());
    auto res = encode_raw(&slice);

    // Resize to actual used length if successful
    if (res == POSTCARD_SUCCESS) {
      vec.resize(slice.len);
    } else {
      vec.clear(); // Clear the vector on error
    }

    return vec;
  }

  postcard_error_t encode_raw(postcard_slice_t* slice) const {
    postcard_error_t result;

    if ([[maybe_unused]] auto val = std::get_if<0>((std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this)) {
        result = postcard_encode_u8(slice, 0);
        if (result != POSTCARD_SUCCESS) return result;
        result = (*val).encode_raw(slice);
        if (result != POSTCARD_SUCCESS) return result;
    }
    else if ([[maybe_unused]] auto val = std::get_if<1>((std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this)) {
        result = postcard_encode_u8(slice, 1);
        if (result != POSTCARD_SUCCESS) return result;
        result = (*val).encode_raw(slice);
        if (result != POSTCARD_SUCCESS) return result;
    }
    else if ([[maybe_unused]] auto val = std::get_if<2>((std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this)) {
        result = postcard_encode_u8(slice, 2);
        if (result != POSTCARD_SUCCESS) return result;
        
        if (result != POSTCARD_SUCCESS) return result;
    }
    else if ([[maybe_unused]] auto val = std::get_if<3>((std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this)) {
        result = postcard_encode_u8(slice, 3);
        if (result != POSTCARD_SUCCESS) return result;
        result = (*val).encode_raw(slice);
        if (result != POSTCARD_SUCCESS) return result;
    }
    else if ([[maybe_unused]] auto val = std::get_if<4>((std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this)) {
        result = postcard_encode_u8(slice, 4);
        if (result != POSTCARD_SUCCESS) return result;
        result = (*val).encode_raw(slice);
        if (result != POSTCARD_SUCCESS) return result;
    }
    else if ([[maybe_unused]] auto val = std::get_if<5>((std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this)) {
        result = postcard_encode_u8(slice, 5);
        if (result != POSTCARD_SUCCESS) return result;
        result = (*val).encode_raw(slice);
        if (result != POSTCARD_SUCCESS) return result;
    }
    else if ([[maybe_unused]] auto val = std::get_if<6>((std::variant<OpData, OpTable, std::monostate, OpPair, OpSchema, OpTimestamp, OpExt>*)this)) {
        result = postcard_encode_u8(slice, 6);
        if (result != POSTCARD_SUCCESS) return result;
        result = (*val).encode_raw(slice);
        if (result != POSTCARD_SUCCESS) return result;
    }
    
    else {
        return POSTCARD_ERROR_INVALID_INPUT;
    }

    return POSTCARD_SUCCESS;
  }

  postcard_error_t decode(std::span<const uint8_t>& input) {
    postcard_slice_t slice;
    postcard_init_slice(&slice, const_cast<uint8_t*>(input.data()), input.size());
    postcard_error_t result = decode_raw(&slice);
    if (result == POSTCARD_SUCCESS) {
      // Update the input span to point past the decoded data
      input = input.subspan(slice.len);
    }
    return result;
  }

  postcard_error_t decode_raw(postcard_slice_t* slice) {
    postcard_error_t result;
    uint8_t tag;
    result = postcard_decode_u8(slice, &tag);
    if (result != POSTCARD_SUCCESS) return result;

    switch(tag) {
        case 0: {  // Data
            
            OpData val;
            result = val.decode_raw(slice);
            if (result != POSTCARD_SUCCESS) return result;
            this->emplace<0>(val);
            
            break;
        }
        case 1: {  // Table
            
            OpTable val;
            result = val.decode_raw(slice);
            if (result != POSTCARD_SUCCESS) return result;
            this->emplace<1>(val);
            
            break;
        }
        case 2: {  // None
            
            this->emplace<2>(std::monostate{});
            
            break;
        }
        case 3: {  // Pair
            
            OpPair val;
            result = val.decode_raw(slice);
            if (result != POSTCARD_SUCCESS) return result;
            this->emplace<3>(val);
            
            break;
        }
        case 4: {  // Schema
            
            OpSchema val;
            result = val.decode_raw(slice);
            if (result != POSTCARD_SUCCESS) return result;
            this->emplace<4>(val);
            
            break;
        }
        case 5: {  // Timestamp
            
            OpTimestamp val;
            result = val.decode_raw(slice);
            if (result != POSTCARD_SUCCESS) return result;
            this->emplace<5>(val);
            
            break;
        }
        case 6: {  // Ext
            
            OpExt val;
            result = val.decode_raw(slice);
            if (result != POSTCARD_SUCCESS) return result;
            this->emplace<6>(val);
            
            break;
        }
        
        default:
            return POSTCARD_ERROR_INVALID_INPUT;
    }

    return POSTCARD_SUCCESS;
  }
};

struct OpRef {
  static constexpr std::string_view TYPE_NAME = "OpRef";

  uint16_t value;

  size_t encoded_size() const {
    size_t size = 0;
    size += postcard_size_u16(value);
    return size;
  }

  postcard_error_t encode(std::span<std::byte>& output) const {
    postcard_slice_t slice;
    postcard_init_slice(&slice, reinterpret_cast<uint8_t*>(output.data()), output.size());
    auto res = encode_raw(&slice);
    if(res != POSTCARD_SUCCESS) return res;
    output = output.subspan(0, slice.len);
    return POSTCARD_SUCCESS;
  }

  std::vector<std::byte> encode_vec() const {
    // Pre-allocate vector with the required size
    std::vector<std::byte> vec(encoded_size());

    // Create a span from the vector
    auto span = std::span<std::byte>(vec);

    // Encode into the span
    postcard_slice_t slice;
    postcard_init_slice(&slice, reinterpret_cast<uint8_t*>(span.data()), span.size());
    auto res = encode_raw(&slice);

    // Resize to actual used length if successful
    if (res == POSTCARD_SUCCESS) {
      vec.resize(slice.len);
    } else {
      vec.clear(); // Clear the vector on error
    }

    return vec;
  }

  postcard_error_t encode_raw(postcard_slice_t* slice) const {
    postcard_error_t result;
    result = postcard_encode_u16(slice, value);
    return result;
  }

  postcard_error_t decode(std::span<const std::byte>& input) {
    postcard_slice_t slice;
    postcard_init_slice(&slice, const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(input.data())), input.size());
    postcard_error_t result = decode_raw(&slice);
    if (result == POSTCARD_SUCCESS) {
      // Update the input span to point past the decoded data
      input = input.subspan(slice.len);
    }
    return result;
  }

  postcard_error_t decode_raw(postcard_slice_t* slice) {
    postcard_error_t result;
    result = postcard_decode_u16(slice, &value);
    return result;
  }
};

class PrimType : public std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate> {
public:
  static constexpr std::string_view TYPE_NAME = "PrimType";
  // Inherit constructors from std::variant
  using std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>::variant;

  
  // Static constructor for unit variant U8
  static PrimType U8() {
    return PrimType{std::in_place_index<0>, std::monostate{}};
  }
  

  // Accessor method for variant U8
  bool is_u_8() const {
    return this->index() == 0;
  }
  
  const std::monostate* get_u_8() const {
    return std::get_if<0>((const std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this);
  }

  std::monostate* get_u_8() {
    return std::get_if<0>((std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this);
  }
  

  
  // Static constructor for unit variant U16
  static PrimType U16() {
    return PrimType{std::in_place_index<1>, std::monostate{}};
  }
  

  // Accessor method for variant U16
  bool is_u_16() const {
    return this->index() == 1;
  }
  
  const std::monostate* get_u_16() const {
    return std::get_if<1>((const std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this);
  }

  std::monostate* get_u_16() {
    return std::get_if<1>((std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this);
  }
  

  
  // Static constructor for unit variant U32
  static PrimType U32() {
    return PrimType{std::in_place_index<2>, std::monostate{}};
  }
  

  // Accessor method for variant U32
  bool is_u_32() const {
    return this->index() == 2;
  }
  
  const std::monostate* get_u_32() const {
    return std::get_if<2>((const std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this);
  }

  std::monostate* get_u_32() {
    return std::get_if<2>((std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this);
  }
  

  
  // Static constructor for unit variant U64
  static PrimType U64() {
    return PrimType{std::in_place_index<3>, std::monostate{}};
  }
  

  // Accessor method for variant U64
  bool is_u_64() const {
    return this->index() == 3;
  }
  
  const std::monostate* get_u_64() const {
    return std::get_if<3>((const std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this);
  }

  std::monostate* get_u_64() {
    return std::get_if<3>((std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this);
  }
  

  
  // Static constructor for unit variant I8
  static PrimType I8() {
    return PrimType{std::in_place_index<4>, std::monostate{}};
  }
  

  // Accessor method for variant I8
  bool is_i_8() const {
    return this->index() == 4;
  }
  
  const std::monostate* get_i_8() const {
    return std::get_if<4>((const std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this);
  }

  std::monostate* get_i_8() {
    return std::get_if<4>((std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this);
  }
  

  
  // Static constructor for unit variant I16
  static PrimType I16() {
    return PrimType{std::in_place_index<5>, std::monostate{}};
  }
  

  // Accessor method for variant I16
  bool is_i_16() const {
    return this->index() == 5;
  }
  
  const std::monostate* get_i_16() const {
    return std::get_if<5>((const std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this);
  }

  std::monostate* get_i_16() {
    return std::get_if<5>((std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this);
  }
  

  
  // Static constructor for unit variant I32
  static PrimType I32() {
    return PrimType{std::in_place_index<6>, std::monostate{}};
  }
  

  // Accessor method for variant I32
  bool is_i_32() const {
    return this->index() == 6;
  }
  
  const std::monostate* get_i_32() const {
    return std::get_if<6>((const std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this);
  }

  std::monostate* get_i_32() {
    return std::get_if<6>((std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this);
  }
  

  
  // Static constructor for unit variant I64
  static PrimType I64() {
    return PrimType{std::in_place_index<7>, std::monostate{}};
  }
  

  // Accessor method for variant I64
  bool is_i_64() const {
    return this->index() == 7;
  }
  
  const std::monostate* get_i_64() const {
    return std::get_if<7>((const std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this);
  }

  std::monostate* get_i_64() {
    return std::get_if<7>((std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this);
  }
  

  
  // Static constructor for unit variant Bool
  static PrimType Bool() {
    return PrimType{std::in_place_index<8>, std::monostate{}};
  }
  

  // Accessor method for variant Bool
  bool is_bool() const {
    return this->index() == 8;
  }
  
  const std::monostate* get_bool() const {
    return std::get_if<8>((const std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this);
  }

  std::monostate* get_bool() {
    return std::get_if<8>((std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this);
  }
  

  
  // Static constructor for unit variant F32
  static PrimType F32() {
    return PrimType{std::in_place_index<9>, std::monostate{}};
  }
  

  // Accessor method for variant F32
  bool is_f_32() const {
    return this->index() == 9;
  }
  
  const std::monostate* get_f_32() const {
    return std::get_if<9>((const std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this);
  }

  std::monostate* get_f_32() {
    return std::get_if<9>((std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this);
  }
  

  
  // Static constructor for unit variant F64
  static PrimType F64() {
    return PrimType{std::in_place_index<10>, std::monostate{}};
  }
  

  // Accessor method for variant F64
  bool is_f_64() const {
    return this->index() == 10;
  }
  
  const std::monostate* get_f_64() const {
    return std::get_if<10>((const std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this);
  }

  std::monostate* get_f_64() {
    return std::get_if<10>((std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this);
  }
  

  

  size_t encoded_size() const {
    size_t size = 0;

    // Tag size (discriminant)
    size += postcard_size_u8(); // Just for the variant tag

    if ([[maybe_unused]] auto val = std::get_if<0>((const std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this)) {
        
    }
    else if ([[maybe_unused]] auto val = std::get_if<1>((const std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this)) {
        
    }
    else if ([[maybe_unused]] auto val = std::get_if<2>((const std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this)) {
        
    }
    else if ([[maybe_unused]] auto val = std::get_if<3>((const std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this)) {
        
    }
    else if ([[maybe_unused]] auto val = std::get_if<4>((const std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this)) {
        
    }
    else if ([[maybe_unused]] auto val = std::get_if<5>((const std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this)) {
        
    }
    else if ([[maybe_unused]] auto val = std::get_if<6>((const std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this)) {
        
    }
    else if ([[maybe_unused]] auto val = std::get_if<7>((const std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this)) {
        
    }
    else if ([[maybe_unused]] auto val = std::get_if<8>((const std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this)) {
        
    }
    else if ([[maybe_unused]] auto val = std::get_if<9>((const std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this)) {
        
    }
    else if ([[maybe_unused]] auto val = std::get_if<10>((const std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this)) {
        
    }
    

    return size;
  }

  postcard_error_t encode(std::span<uint8_t>& output) const {
    postcard_slice_t slice;
    postcard_init_slice(&slice, output.data(), output.size());
    auto res = encode_raw(&slice);
    if(res != POSTCARD_SUCCESS) return res;
    output = output.subspan(0, slice.len);
    return POSTCARD_SUCCESS;
  }

  std::vector<uint8_t> encode_vec() const {
    // Pre-allocate vector with the required size
    std::vector<uint8_t> vec(encoded_size());

    // Create a span from the vector
    auto span = std::span<uint8_t>(vec);

    // Encode into the span
    postcard_slice_t slice;
    postcard_init_slice(&slice, span.data(), span.size());
    auto res = encode_raw(&slice);

    // Resize to actual used length if successful
    if (res == POSTCARD_SUCCESS) {
      vec.resize(slice.len);
    } else {
      vec.clear(); // Clear the vector on error
    }

    return vec;
  }

  postcard_error_t encode_raw(postcard_slice_t* slice) const {
    postcard_error_t result;

    if ([[maybe_unused]] auto val = std::get_if<0>((std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this)) {
        result = postcard_encode_u8(slice, 0);
        if (result != POSTCARD_SUCCESS) return result;
        
        if (result != POSTCARD_SUCCESS) return result;
    }
    else if ([[maybe_unused]] auto val = std::get_if<1>((std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this)) {
        result = postcard_encode_u8(slice, 1);
        if (result != POSTCARD_SUCCESS) return result;
        
        if (result != POSTCARD_SUCCESS) return result;
    }
    else if ([[maybe_unused]] auto val = std::get_if<2>((std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this)) {
        result = postcard_encode_u8(slice, 2);
        if (result != POSTCARD_SUCCESS) return result;
        
        if (result != POSTCARD_SUCCESS) return result;
    }
    else if ([[maybe_unused]] auto val = std::get_if<3>((std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this)) {
        result = postcard_encode_u8(slice, 3);
        if (result != POSTCARD_SUCCESS) return result;
        
        if (result != POSTCARD_SUCCESS) return result;
    }
    else if ([[maybe_unused]] auto val = std::get_if<4>((std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this)) {
        result = postcard_encode_u8(slice, 4);
        if (result != POSTCARD_SUCCESS) return result;
        
        if (result != POSTCARD_SUCCESS) return result;
    }
    else if ([[maybe_unused]] auto val = std::get_if<5>((std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this)) {
        result = postcard_encode_u8(slice, 5);
        if (result != POSTCARD_SUCCESS) return result;
        
        if (result != POSTCARD_SUCCESS) return result;
    }
    else if ([[maybe_unused]] auto val = std::get_if<6>((std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this)) {
        result = postcard_encode_u8(slice, 6);
        if (result != POSTCARD_SUCCESS) return result;
        
        if (result != POSTCARD_SUCCESS) return result;
    }
    else if ([[maybe_unused]] auto val = std::get_if<7>((std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this)) {
        result = postcard_encode_u8(slice, 7);
        if (result != POSTCARD_SUCCESS) return result;
        
        if (result != POSTCARD_SUCCESS) return result;
    }
    else if ([[maybe_unused]] auto val = std::get_if<8>((std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this)) {
        result = postcard_encode_u8(slice, 8);
        if (result != POSTCARD_SUCCESS) return result;
        
        if (result != POSTCARD_SUCCESS) return result;
    }
    else if ([[maybe_unused]] auto val = std::get_if<9>((std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this)) {
        result = postcard_encode_u8(slice, 9);
        if (result != POSTCARD_SUCCESS) return result;
        
        if (result != POSTCARD_SUCCESS) return result;
    }
    else if ([[maybe_unused]] auto val = std::get_if<10>((std::variant<std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate, std::monostate>*)this)) {
        result = postcard_encode_u8(slice, 10);
        if (result != POSTCARD_SUCCESS) return result;
        
        if (result != POSTCARD_SUCCESS) return result;
    }
    
    else {
        return POSTCARD_ERROR_INVALID_INPUT;
    }

    return POSTCARD_SUCCESS;
  }

  postcard_error_t decode(std::span<const uint8_t>& input) {
    postcard_slice_t slice;
    postcard_init_slice(&slice, const_cast<uint8_t*>(input.data()), input.size());
    postcard_error_t result = decode_raw(&slice);
    if (result == POSTCARD_SUCCESS) {
      // Update the input span to point past the decoded data
      input = input.subspan(slice.len);
    }
    return result;
  }

  postcard_error_t decode_raw(postcard_slice_t* slice) {
    postcard_error_t result;
    uint8_t tag;
    result = postcard_decode_u8(slice, &tag);
    if (result != POSTCARD_SUCCESS) return result;

    switch(tag) {
        case 0: {  // U8
            
            this->emplace<0>(std::monostate{});
            
            break;
        }
        case 1: {  // U16
            
            this->emplace<1>(std::monostate{});
            
            break;
        }
        case 2: {  // U32
            
            this->emplace<2>(std::monostate{});
            
            break;
        }
        case 3: {  // U64
            
            this->emplace<3>(std::monostate{});
            
            break;
        }
        case 4: {  // I8
            
            this->emplace<4>(std::monostate{});
            
            break;
        }
        case 5: {  // I16
            
            this->emplace<5>(std::monostate{});
            
            break;
        }
        case 6: {  // I32
            
            this->emplace<6>(std::monostate{});
            
            break;
        }
        case 7: {  // I64
            
            this->emplace<7>(std::monostate{});
            
            break;
        }
        case 8: {  // Bool
            
            this->emplace<8>(std::monostate{});
            
            break;
        }
        case 9: {  // F32
            
            this->emplace<9>(std::monostate{});
            
            break;
        }
        case 10: {  // F64
            
            this->emplace<10>(std::monostate{});
            
            break;
        }
        
        default:
            return POSTCARD_ERROR_INVALID_INPUT;
    }

    return POSTCARD_SUCCESS;
  }
};

struct VTable {
  static constexpr std::string_view TYPE_NAME = "VTable";

  std::vector<Op> ops;
  std::vector<Field> fields;
  std::vector<uint8_t> data;
  


  size_t encoded_size() const {
    size_t size = 0;
    size += postcard_size_seq(ops.size());
    for([[maybe_unused]] const auto& val: ops) {
      size += val.encoded_size();
    }
    size += postcard_size_seq(fields.size());
    for([[maybe_unused]] const auto& val: fields) {
      size += val.encoded_size();
    }
    size += postcard_size_seq(data.size());
    for([[maybe_unused]] const auto& val: data) {
      size += postcard_size_u8();
    }
    
    return size;
  }

  postcard_error_t encode(std::span<uint8_t>& output) const {
    postcard_slice_t slice;
    postcard_init_slice(&slice, output.data(), output.size());
    auto res = encode_raw(&slice);
    if(res != POSTCARD_SUCCESS) return res;
    output = output.subspan(0, slice.len);
    return POSTCARD_SUCCESS;
  }

  std::vector<uint8_t> encode_vec() const {
    // Pre-allocate vector with the required size
    std::vector<uint8_t> vec(encoded_size());

    // Create a span from the vector
    auto span = std::span<uint8_t>(vec);

    // Encode into the span
    postcard_slice_t slice;
    postcard_init_slice(&slice, span.data(), span.size());
    auto res = encode_raw(&slice);

    // Resize to actual used length if successful
    if (res == POSTCARD_SUCCESS) {
      vec.resize(slice.len);
    } else {
      vec.clear(); // Clear the vector on error
    }

    return vec;
  }

  postcard_error_t encode_raw(postcard_slice_t* slice) const {
    postcard_error_t result;
    result = postcard_start_seq(slice, ops.size());
    for(const auto& val: ops) {
      result = val.encode_raw(slice);
      if(result != POSTCARD_SUCCESS) return result;
    }
        if(result != POSTCARD_SUCCESS) return result;
    result = postcard_start_seq(slice, fields.size());
    for(const auto& val: fields) {
      result = val.encode_raw(slice);
      if(result != POSTCARD_SUCCESS) return result;
    }
        if(result != POSTCARD_SUCCESS) return result;
    result = postcard_start_seq(slice, data.size());
    for(const auto& val: data) {
      result = postcard_encode_u8(slice, val);
      if(result != POSTCARD_SUCCESS) return result;
    }
        if(result != POSTCARD_SUCCESS) return result;
    
    return POSTCARD_SUCCESS;
  }

  postcard_error_t decode(std::span<const uint8_t>& input) {
    postcard_slice_t slice;
    postcard_init_slice(&slice, const_cast<uint8_t*>(input.data()), input.size());
    postcard_error_t result = decode_raw(&slice);
    if (result == POSTCARD_SUCCESS) {
      // Update the input span to point past the decoded data
      input = input.subspan(slice.len);
    }
    return result;
  }

  postcard_error_t decode_raw(postcard_slice_t* slice) {
    postcard_error_t result;
    size_t ops_len;
    result = postcard_decode_seq_len(slice, &ops_len);
    if (result != POSTCARD_SUCCESS) return result;
    ops.clear();
    ops.reserve(ops_len);
    for(size_t i = 0; i < ops_len; i++) {
        Op val;
        result = val.decode_raw(slice);
        ops.push_back(val);
    }
    if(result != POSTCARD_SUCCESS) return result;

    size_t fields_len;
    result = postcard_decode_seq_len(slice, &fields_len);
    if (result != POSTCARD_SUCCESS) return result;
    fields.clear();
    fields.reserve(fields_len);
    for(size_t i = 0; i < fields_len; i++) {
        Field val;
        result = val.decode_raw(slice);
        fields.push_back(val);
    }
    if(result != POSTCARD_SUCCESS) return result;

    size_t data_len;
    result = postcard_decode_seq_len(slice, &data_len);
    if (result != POSTCARD_SUCCESS) return result;
    data.clear();
    data.reserve(data_len);
    for(size_t i = 0; i < data_len; i++) {
        uint8_t val;
        result = postcard_decode_u8(slice, &val);
        data.push_back(val);
    }
    if(result != POSTCARD_SUCCESS) return result;

    
    return POSTCARD_SUCCESS;
  }
};

struct VTableMsg {
  static constexpr std::string_view TYPE_NAME = "VTableMsg";

  std::tuple<uint8_t, uint8_t> id;
  VTable vtable;
  


  size_t encoded_size() const {
    size_t size = 0;
    size += postcard_size_u8();size += postcard_size_u8();
    size += vtable.encoded_size();
    
    return size;
  }

  postcard_error_t encode(std::span<uint8_t>& output) const {
    postcard_slice_t slice;
    postcard_init_slice(&slice, output.data(), output.size());
    auto res = encode_raw(&slice);
    if(res != POSTCARD_SUCCESS) return res;
    output = output.subspan(0, slice.len);
    return POSTCARD_SUCCESS;
  }

  std::vector<uint8_t> encode_vec() const {
    // Pre-allocate vector with the required size
    std::vector<uint8_t> vec(encoded_size());

    // Create a span from the vector
    auto span = std::span<uint8_t>(vec);

    // Encode into the span
    postcard_slice_t slice;
    postcard_init_slice(&slice, span.data(), span.size());
    auto res = encode_raw(&slice);

    // Resize to actual used length if successful
    if (res == POSTCARD_SUCCESS) {
      vec.resize(slice.len);
    } else {
      vec.clear(); // Clear the vector on error
    }

    return vec;
  }

  postcard_error_t encode_raw(postcard_slice_t* slice) const {
    postcard_error_t result;
    
                        {
                            auto val = std::get<0>(id);
                            result = postcard_encode_u8(slice, val);
                        }
                        
                        {
                            auto val = std::get<1>(id);
                            result = postcard_encode_u8(slice, val);
                        }
                        
        if(result != POSTCARD_SUCCESS) return result;
    result = vtable.encode_raw(slice);
        if(result != POSTCARD_SUCCESS) return result;
    
    return POSTCARD_SUCCESS;
  }

  postcard_error_t decode(std::span<const uint8_t>& input) {
    postcard_slice_t slice;
    postcard_init_slice(&slice, const_cast<uint8_t*>(input.data()), input.size());
    postcard_error_t result = decode_raw(&slice);
    if (result == POSTCARD_SUCCESS) {
      // Update the input span to point past the decoded data
      input = input.subspan(slice.len);
    }
    return result;
  }

  postcard_error_t decode_raw(postcard_slice_t* slice) {
    postcard_error_t result;
    uint8_t val0;
    result = postcard_decode_u8(slice, &val0);
    uint8_t val1;
    result = postcard_decode_u8(slice, &val1);
    id =  std::tuple<uint8_t, uint8_t>(val0, val1);
    if(result != POSTCARD_SUCCESS) return result;

    result = vtable.decode_raw(slice);
    if(result != POSTCARD_SUCCESS) return result;

    
    return POSTCARD_SUCCESS;
  }
};

struct VTableStream {
  static constexpr std::string_view TYPE_NAME = "VTableStream";

  std::tuple<uint8_t, uint8_t> id;
  


  size_t encoded_size() const {
    size_t size = 0;
    size += postcard_size_u8();size += postcard_size_u8();
    
    return size;
  }

  postcard_error_t encode(std::span<uint8_t>& output) const {
    postcard_slice_t slice;
    postcard_init_slice(&slice, output.data(), output.size());
    auto res = encode_raw(&slice);
    if(res != POSTCARD_SUCCESS) return res;
    output = output.subspan(0, slice.len);
    return POSTCARD_SUCCESS;
  }

  std::vector<uint8_t> encode_vec() const {
    // Pre-allocate vector with the required size
    std::vector<uint8_t> vec(encoded_size());

    // Create a span from the vector
    auto span = std::span<uint8_t>(vec);

    // Encode into the span
    postcard_slice_t slice;
    postcard_init_slice(&slice, span.data(), span.size());
    auto res = encode_raw(&slice);

    // Resize to actual used length if successful
    if (res == POSTCARD_SUCCESS) {
      vec.resize(slice.len);
    } else {
      vec.clear(); // Clear the vector on error
    }

    return vec;
  }

  postcard_error_t encode_raw(postcard_slice_t* slice) const {
    postcard_error_t result;
    
                        {
                            auto val = std::get<0>(id);
                            result = postcard_encode_u8(slice, val);
                        }
                        
                        {
                            auto val = std::get<1>(id);
                            result = postcard_encode_u8(slice, val);
                        }
                        
        if(result != POSTCARD_SUCCESS) return result;
    
    return POSTCARD_SUCCESS;
  }

  postcard_error_t decode(std::span<const uint8_t>& input) {
    postcard_slice_t slice;
    postcard_init_slice(&slice, const_cast<uint8_t*>(input.data()), input.size());
    postcard_error_t result = decode_raw(&slice);
    if (result == POSTCARD_SUCCESS) {
      // Update the input span to point past the decoded data
      input = input.subspan(slice.len);
    }
    return result;
  }

  postcard_error_t decode_raw(postcard_slice_t* slice) {
    postcard_error_t result;
    uint8_t val0;
    result = postcard_decode_u8(slice, &val0);
    uint8_t val1;
    result = postcard_decode_u8(slice, &val1);
    id =  std::tuple<uint8_t, uint8_t>(val0, val1);
    if(result != POSTCARD_SUCCESS) return result;

    
    return POSTCARD_SUCCESS;
  }
};

struct ComponentMetadata {
  static constexpr std::string_view TYPE_NAME = "ComponentMetadata";

  uint64_t component_id;
  std::string name;
  std::unordered_map<std::string, std::string> metadata;
  bool asset;
  


  size_t encoded_size() const {
    size_t size = 0;
    size += postcard_size_u64(component_id);
    size += postcard_size_string(name.length());
    size += postcard_size_map(metadata.size());
    for([[maybe_unused]] const auto& [k, v]: metadata) {
      size += postcard_size_string(k.length());
      size += postcard_size_string(v.length());
    }
    size += postcard_size_bool();
    
    return size;
  }

  postcard_error_t encode(std::span<uint8_t>& output) const {
    postcard_slice_t slice;
    postcard_init_slice(&slice, output.data(), output.size());
    auto res = encode_raw(&slice);
    if(res != POSTCARD_SUCCESS) return res;
    output = output.subspan(0, slice.len);
    return POSTCARD_SUCCESS;
  }

  std::vector<uint8_t> encode_vec() const {
    // Pre-allocate vector with the required size
    std::vector<uint8_t> vec(encoded_size());

    // Create a span from the vector
    auto span = std::span<uint8_t>(vec);

    // Encode into the span
    postcard_slice_t slice;
    postcard_init_slice(&slice, span.data(), span.size());
    auto res = encode_raw(&slice);

    // Resize to actual used length if successful
    if (res == POSTCARD_SUCCESS) {
      vec.resize(slice.len);
    } else {
      vec.clear(); // Clear the vector on error
    }

    return vec;
  }

  postcard_error_t encode_raw(postcard_slice_t* slice) const {
    postcard_error_t result;
    result = postcard_encode_u64(slice, component_id);
        if(result != POSTCARD_SUCCESS) return result;
    result = postcard_encode_string(slice, name.c_str(), name.length());
        if(result != POSTCARD_SUCCESS) return result;
    result = postcard_start_map(slice, metadata.size());
    for(const auto& [k, v]: metadata) {
      result = postcard_encode_string(slice, k.c_str(), k.length());
      result = postcard_encode_string(slice, v.c_str(), v.length());
      if(result != POSTCARD_SUCCESS) return result;
    }
        if(result != POSTCARD_SUCCESS) return result;
    result = postcard_encode_bool(slice, asset);
        if(result != POSTCARD_SUCCESS) return result;
    
    return POSTCARD_SUCCESS;
  }

  postcard_error_t decode(std::span<const uint8_t>& input) {
    postcard_slice_t slice;
    postcard_init_slice(&slice, const_cast<uint8_t*>(input.data()), input.size());
    postcard_error_t result = decode_raw(&slice);
    if (result == POSTCARD_SUCCESS) {
      // Update the input span to point past the decoded data
      input = input.subspan(slice.len);
    }
    return result;
  }

  postcard_error_t decode_raw(postcard_slice_t* slice) {
    postcard_error_t result;
    result = postcard_decode_u64(slice, &component_id);
    if(result != POSTCARD_SUCCESS) return result;

    size_t name_len;
    result = postcard_decode_string_len(slice, &name_len);
    if (result != POSTCARD_SUCCESS) return result;
    name.resize(name_len);
    if (name_len > 0) {
        result = postcard_decode_string(slice, name.data(), name_len, name_len);
        if (result != POSTCARD_SUCCESS) return result;
    }
    if(result != POSTCARD_SUCCESS) return result;

    size_t metadata_len;
    result = postcard_decode_map_len(slice, &metadata_len);
    if (result != POSTCARD_SUCCESS) return result;
    metadata.clear();
    for(size_t i = 0; i < metadata_len; i++) {
        std::string k;
        size_t k_len;
        result = postcard_decode_string_len(slice, &k_len);
        if (result != POSTCARD_SUCCESS) return result;
        k.resize(k_len);
        if (k_len > 0) {
            result = postcard_decode_string(slice, k.data(), k_len, k_len);
            if (result != POSTCARD_SUCCESS) return result;
        }
        if (result != POSTCARD_SUCCESS) return result;
        std::string v;
        size_t v_len;
        result = postcard_decode_string_len(slice, &v_len);
        if (result != POSTCARD_SUCCESS) return result;
        v.resize(v_len);
        if (v_len > 0) {
            result = postcard_decode_string(slice, v.data(), v_len, v_len);
            if (result != POSTCARD_SUCCESS) return result;
        }
        if (result != POSTCARD_SUCCESS) return result;
        metadata[k] = v;
    }
    if(result != POSTCARD_SUCCESS) return result;

    result = postcard_decode_bool(slice, &asset);
    if(result != POSTCARD_SUCCESS) return result;

    
    return POSTCARD_SUCCESS;
  }
};

struct SetComponentMetadata {
  static constexpr std::string_view TYPE_NAME = "SetComponentMetadata";

  ComponentMetadata value;

  size_t encoded_size() const {
    size_t size = 0;
    size += value.encoded_size();
    return size;
  }

  postcard_error_t encode(std::span<std::byte>& output) const {
    postcard_slice_t slice;
    postcard_init_slice(&slice, reinterpret_cast<uint8_t*>(output.data()), output.size());
    auto res = encode_raw(&slice);
    if(res != POSTCARD_SUCCESS) return res;
    output = output.subspan(0, slice.len);
    return POSTCARD_SUCCESS;
  }

  std::vector<std::byte> encode_vec() const {
    // Pre-allocate vector with the required size
    std::vector<std::byte> vec(encoded_size());

    // Create a span from the vector
    auto span = std::span<std::byte>(vec);

    // Encode into the span
    postcard_slice_t slice;
    postcard_init_slice(&slice, reinterpret_cast<uint8_t*>(span.data()), span.size());
    auto res = encode_raw(&slice);

    // Resize to actual used length if successful
    if (res == POSTCARD_SUCCESS) {
      vec.resize(slice.len);
    } else {
      vec.clear(); // Clear the vector on error
    }

    return vec;
  }

  postcard_error_t encode_raw(postcard_slice_t* slice) const {
    postcard_error_t result;
    result = value.encode_raw(slice);
    return result;
  }

  postcard_error_t decode(std::span<const std::byte>& input) {
    postcard_slice_t slice;
    postcard_init_slice(&slice, const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(input.data())), input.size());
    postcard_error_t result = decode_raw(&slice);
    if (result == POSTCARD_SUCCESS) {
      // Update the input span to point past the decoded data
      input = input.subspan(slice.len);
    }
    return result;
  }

  postcard_error_t decode_raw(postcard_slice_t* slice) {
    postcard_error_t result;
    result = value.decode_raw(slice);
    return result;
  }
};

#ifndef ELO_DB_HELPERS_H
#define ELO_DB_HELPERS_H

#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>
#include <vector>

inline uint32_t fnv1a_hash_32(const std::string_view str)
{
    uint32_t hash = 0x811c9dc5;
    size_t i = 0;
    for (auto c : str) {
        if (++i >= 32) {
            break;
        }
        hash ^= static_cast<uint8_t>(c);
        hash *= 0x01000193;
    }
    return hash;
}

inline uint64_t fnv1a_hash_64(const std::string_view str)
{
    uint64_t hash = 0xcbf29ce484222325;
    size_t i = 0;
    for (auto c : str) {
        if (++i >= 64) {
            break;
        }
        hash ^= static_cast<uint8_t>(c);
        hash *= 0x00000100000001B3;
    }
    return hash;
}

inline uint16_t fnv1a_hash_16_xor(const std::string_view str)
{
    auto hash = fnv1a_hash_32(str);
    uint16_t upper = static_cast<uint16_t>((hash >> 16) & 0xFFFF);
    uint16_t lower = static_cast<uint16_t>(hash & 0xFFFF);
    return upper ^ lower;
}

inline std::array<uint8_t, 2> msg_id(const std::string_view str)
{
    auto hash = fnv1a_hash_16_xor(str);
    return { static_cast<uint8_t>(hash & 0xff), static_cast<uint8_t>((hash >> 8) & 0xff) };
}

inline uint64_t component_id(const std::string_view str)
{
    auto hash = fnv1a_hash_64(str) & ~(1ul << 63);
    return hash;
}

enum class PacketType : uint8_t {
    MSG = 0,
    TABLE = 1,
    TIME_SERIES = 2
};

struct PacketHeader {
    uint32_t len;
    PacketType ty;
    std::array<uint8_t, 2> packet_id;
    uint8_t request_id;
};

template <typename T>
class Msg {
    PacketHeader header;
    T payload;

public:
    Msg(T p)
    {
        auto packet_id = msg_id(T::TYPE_NAME);
        header = PacketHeader {
            .len = 0,
            .ty = PacketType::MSG,
            .packet_id = packet_id,
            .request_id = 0,
        };
        payload = p;
    }

    std::vector<uint8_t> encode_vec()
    {
        auto t_size = payload.encoded_size();
        header.len = t_size + 4;
        auto header_len = sizeof(PacketHeader);
        auto buf = std::vector<uint8_t>(t_size + header_len);
        std::memcpy(buf.data(), &header, header_len);
        auto span = std::span<uint8_t>(buf).subspan(header_len, t_size);
        postcard_slice_t slice;
        postcard_init_slice(&slice, span.data(), span.size());
        auto res = payload.encode_raw(&slice);
        if (res == POSTCARD_SUCCESS) {
            buf.resize(slice.len + header_len);
        } else {
            buf.clear();
        }

        return buf;
    }
};

SetComponentMetadata set_component_name(std::string name) {
   return SetComponentMetadata(ComponentMetadata {
       .component_id = component_id(name),
       .name = std::move(name),
   });
}

#endif


#ifndef ELO_DB_VTABLE_H
#define ELO_DB_VTABLE_H

#include <string_view>
#if __has_include("db.hpp")
#include "db.hpp"
#endif

#include <cstddef>
#include <cstdint>
#include <memory>
#include <print>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

namespace vtable {

/// A builder for VTable operations
class OpBuilder {
public:
    struct Data {
        size_t align;
        std::vector<uint8_t> data;
    };

    struct Table {
        uint16_t offset;
        uint16_t len;
    };

    struct Pair {
        std::shared_ptr<OpBuilder> entity_id;
        std::shared_ptr<OpBuilder> component_id;
    };

    struct Schema {
        std::shared_ptr<OpBuilder> ty;
        std::shared_ptr<OpBuilder> dim;
        std::shared_ptr<OpBuilder> arg;
    };

    struct Timestamp {
        std::shared_ptr<OpBuilder> timestamp;
        std::shared_ptr<OpBuilder> arg;
    };

    struct Ext {
        std::tuple<uint8_t, uint8_t> id;
        std::shared_ptr<OpBuilder> data;
        std::shared_ptr<OpBuilder> arg;
    };

    using ValueType = std::variant<Data, Table, Pair, Schema, Timestamp, Ext>;
    ValueType value;

    explicit OpBuilder(const Data& data)
        : value(data)
    {
    }

    explicit OpBuilder(const Table& table)
        : value(table)
    {
    }

    explicit OpBuilder(const Pair& pair)
        : value(pair)
    {
    }

    explicit OpBuilder(const Schema& schema)
        : value(schema)
    {
    }

    explicit OpBuilder(const Timestamp& timestamp)
        : value(timestamp)
    {
    }

    explicit OpBuilder(const Ext& ext)
        : value(ext)
    {
    }

    // Prevent copy, only allow move
    OpBuilder(const OpBuilder&) = delete;
    OpBuilder& operator=(const OpBuilder&) = delete;

    OpBuilder(OpBuilder&& other) noexcept = default;
    OpBuilder& operator=(OpBuilder&& other) noexcept = default;
};

/// A builder for VTable fields
class FieldBuilder {
public:
    uint16_t offset;
    uint16_t len;
    std::shared_ptr<OpBuilder> arg;

    FieldBuilder(uint16_t offset, uint16_t len, std::shared_ptr<OpBuilder> arg)
        : offset(offset)
        , len(len)
        , arg(std::move(arg))
    {
    }
};

namespace builder {

    /// Create an OpBuilder that includes the passed in data in the data section
    /// ## Safety
    /// The type passed here should be safe to memcpy
    template <typename T>
    std::shared_ptr<OpBuilder> data(const std::vector<T>& values)
    {
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(values.data());
        std::vector<uint8_t> data_vec(bytes, bytes + sizeof(T) * values.size());
        OpBuilder::Data data { alignof(T), std::move(data_vec) };
        return std::make_shared<OpBuilder>(data);
    }

    /// Create an OpBuilder that includes the passed in data in the data section
    /// ## Safety
    /// The type passed here should be safe to memcpy
    template <typename T>
    std::shared_ptr<OpBuilder> data(const T& value)
    {
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
        std::vector<uint8_t> data_vec(bytes, bytes + sizeof(T));
        OpBuilder::Data data { alignof(T), std::move(data_vec) };
        return std::make_shared<OpBuilder>(data);
    }

    /// Create an OpBuilder that includes the passed in data in the data section,
    /// this includes an optional alignment value that
    inline std::shared_ptr<OpBuilder> data(const std::vector<uint8_t>& bytes, size_t align = 1)
    {
        OpBuilder::Data data { align, bytes };
        return std::make_shared<OpBuilder>(data);
    }

    /// Creates an OpBuilder with the table specified offset and length
    inline std::shared_ptr<OpBuilder> raw_table(uint16_t offset, uint16_t len)
    {
        OpBuilder::Table table { offset, len };
        return std::make_shared<OpBuilder>(table);
    }

    /// Creates a pair operation builder from an entity ID and component ID
    inline std::shared_ptr<OpBuilder> pair(uint64_t entity_id, std::string_view component_name)
    {
        auto id = component_id(component_name);
        auto entity_id_op = data(entity_id);
        auto component_id_op = data(id);

        OpBuilder::Pair pair { std::move(entity_id_op), std::move(component_id_op) };
        return std::make_shared<OpBuilder>(pair);
    }

    /// Creates a schema operation builder from a primitive type, dimensions, and an argument
    inline std::shared_ptr<OpBuilder> schema(PrimType ty, const std::vector<uint64_t>& dim, std::shared_ptr<OpBuilder> arg)
    {
        auto ty_op = builder::data(static_cast<uint64_t>(ty.index()));
        auto dim_op = builder::data(dim);

        OpBuilder::Schema schema { std::move(ty_op), std::move(dim_op), std::move(arg) };
        return std::make_shared<OpBuilder>(schema);
    }

    /// Creates a timestamp operation builder from a timestamp source and an argument
    inline std::shared_ptr<OpBuilder> timestamp(std::shared_ptr<OpBuilder> timestamp, std::shared_ptr<OpBuilder> arg)
    {
        OpBuilder::Timestamp ts { std::move(timestamp), std::move(arg) };
        return std::make_shared<OpBuilder>(ts);
    }

    /**
     * Creates an extension operation builder from a message ID, data, and an argument
     */
    inline std::shared_ptr<OpBuilder> ext(std::tuple<uint8_t, uint8_t> id, std::shared_ptr<OpBuilder> data,
        std::shared_ptr<OpBuilder> arg)
    {
        OpBuilder::Ext ext { id, std::move(data), std::move(arg) };
        return std::make_shared<OpBuilder>(ext);
    }

    /// Creates a field builder with the specified offset, length, and argument
    inline FieldBuilder raw_field(uint16_t offset, uint16_t len, std::shared_ptr<OpBuilder> arg)
    {
        return FieldBuilder(offset, len, std::move(arg));
    }

    /// Creates a field builder from a class and its field
    /// ## Usage
    /// ```cpp
    ///  builder::field<Foo, &Foo::time>(builder::schema(PrimType::F64(), {}, builder::pair(1, "time"))),
    /// ```
    template <typename Class, auto MemberPtr>
    inline FieldBuilder field(std::shared_ptr<OpBuilder> arg)
    {
        using FieldType = std::remove_reference_t<decltype(std::declval<Class>().*MemberPtr)>;
        size_t offset = reinterpret_cast<size_t>(&(reinterpret_cast<Class*>(0)->*MemberPtr));
        size_t size = sizeof(FieldType);
        return raw_field(static_cast<uint16_t>(offset), static_cast<uint16_t>(size), std::move(arg));
    }

    /// A builder for constructing VTables
    class VTableBuilder {
    private:
        VTable vtable;
        std::unordered_map<const OpBuilder*, OpRef> visited;

    public:
        VTableBuilder() = default;

        /// Visits an operation builder, adding it to the VTable and returning its OpRef
        OpRef visit(const std::shared_ptr<OpBuilder>& op)
        {
            auto it = visited.find(op.get());
            if (it != visited.end()) {
                return it->second;
            }
            Op result_op;

            std::visit([&](const auto& val) {
                using T = std::decay_t<decltype(val)>;

                if constexpr (std::is_same_v<T, OpBuilder::Data>) {
                    const auto& data_op = val;

                    const size_t align = data_op.align;
                    const size_t padding = (align - (vtable.data.size() % align)) % align;

                    for (size_t i = 0; i < padding; i++) {
                        vtable.data.push_back(0);
                    }

                    const uint16_t offset = static_cast<uint16_t>(vtable.data.size());
                    const uint16_t len = static_cast<uint16_t>(data_op.data.size());

                    vtable.data.insert(vtable.data.end(), data_op.data.begin(), data_op.data.end());

                    result_op = Op::Data(OpData { offset, len });

                } else if constexpr (std::is_same_v<T, OpBuilder::Table>) {
                    const auto& table_op = val;
                    result_op = Op::Table(OpTable { table_op.offset, table_op.len });
                } else if constexpr (std::is_same_v<T, OpBuilder::Pair>) {
                    const auto& pair_op = val;
                    OpRef entity_id = visit(pair_op.entity_id);
                    OpRef component_id = visit(pair_op.component_id);

                    result_op = Op::Pair(OpPair {
                        static_cast<uint16_t>(entity_id.value),
                        static_cast<uint16_t>(component_id.value) });
                } else if constexpr (std::is_same_v<T, OpBuilder::Schema>) {
                    const auto& schema_op = val;
                    OpRef ty = visit(schema_op.ty);
                    OpRef dim = visit(schema_op.dim);
                    OpRef arg = visit(schema_op.arg);

                    result_op = Op::Schema(OpSchema {
                        static_cast<uint16_t>(ty.value),
                        static_cast<uint16_t>(dim.value),
                        static_cast<uint16_t>(arg.value) });
                } else if constexpr (std::is_same_v<T, OpBuilder::Timestamp>) {
                    const auto& timestamp_op = val;
                    OpRef timestamp = visit(timestamp_op.timestamp);
                    OpRef arg = visit(timestamp_op.arg);

                    result_op = Op::Timestamp(OpTimestamp {
                        static_cast<uint16_t>(timestamp.value),
                        static_cast<uint16_t>(arg.value) });
                } else if constexpr (std::is_same_v<T, OpBuilder::Ext>) {
                    const auto& ext_op = val;
                    OpRef data = visit(ext_op.data);
                    OpRef arg = visit(ext_op.arg);

                    result_op = Op::Ext(OpExt {
                        static_cast<uint16_t>(arg.value),
                        ext_op.id,
                        static_cast<uint16_t>(data.value) });
                }
            },
                op->value);

            OpRef op_ref(static_cast<uint16_t>(vtable.ops.size()));
            vtable.ops.push_back(result_op);
            visited[op.get()] = op_ref;

            return op_ref;
        }

        /// Adds a field to the VTable
        void push_field(const FieldBuilder& field_builder)
        {
            OpRef arg = visit(field_builder.arg);

            Field field {
                field_builder.offset,
                field_builder.len,
                static_cast<uint16_t>(arg.value)
            };

            vtable.fields.push_back(field);
        }

        /// Builds and returns the VTable
        VTable build() const
        {
            return vtable;
        }
    };

    /// Creates a VTable from the provided field builders
    inline VTable vtable(const std::initializer_list<FieldBuilder> fields)
    {
        VTableBuilder builder;

        for (const auto& field : fields) {
            builder.push_field(field);
        }

        return builder.build();
    }

// Convenience macro to create a table operation for a struct field
#define TABLE(type, field)                            \
    vtable::builder::raw_table(                       \
        static_cast<uint16_t>(offsetof(type, field)), \
        static_cast<uint16_t>(sizeof(((type*)nullptr)->field)))

// Convenience macro to create a field builder for a struct field
#define FIELD(type, field, arg)                                 \
    vtable::builder::raw_field(                                 \
        static_cast<uint16_t>(offsetof(type, field)),           \
        static_cast<uint16_t>(sizeof(((type*)nullptr)->field)), \
        arg)

} // namespace builder
} // namespace vtable

#endif // ELO_DB_VTABLE_H



#endif