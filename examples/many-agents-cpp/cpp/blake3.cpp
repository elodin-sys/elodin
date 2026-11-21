#include "blake3.hpp"

#include <algorithm>
#include <cstring>

namespace {

constexpr uint32_t IV[8] = {
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
};
constexpr size_t PERMUTATION[16] = { 2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8 };
constexpr uint32_t CHUNK_START = 1, CHUNK_END = 2, PARENT = 4, ROOT = 8;
constexpr size_t BLOCK_LEN = 64, CHUNK_LEN = 1024;

uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void g(uint32_t* s, size_t a, size_t b, size_t c, size_t d, uint32_t x, uint32_t y)
{
    s[a] = s[a] + s[b] + x;
    s[d] = rotr(s[d] ^ s[a], 16);
    s[c] = s[c] + s[d];
    s[b] = rotr(s[b] ^ s[c], 12);
    s[a] = s[a] + s[b] + y;
    s[d] = rotr(s[d] ^ s[a], 8);
    s[c] = s[c] + s[d];
    s[b] = rotr(s[b] ^ s[c], 7);
}

// The 16 output words of compressing one block.
void compress(const uint32_t cv[8], const uint32_t block[16], uint64_t counter, uint32_t block_len,
    uint32_t flags, uint32_t out[16])
{
    uint32_t s[16] = {
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        static_cast<uint32_t>(counter),
        static_cast<uint32_t>(counter >> 32),
        block_len,
        flags,
    };
    uint32_t m[16];
    std::memcpy(m, block, sizeof m);
    for (int round = 0; round < 7; ++round) {
        g(s, 0, 4, 8, 12, m[0], m[1]);
        g(s, 1, 5, 9, 13, m[2], m[3]);
        g(s, 2, 6, 10, 14, m[4], m[5]);
        g(s, 3, 7, 11, 15, m[6], m[7]);
        g(s, 0, 5, 10, 15, m[8], m[9]);
        g(s, 1, 6, 11, 12, m[10], m[11]);
        g(s, 2, 7, 8, 13, m[12], m[13]);
        g(s, 3, 4, 9, 14, m[14], m[15]);
        uint32_t permuted[16];
        for (size_t i = 0; i < 16; ++i)
            permuted[i] = m[PERMUTATION[i]];
        std::memcpy(m, permuted, sizeof m);
    }
    for (size_t i = 0; i < 8; ++i) {
        out[i] = s[i] ^ s[i + 8];
        out[i + 8] = s[i + 8] ^ cv[i];
    }
}

void words(const uint8_t bytes[64], uint32_t out[16])
{
    for (size_t i = 0; i < 16; ++i)
        out[i] = static_cast<uint32_t>(bytes[4 * i]) | static_cast<uint32_t>(bytes[4 * i + 1]) << 8
            | static_cast<uint32_t>(bytes[4 * i + 2]) << 16 | static_cast<uint32_t>(bytes[4 * i + 3]) << 24;
}

// A node not yet compressed: the last block of a chunk, or a parent of two chaining values.
struct Output {
    uint32_t cv[8];
    uint32_t block[16];
    uint64_t counter;
    uint32_t block_len;
    uint32_t flags;

    void chaining_value(uint32_t out[8]) const
    {
        uint32_t full[16];
        compress(cv, block, counter, block_len, flags, full);
        std::memcpy(out, full, 32);
    }
};

Output parent(const uint32_t left[8], const uint32_t right[8])
{
    Output output {};
    std::memcpy(output.cv, IV, sizeof IV);
    std::memcpy(output.block, left, 32);
    std::memcpy(output.block + 8, right, 32);
    output.block_len = BLOCK_LEN;
    output.flags = PARENT;
    return output;
}

} // namespace

void Blake3::start_chunk(Chunk& chunk, uint64_t counter)
{
    std::memcpy(chunk.cv, IV, sizeof IV);
    chunk.counter = counter;
    chunk.block_len = 0;
    chunk.blocks_compressed = 0;
}

Blake3::Blake3() { start_chunk(chunk_, 0); }

void Blake3::push_chunk(const uint32_t chunk_cv[8], uint64_t total_chunks)
{
    uint32_t cv[8];
    std::memcpy(cv, chunk_cv, sizeof cv);
    // Each completed subtree merges with the one before it, like carrying in binary addition.
    while ((total_chunks & 1) == 0) {
        parent(stack_[--stack_len_], cv).chaining_value(cv);
        total_chunks >>= 1;
    }
    std::memcpy(stack_[stack_len_++], cv, sizeof cv);
}

void Blake3::update(const void* data, size_t size)
{
    const uint8_t* input = static_cast<const uint8_t*>(data);
    while (size > 0) {
        const size_t chunk_len = BLOCK_LEN * chunk_.blocks_compressed + chunk_.block_len;
        if (chunk_len == CHUNK_LEN) {
            Output last {};
            std::memcpy(last.cv, chunk_.cv, sizeof last.cv);
            words(chunk_.block, last.block);
            last.counter = chunk_.counter;
            last.block_len = static_cast<uint32_t>(chunk_.block_len);
            last.flags = CHUNK_END;
            uint32_t cv[8];
            last.chaining_value(cv);
            const uint64_t total = chunk_.counter + 1;
            push_chunk(cv, total);
            start_chunk(chunk_, total);
            continue;
        }
        if (chunk_.block_len == BLOCK_LEN) {
            uint32_t block[16], out[16];
            words(chunk_.block, block);
            const uint32_t start = chunk_.blocks_compressed == 0 ? CHUNK_START : 0;
            compress(chunk_.cv, block, chunk_.counter, BLOCK_LEN, start, out);
            std::memcpy(chunk_.cv, out, 32);
            ++chunk_.blocks_compressed;
            chunk_.block_len = 0;
        }
        const size_t take = std::min({ BLOCK_LEN - chunk_.block_len,
            CHUNK_LEN - (BLOCK_LEN * chunk_.blocks_compressed + chunk_.block_len), size });
        std::memcpy(chunk_.block + chunk_.block_len, input, take);
        chunk_.block_len += take;
        input += take;
        size -= take;
    }
}

Digest Blake3::finalize() const
{
    uint8_t padded[64] = {};
    std::memcpy(padded, chunk_.block, chunk_.block_len);
    Output output {};
    std::memcpy(output.cv, chunk_.cv, sizeof output.cv);
    words(padded, output.block);
    output.counter = chunk_.counter;
    output.block_len = static_cast<uint32_t>(chunk_.block_len);
    output.flags = (chunk_.blocks_compressed == 0 ? CHUNK_START : 0) | CHUNK_END;
    for (size_t i = stack_len_; i-- > 0;) {
        uint32_t cv[8];
        output.chaining_value(cv);
        output = parent(stack_[i], cv);
    }
    uint32_t out[16];
    compress(output.cv, output.block, 0, output.block_len, output.flags | ROOT, out);
    Digest digest;
    for (size_t i = 0; i < 8; ++i)
        for (size_t j = 0; j < 4; ++j)
            digest[4 * i + j] = static_cast<uint8_t>(out[i] >> (8 * j));
    return digest;
}

Digest Blake3::hash(const void* data, size_t size)
{
    Blake3 hasher;
    hasher.update(data, size);
    return hasher.finalize();
}

std::string hex(const Digest& digest)
{
    static const char digits[] = "0123456789abcdef";
    std::string out;
    for (uint8_t byte : digest) {
        out += digits[byte >> 4];
        out += digits[byte & 15];
    }
    return out;
}
