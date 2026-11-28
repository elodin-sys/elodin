// Portable BLAKE3-256 (unkeyed), after the reference implementation in the BLAKE3 repository.
// The witness hashes every record with it, as Elodin does.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

using Digest = std::array<uint8_t, 32>;

class Blake3 {
public:
    Blake3();
    void update(const void* data, size_t size);
    Digest finalize() const;

    static Digest hash(const void* data, size_t size);

private:
    struct Chunk {
        uint32_t cv[8];
        uint64_t counter;
        uint8_t block[64];
        size_t block_len;
        size_t blocks_compressed;
    };

    void push_chunk(const uint32_t cv[8], uint64_t total_chunks);
    static void start_chunk(Chunk& chunk, uint64_t counter);

    Chunk chunk_;
    uint32_t stack_[54][8];
    size_t stack_len_ = 0;
};

std::string hex(const Digest& digest);
