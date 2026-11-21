// Elodin's deterministic witness file (libs/nox-py/src/witness/format.rs), written from C++ so
// `elodin determinism compare` can check this program against an Elodin run tick by tick:
//
//     file    := "ELWITNES" version:u32 header_len:u32 header_json frame* trailer
//     frame   := 0x01 tick:u64 timestamp:i64 digest:[u8; 32]* raw:u8 (len:u32 bytes)*
//     trailer := 0x02 json_len:u32 trailer_json
#pragma once

#include "blake3.hpp"

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace witness {

struct Field {
    std::string name;
    std::string component;
    std::vector<uint64_t> shape;
    uint64_t row_bytes;
    bool write;
    // The entity of each row.
    std::vector<uint64_t> entities;
};

struct Batch {
    std::string name;
    std::vector<Field> fields;
};

struct Channel {
    std::string name;
    size_t batch;
    uint64_t row_bytes;
    // Bytes from outside the simulation: kept with the raw bytes, never hashed.
    bool external;
};

struct Hook {
    std::string name;
    // The batch a native hook was loaded with; empty for none.
    std::string batch;
    std::vector<std::pair<uint64_t, uint64_t>> partitions;
};

enum class Kind { Field,
    Channel,
    External,
    Deferred,
    Recorded };

// One record of every tick, in the order the schema lists them.
struct Record {
    size_t stage;
    Kind kind;
    std::string name;
    // Index of the batch whose rows it has; -1 for deferred writes.
    long batch;
    size_t index;
    uint64_t row_bytes;
};

class Writer {
public:
    // The records are Elodin's for these batches and channels and `hooks.size()` hooks.
    Writer(const std::string& path, std::vector<Batch> batches, std::vector<Channel> channels,
        std::vector<Hook> hooks, const std::map<uint64_t, std::string>& entity_names, bool raw);
    ~Writer();
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;

    const std::vector<Record>& records() const { return records_; }

    // A tick takes every record's bytes, in schema order, between `begin` and `end`. An empty
    // channel is one no hook filled.
    void begin(uint64_t tick, int64_t timestamp);
    void add(const void* bytes, size_t size);
    void end();

    // Writes the trailer and returns the root, the digest of the whole run.
    std::string finish();
    uint64_t ticks() const { return ticks_; }

private:
    std::FILE* out_;
    std::vector<Record> records_;
    std::vector<size_t> stage_hashed_;
    bool raw_;
    Blake3 root_;
    uint64_t ticks_ = 0;
    uint64_t tick_ = 0;
    int64_t timestamp_ = 0;
    std::vector<uint8_t> bytes_;
    std::vector<size_t> ends_;
};

} // namespace witness
