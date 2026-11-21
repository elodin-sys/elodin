// The fixed-size UDP packets of examples/many-agents/protocol.py: a 1024-byte state packet for
// each agent, and the 104-byte reply its flight software sends back. All little-endian.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "the packets are little-endian");

namespace protocol {

constexpr size_t HEADER_SIZE = 16;
constexpr size_t STATE_DOUBLES = 126;
constexpr size_t COMMAND_DOUBLES = 10;
constexpr size_t STATE_SIZE = HEADER_SIZE + STATE_DOUBLES * 8;
// The reply ends with the time the flight software took to answer, which describes the host.
constexpr size_t COMMAND_SIZE = HEADER_SIZE + COMMAND_DOUBLES * 8 + 8;
// The reply's values: everything the flight software decided.
constexpr size_t RESPONSE_SIZE = COMMAND_DOUBLES * 8;
constexpr uint16_t VERSION = 1;
constexpr char STATE_MAGIC[4] = { 'M', 'A', 'L', 'S' };
constexpr char COMMAND_MAGIC[4] = { 'M', 'A', 'L', 'C' };

static_assert(STATE_SIZE == 1024 && COMMAND_SIZE == 104);

// Where each part of the state sits among its doubles.
constexpr size_t AGENT_POSITION = 0;
constexpr size_t AGENT_VELOCITY = 3;
constexpr size_t OBJECT_POSITION = 6;
constexpr size_t OBJECT_VELOCITY = 9;
constexpr size_t DIAGNOSTICS = 12;

inline void write_header(uint8_t* packet, const char (&magic)[4], uint64_t tick)
{
    const uint16_t version = VERSION, flags = 0;
    std::memcpy(packet, magic, 4);
    std::memcpy(packet + 4, &version, 2);
    std::memcpy(packet + 6, &flags, 2);
    std::memcpy(packet + 8, &tick, 8);
}

// Whether `packet` starts with `magic`, version 1 and no flags; the tick goes to `tick`.
inline bool read_header(const uint8_t* packet, const char (&magic)[4], uint64_t* tick)
{
    uint16_t version, flags;
    std::memcpy(&version, packet + 4, 2);
    std::memcpy(&flags, packet + 6, 2);
    std::memcpy(tick, packet + 8, 8);
    return std::memcmp(packet, magic, 4) == 0 && version == VERSION && flags == 0;
}

inline double read_f64(const uint8_t* packet, size_t index)
{
    double value;
    std::memcpy(&value, packet + HEADER_SIZE + index * 8, 8);
    return value;
}

inline void write_f64(uint8_t* packet, size_t index, double value)
{
    std::memcpy(packet + HEADER_SIZE + index * 8, &value, 8);
}

} // namespace protocol
