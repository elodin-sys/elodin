// One agent's flight software: steer toward the object it follows.
#pragma once

#include <cstddef>
#include <cstdint>

class FlightSoftware {
public:
    explicit FlightSoftware(uint64_t agent);

    // Answers one state packet with a reply packet for its tick, whose service time the caller
    // fills in. False, with `reply` untouched, when `state` is not a valid state packet.
    bool step(const uint8_t* state, size_t size, uint8_t* reply) const;

    uint64_t agent() const { return agent_; }

private:
    uint64_t agent_;
};
