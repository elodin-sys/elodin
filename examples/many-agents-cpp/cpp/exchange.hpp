// What the native hook of examples/many-agents does on every tick: send each agent's flight
// software its state packet, and wait for every reply.
#pragma once

#include "flight_software.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <sys/types.h>
#include <vector>

class Exchange {
public:
    virtual ~Exchange() = default;
    // Takes `agents` state packets back to back and fills as many replies; throws on failure.
    virtual void run(const uint8_t* states, uint8_t* replies) = 0;
};

// Calls each agent's flight software directly.
class InProcessExchange final : public Exchange {
public:
    explicit InProcessExchange(size_t agents);
    void run(const uint8_t* states, uint8_t* replies) override;

private:
    std::vector<FlightSoftware> agents_;
};

// One flight-software process per agent, over UDP on 127.0.0.1, as Elodin's process recipes
// run them: agent i reads states on PORT_BASE + 2i and replies to PORT_BASE + 2i + 1.
class ProcessExchange final : public Exchange {
public:
    ProcessExchange(const std::string& binary, size_t agents, uint16_t port_base,
        std::chrono::milliseconds timeout);
    ~ProcessExchange() override;
    void run(const uint8_t* states, uint8_t* replies) override;

private:
    void stop();

    size_t agents_;
    uint16_t port_base_;
    std::chrono::milliseconds timeout_;
    int sender_ = -1;
    std::vector<int> receivers_;
    std::vector<pid_t> children_;
};

std::unique_ptr<Exchange> make_exchange(const std::string& transport, const std::string& fsw_binary,
    size_t agents, uint16_t port_base, std::chrono::milliseconds timeout);
