#include "flight_software.hpp"

#include "protocol.hpp"

#include <algorithm>
#include <cstring>

FlightSoftware::FlightSoftware(uint64_t agent)
    : agent_(agent)
{
}

bool FlightSoftware::step(const uint8_t* state, size_t size, uint8_t* reply) const
{
    uint64_t tick;
    if (size != protocol::STATE_SIZE || !protocol::read_header(state, protocol::STATE_MAGIC, &tick))
        return false;
    std::memset(reply, 0, protocol::COMMAND_SIZE);
    protocol::write_header(reply, protocol::COMMAND_MAGIC, tick);
    for (size_t axis = 0; axis < 3; ++axis) {
        const double position_error = protocol::read_f64(state, protocol::OBJECT_POSITION + axis)
            - protocol::read_f64(state, protocol::AGENT_POSITION + axis);
        const double velocity_error = protocol::read_f64(state, protocol::OBJECT_VELOCITY + axis)
            - protocol::read_f64(state, protocol::AGENT_VELOCITY + axis);
        // PD guidance, saturated at the acceleration the vehicle can command.
        const double command = std::clamp(1.4 * position_error + 0.45 * velocity_error, -12.0, 12.0);
        protocol::write_f64(reply, axis, command);
    }
    return true;
}
