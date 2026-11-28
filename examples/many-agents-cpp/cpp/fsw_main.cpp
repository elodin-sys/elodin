// One agent's flight software as a process: answers each state packet that arrives on
// STATE_PORT with a reply to COMMAND_PORT, both on 127.0.0.1. A drop-in for the Rust controller
// of examples/many-agents.
//
//     many_agents_fsw STATE_PORT COMMAND_PORT AGENT
#include "flight_software.hpp"
#include "protocol.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static sockaddr_in localhost(uint16_t port)
{
    sockaddr_in address {};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    return address;
}

int main(int argc, char** argv)
{
    if (argc != 4) {
        std::fprintf(stderr, "usage: many_agents_fsw STATE_PORT COMMAND_PORT AGENT\n");
        return 2;
    }
    const sockaddr_in state_address = localhost(static_cast<uint16_t>(std::atoi(argv[1])));
    const sockaddr_in command_address = localhost(static_cast<uint16_t>(std::atoi(argv[2])));
    const FlightSoftware fsw(std::strtoull(argv[3], nullptr, 10));

    const int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0 || bind(sock, reinterpret_cast<const sockaddr*>(&state_address), sizeof state_address) != 0) {
        std::perror("many_agents_fsw: bind");
        return 1;
    }
    std::printf("ready\n");
    std::fflush(stdout);

    uint8_t state[protocol::STATE_SIZE];
    uint8_t reply[protocol::COMMAND_SIZE];
    for (;;) {
        const ssize_t size = recv(sock, state, sizeof state, 0);
        if (size < 0) {
            std::perror("many_agents_fsw: recv");
            return 1;
        }
        const auto received = std::chrono::steady_clock::now();
        // Like the Rust controller, a malformed packet gets no reply.
        if (!fsw.step(state, static_cast<size_t>(size), reply))
            continue;
        const auto service = std::chrono::steady_clock::now() - received;
        const uint64_t service_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(service).count());
        std::memcpy(reply + protocol::COMMAND_SIZE - 8, &service_ns, 8);
        sendto(sock, reply, sizeof reply, 0, reinterpret_cast<const sockaddr*>(&command_address),
            sizeof command_address);
    }
}
