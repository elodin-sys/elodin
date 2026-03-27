///$(which true);FLAGS="--std=c++23";THIS_FILE="$(cd "$(dirname "$0")"; pwd -P)/$(basename "$0")";OUT_FILE="/tmp/build-cache/$THIS_FILE";mkdir -p "$(dirname "$OUT_FILE")";test "$THIS_FILE" -ot "$OUT_FILE" || $(which clang++ || which g++) $FLAGS "$THIS_FILE" -o "$OUT_FILE" || exit $?;exec bash -c "exec -a \"$0\" \"$OUT_FILE\" $([ $# -eq 0 ] || printf ' "%s"' "$@")"
//
// Simple pattern: one TCP connection per component.
// Each component gets its own Socket, VTable (1 field), and send loop
// on a dedicated thread.
//
// This works correctly but has higher overhead at scale: every packet pays
// a fixed cost (~6.6µs) for protocol parsing, write-lock acquisition,
// vtable lookup, and mmap write. With N components at F Hz, that's N*F
// packets/second instead of just F.
//
// For high-frequency or high-component-count workloads, prefer
// client-batched.cpp which packs all components into a single table packet.

#include <arpa/inet.h>
#include <cstdint>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <print>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "db.hpp"

using namespace vtable;
using namespace vtable::builder;
using namespace std::chrono;

class Socket {
public:
    Socket(const char* ip, uint16_t port)
    {
        fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ < 0) {
            throw std::system_error(
                errno, std::generic_category(), "Failed to create socket");
        }
        struct sockaddr_in server_addr = { .sin_family = AF_INET,
            .sin_port = htons(port),
            .sin_addr = { .s_addr = inet_addr(ip) } };

        if (::connect(fd_,
                reinterpret_cast<struct sockaddr*>(&server_addr),
                sizeof(server_addr))
            < 0) {
            throw std::system_error(
                errno, std::generic_category(), "Failed to connect");
        }
    }

    ~Socket()
    {
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    void write_all(const void* data, size_t len)
    {
        auto ptr = static_cast<const uint8_t*>(data);
        auto remaining = len;

        while (remaining > 0) {
            auto written = write(fd_, ptr, remaining);
            if (written < 0) {
                throw std::system_error(
                    errno, std::generic_category(), "Failed to write");
            }
            ptr += written;
            remaining -= written;
        }
    }

    template <typename T>
    void send(T msg)
    {
        auto buf = Msg(msg).encode_vec();
        write_all(buf.data(), buf.size());
    }

private:
    int fd_ = -1;
};

// Each call opens its own TCP connection, registers a 1-field VTable,
// and sends one packet per tick until the process exits.
void component_writer(
    const char* ip, uint16_t port,
    uint8_t vtable_id,
    const std::string& component_name,
    const std::vector<std::string>& labels,
    PrimType prim_type,
    const std::vector<uint64_t>& dims,
    uint16_t value_bytes)
{
    try {
        std::println("[{}] connecting to {}:{}", component_name, ip, port);
        auto sock = Socket(ip, port);

        auto time = builder::raw_table(0, 8);
        auto table = builder::vtable({
            builder::raw_field(8, value_bytes,
                schema(prim_type, dims, timestamp(time, component(component_name)))),
        });

        sock.send(VTableMsg {
            .id = { vtable_id, 0 },
            .vtable = table,
        });
        sock.send(set_component_metadata(component_name, labels));

        // Packet layout: [int64_t time][float values...]
        std::vector<uint8_t> buf(8 + value_bytes, 0);
        size_t num_floats = value_bytes / sizeof(float);

        while (true) {
            int64_t t = system_clock::now().time_since_epoch() / microseconds(1);
            std::memcpy(buf.data(), &t, 8);

            auto* vals = reinterpret_cast<float*>(buf.data() + 8);
            for (size_t i = 0; i < num_floats; ++i) {
                vals[i] = std::sin(static_cast<double>(t) / 1000000.0 + i);
            }

            auto header = PacketHeader {
                .len = static_cast<uint32_t>(4 + buf.size()),
                .ty = PacketType::TABLE,
                .packet_id = { vtable_id, 0 },
                .request_id = 0,
            };
            sock.write_all(&header, sizeof(header));
            sock.write_all(buf.data(), buf.size());
            usleep(1000);
        }
    } catch (const std::exception& e) {
        std::cerr << "[" << component_name << "] error: " << e.what() << std::endl;
    }
}

int main()
try {
    const char* ip = "127.0.0.1";
    const uint16_t port = 2240;

    // 6 components -> 6 TCP connections -> 6 packets per tick.
    // At 1kHz this is 6,000 packets/s. At 250Hz with 400 components
    // it would be 100,000 packets/s -- consider client-batched.cpp instead.
    std::println("Launching per-component writers (6 connections)");

    std::thread t_mag(component_writer, ip, port, 1,
        "vehicle.imu.mag", std::vector<std::string>{"x", "y", "z"},
        PrimType::F32(), std::vector<uint64_t>{3}, uint16_t(12));

    std::thread t_gyro(component_writer, ip, port, 2,
        "vehicle.imu.gyro", std::vector<std::string>{"x", "y", "z"},
        PrimType::F32(), std::vector<uint64_t>{3}, uint16_t(12));

    std::thread t_accel(component_writer, ip, port, 3,
        "vehicle.imu.accel", std::vector<std::string>{"x", "y", "z"},
        PrimType::F32(), std::vector<uint64_t>{3}, uint16_t(12));

    std::thread t_temp(component_writer, ip, port, 4,
        "vehicle.temp", std::vector<std::string>{},
        PrimType::F32(), std::vector<uint64_t>{}, uint16_t(4));

    std::thread t_pressure(component_writer, ip, port, 5,
        "vehicle.pressure", std::vector<std::string>{},
        PrimType::F32(), std::vector<uint64_t>{}, uint16_t(4));

    std::thread t_humidity(component_writer, ip, port, 6,
        "vehicle.humidity", std::vector<std::string>{},
        PrimType::F32(), std::vector<uint64_t>{}, uint16_t(4));

    t_mag.join();
    t_gyro.join();
    t_accel.join();
    t_temp.join();
    t_pressure.join();
    t_humidity.join();
    return 0;
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
}
