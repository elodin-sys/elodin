///$(which true);FLAGS="--std=c++23";THIS_FILE="$(cd "$(dirname "$0")"; pwd -P)/$(basename "$0")";OUT_FILE="/tmp/build-cache/$THIS_FILE";mkdir -p "$(dirname "$OUT_FILE")";test "$THIS_FILE" -ot "$OUT_FILE" || $(which clang++ || which g++) $FLAGS "$THIS_FILE" -o "$OUT_FILE" || exit $?;exec bash -c "exec -a \"$0\" \"$OUT_FILE\" $([ $# -eq 0 ] || printf ' "%s"' "$@")"
//
// Write-only pattern: opt into a silent connection and never read replies.
// Use this only for pure ingest sockets. Query and stream requests are ignored
// by the server on silent connections.

#include <arpa/inet.h>
#include <cstdint>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <print>
#include <system_error>

#include "db.hpp"

using namespace vtable;
using namespace vtable::builder;
using namespace std::chrono;

struct SensorData {
    int64_t time;
    float gyro[3];
    float accel[3];
};

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

int main()
try {
    const char* ip = "127.0.0.1";
    const uint16_t port = 2240;

    std::println("Connecting write-only socket");
    auto sock = Socket(ip, port);

    // This must be the first packet on the connection. The server will apply
    // writes but never send replies, errors, or streams on this socket.
    sock.send(ConnectionSettings { .silent = true });

    auto time = builder::raw_table(0, 8);
    auto table = builder::vtable({
        field<SensorData, &SensorData::gyro>(schema(PrimType::F32(), { 3 }, timestamp(time, component("vehicle.imu.gyro")))),
        field<SensorData, &SensorData::accel>(schema(PrimType::F32(), { 3 }, timestamp(time, component("vehicle.imu.accel")))),
    });

    sock.send(VTableMsg {
        .id = { 2, 0 },
        .vtable = table,
    });
    sock.send(set_component_metadata("vehicle.imu.gyro", { "x", "y", "z" }));
    sock.send(set_component_metadata("vehicle.imu.accel", { "x", "y", "z" }));

    auto sensor_data = SensorData {
        .gyro = { 0.0, 0.0, 0.0 },
        .accel = { 0.0, 0.0, 0.0 },
    };

    while (true) {
        auto table_header = PacketHeader {
            .len = 4 + sizeof(sensor_data),
            .ty = PacketType::TABLE,
            .packet_id = { 2, 0 },
            .request_id = 0,
        };

        sensor_data.time = system_clock::now().time_since_epoch() / microseconds(1);
        sensor_data.gyro[2] = std::sin(static_cast<double>(sensor_data.time) / 1000000.0);
        sensor_data.accel[2] = std::cos(static_cast<double>(sensor_data.time) / 1000000.0);

        sock.write_all(&table_header, sizeof(table_header));
        sock.write_all(&sensor_data, sizeof(sensor_data));
        usleep(1000);
    }

    return 0;
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
}
