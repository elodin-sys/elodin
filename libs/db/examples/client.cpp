///$(which true);FLAGS="--std=c++23";THIS_FILE="$(cd "$(dirname "$0")"; pwd -P)/$(basename "$0")";OUT_FILE="/tmp/build-cache/$THIS_FILE";mkdir -p "$(dirname "$OUT_FILE")";test "$THIS_FILE" -ot "$OUT_FILE" || $(which clang++ || which g++) $FLAGS "$THIS_FILE" -o "$OUT_FILE" || exit $?;exec bash -c "exec -a \"$0\" \"$OUT_FILE\" $([ $# -eq 0 ] || printf ' "%s"' "$@")"

#include <arpa/inet.h>
#include <cstdint>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <print>
#include <system_error>
#include <vector>
#include <chrono>
#include <thread>

#include "db.hpp"

using namespace vtable;
using namespace vtable::builder;
using namespace std::chrono;

struct SensorData {
    int64_t time;
    float mag[3];
    float gyro[3];
    float accel[3];
    float temp;
    float pressure;
    float humidity;
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

    template<typename T>
    void send(T msg) {
        auto buf = Msg(msg).encode_vec();
        write_all(buf.data(), buf.size());
    }

    size_t read(uint8_t* data, size_t len)
    {
        auto res = ::read(fd_, data, len);
        if (res < 0) {
            throw std::system_error(errno, std::generic_category(), "Failed to read");
        }
        return res;
    }

private:
    int fd_ = -1;
};

// Thread function to read from a socket connection
void reader_thread_func(const char* ip, uint16_t port) {
    try {
        std::println("Reader thread: connecting to {}:{}", ip, port);
        auto read_sock = Socket(ip, port);
        read_sock.send(VTableStream {
            .id = { 2, 0 },
        });

        // Read loop
        while (true) {
            auto data = std::vector<uint8_t>(1024);
            auto len = read_sock.read(data.data(), data.size());
            if (len == 0) {
                std::println("Reader thread: connection closed");
                break;
            }

            std::println("Reader thread received data: {} bytes", len);
        }
    } catch (const std::exception& e) {
        std::cerr << "Reader thread error: " << e.what() << std::endl;
    }
    std::println("Reader thread: exiting");
}

int main()
try {
    const char* ip = "127.0.0.1";
    const uint16_t port = 2240;

    // Connect the main socket for writing
    std::println("Main thread: connecting writer socket");
    auto sock = Socket(ip, port);
    auto time = builder::raw_table(0, 8);
    auto table = builder::vtable({
        field<SensorData, &SensorData::mag>(schema(PrimType::F32(), { 3 }, timestamp(time, pair(1, "mag")))),
        field<SensorData, &SensorData::gyro>(schema(PrimType::F32(), { 3 }, timestamp(time, pair(1, "gyro")))),
        field<SensorData, &SensorData::accel>(schema(PrimType::F32(), { 3 }, timestamp(time, pair(1, "accel")))),
        field<SensorData, &SensorData::temp>(schema(PrimType::F32(), {}, timestamp(time, pair(1, "temp")))),
        field<SensorData, &SensorData::pressure>(schema(PrimType::F32(), {}, timestamp(time, pair(1, "pressure")))),
        field<SensorData, &SensorData::humidity>(schema(PrimType::F32(), {}, timestamp(time, pair(1, "humidity")))),
    });

    sock.send(VTableMsg {
        .id = { 2, 0 },
        .vtable = table,
    });
    sock.send(set_component_name("mag"));
    sock.send(set_component_name("gyro"));
    sock.send(set_component_name("accel"));
    sock.send(set_component_name("temp"));
    sock.send(set_component_name("pressure"));
    sock.send(set_component_name("humidity"));

    // Start the reader thread
    std::println("Main thread: starting reader thread");
    std::thread reader(reader_thread_func, ip, port);

    auto sensor_data = SensorData {
        .mag = { 0.0, 0.0, 0.0 },
        .gyro = { 0.0, 0.0, 0.0 },
        .accel = { 0.0, 0.0, 0.0 },
        .temp = 0.0,
        .pressure = 2.0,
        .humidity = 3.0
    };
    while (true) {
        // send sin wave data continuously
        auto table_header = PacketHeader {
            .len = 4 + sizeof(sensor_data),
            .ty = PacketType::TABLE,
            .packet_id = { 2, 0 },
            .request_id = 0,
        };

        sensor_data.time = system_clock::now().time_since_epoch() / microseconds(1);
        sensor_data.temp = std::sin(static_cast<double>(sensor_data.time) / 1000000.0);

        std::println("writing {}, {} bytes", sizeof(table_header), sizeof(sensor_data));
        sock.write_all(&table_header, sizeof(table_header));
        sock.write_all(&sensor_data, sizeof(sensor_data));
        usleep(1000);
    }

    // We should never reach here, but if we do:
    reader.join();
    return 0;
} catch (const std::exception& e) {
    std::cerr << "Main thread error: " << e.what() << std::endl;
    return 1;
}
