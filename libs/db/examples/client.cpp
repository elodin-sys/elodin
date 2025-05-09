///$(which true);FLAGS="--std=c++23";THIS_FILE="$(cd "$(dirname "$0")"; pwd -P)/$(basename "$0")";OUT_FILE="/tmp/build-cache/$THIS_FILE";mkdir -p "$(dirname "$OUT_FILE")";test "$THIS_FILE" -ot "$OUT_FILE" || $(which clang++ || which g++) $FLAGS "$THIS_FILE" -o "$OUT_FILE" || exit $?;exec bash -c "exec -a \"$0\" \"$OUT_FILE\" $([ $# -eq 0 ] || printf ' "%s"' "$@")"

#include <arpa/inet.h>
#include <cstdint>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <print>
#include <span>
#include <system_error>
#include <vector>

#include "db.hpp"

using namespace vtable;

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

int main()
try {
    auto sock = Socket("127.0.0.1", 2240);

    // create a new real time stream
    // sending this message will create a new "stream" that sends out all new data
    // as it comes in
    auto stream = Stream {
        .behavior = StreamBehavior::RealTime()
    };
    auto stream_msg = Msg<Stream>(stream);
    auto stream_msg_buf = stream_msg.encode_vec();

    // prints out the Stream msg that will be sent to the DB
    // std::println("stream encoded {}", stream_msg_buf);

    sock.write_all(stream_msg_buf.data(), stream_msg_buf.size());

    // create a new msg stream
    auto msg_stream = MsgStream {
        .msg_id = { 6, 166 }
    };
    auto msg_stream_msg_buf = Msg(msg_stream).encode_vec();

    // prints out the MsgStream msg that will be sent to the DB
    // std::println("msg stream encoded {}", msg_stream_msg_buf);

    sock.write_all(msg_stream_msg_buf.data(), msg_stream_msg_buf.size());

    // sends vtable for sensor data

    auto table = builder::vtable({
        builder::field<SensorData, &SensorData::mag>(builder::schema(PrimType::F32(), { 3 }, builder::pair(1, "mag"))),
        builder::field<SensorData, &SensorData::gyro>(builder::schema(PrimType::F32(), { 3 }, builder::pair(1, "gyro"))),
        builder::field<SensorData, &SensorData::accel>(builder::schema(PrimType::F32(), { 3 }, builder::pair(1, "accel"))),
        builder::field<SensorData, &SensorData::temp>(builder::schema(PrimType::F32(), {}, builder::pair(1, "temp"))),
        builder::field<SensorData, &SensorData::pressure>(builder::schema(PrimType::F32(), {}, builder::pair(1, "pressure"))),
        builder::field<SensorData, &SensorData::humidity>(builder::schema(PrimType::F32(), {}, builder::pair(1, "humidity"))),
    });

    auto vtable_msg = Msg(VTableMsg {
        .id = { 2, 0 },
        .vtable = table,
    });
    auto vtable_msg_encoded = vtable_msg.encode_vec();
    sock.write_all(vtable_msg_encoded.data(), vtable_msg_encoded.size());

    sock.send(set_component_name("mag"));
    sock.send(set_component_name("gyro"));
    sock.send(set_component_name("accel"));
    sock.send(set_component_name("temp"));
    sock.send(set_component_name("pressure"));
    sock.send(set_component_name("humidity"));

    auto sensor_data = SensorData {
        .mag = { 0.0, 0.0, 0.0 },
        .gyro = { 0.0, 0.0, 0.0 },
        .accel = { 0.0, 0.0, 0.0 },
        .temp = 1.0,
        .pressure = 2.0,
        .humidity = 3.0
    };

    double time = 0;
    while (true) {
        auto data = std::vector<uint8_t>(256);
        auto len = sock.read(data.data(), data.size());
        if (len == 0) {
            return 0;
        }
        // std::println("received data {}", std::span(data.data(), len));

        // send sin wave data continuously
        auto table_header = PacketHeader {
            .len = 4 + sizeof(sensor_data),
            .ty = PacketType::TABLE,
            .packet_id = { 2, 0 },
            .request_id = 0,
        };

        time += 1.0;
        sensor_data.temp = std::sin(static_cast<double>(time / 100000.0));

        sock.write_all(&table_header, sizeof(table_header));
        sock.write_all(&sensor_data, sizeof(sensor_data));
    }

    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
}
