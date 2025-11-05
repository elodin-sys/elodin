///$(which true);FLAGS="--std=c++23";THIS_FILE="$(cd "$(dirname "$0")"; pwd -P)/$(basename "$0")";OUT_FILE="/tmp/build-cache/$THIS_FILE";mkdir -p "$(dirname "$OUT_FILE")";test "$THIS_FILE" -ot "$OUT_FILE" || $(which clang++ || which g++) $FLAGS "$THIS_FILE" -o "$OUT_FILE" || exit $?;exec bash -c "exec -a \"$0\" \"$OUT_FILE\" $([ $# -eq 0 ] || printf ' "%s"' "$@")"

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
#include <thread>
#include <vector>
#include <iomanip>

#include "db.hpp"

using namespace vtable;
using namespace vtable::builder;
using namespace std::chrono;

struct RocketData {
    int64_t time; // elodin time
    double commanded_deflect;
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

void print_bytes_hex(const uint8_t* data, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        std::cout << std::hex            // switch to hex format
                  << std::setw(2)        // always print 2 digits
                  << std::setfill('0')   // pad with leading zeros
                  << static_cast<int>(data[i]) << ' ';
    }
    std::cout << std::dec << std::endl;  // switch back to decimal
}

//
const uint8_t ID[2] = { 4, 0 };

// Thread function to read from a socket connection
void reader_thread_func(const char* ip, uint16_t port)
{
    try {
        std::println("Reader thread: connecting to {}:{}", ip, port);
        auto read_sock = Socket(ip, port);
        read_sock.send(VTableStream {
            .id = { ID[0], ID[1] },
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
            print_bytes_hex(data.data(), len);
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

    std::println("Global data size {}", sizeof(RocketData));
    // Connect the main socket for writing
    std::println("Main thread: connecting writer socket");
    auto sock = Socket(ip, port);
    auto time = builder::raw_table(0, 8);

    auto table = builder::vtable({
        // field<RocketData, &RocketData::commanded_deflect>(schema(PrimType::F32(), { }, timestamp(time, component("rocket.commanded_deflect")))),
        field<RocketData, &RocketData::commanded_deflect>(schema(PrimType::F64(), { }, timestamp(time, component("rocket.fin_control_trim")))),
    });

    sock.send(VTableMsg {
        .id = { ID[0], ID[1] },
        .vtable = table,
    });

    // Start the reader thread
    std::println("Main thread: starting reader thread");
    std::thread reader(reader_thread_func, ip, port);

    auto rocket_data = RocketData {
    .commanded_deflect = 0.0,
    };
    std::println("Size of packet header {}", sizeof(PacketHeader));

    while (true) {
        // send sin wave data continuously
        auto table_header = PacketHeader {
            .len = 4 + sizeof(rocket_data),
            .ty = PacketType::TABLE,
            .packet_id = { ID[0], ID[1] },
            .request_id = 0,
        };
        int64_t t = system_clock::now().time_since_epoch() / microseconds(1);
        // rocket_data.time = t - 6000000;
        rocket_data.time = t;
        // std::println("time orig  {}\ntime after {}", t, rocket_data.time);
        rocket_data.commanded_deflect = 0.1 * std::sin(static_cast<double>(t) / 1000000.0);

        sock.write_all(&table_header, sizeof(table_header));
        sock.write_all(&rocket_data, sizeof(rocket_data));
        usleep(1000);
    }

    // We should never reach here, but if we do:
    reader.join();
    return 0;
} catch (const std::exception& e) {
    std::cerr << "Main thread error: " << e.what() << std::endl;
    return 1;
}
