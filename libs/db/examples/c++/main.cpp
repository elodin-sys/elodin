#include <iostream>
#include <cmath>
#include <system_error>
#include <array>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include "./vtable.h"

enum class PacketType : uint8_t {
    MSG = 0,
    TABLE = 1,
    TIME_SERIES = 2
};

struct PacketHeader {
    uint64_t len;
    PacketType ty;
    std::array<uint8_t, 3> packet_id;
    uint32_t req_id = 0;
};

class Socket {
public:
    Socket(const char* ip, uint16_t port) {
        fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ < 0) {
            throw std::system_error(errno, std::generic_category(), "Failed to create socket");
        }
        struct sockaddr_in server_addr = {
            .sin_family = AF_INET,
            .sin_port = htons(port),
            .sin_addr.s_addr = inet_addr(ip)
        };

        if (::connect(fd_, reinterpret_cast<struct sockaddr*>(&server_addr), sizeof(server_addr)) < 0) {
            throw std::system_error(errno, std::generic_category(), "Failed to connect");
        }
    }

    ~Socket() {
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    void write_all(const void* data, size_t len) {
        auto ptr = static_cast<const uint8_t*>(data);
        auto remaining = len;

        while (remaining > 0) {
            auto written = write(fd_, ptr, remaining);
            if (written < 0) {
                throw std::system_error(errno, std::generic_category(), "Failed to write");
            }
            ptr += written;
            remaining -= written;
        }
    }

private:
    int fd_ = -1;
};

int main() try {
    auto sock = Socket("127.0.0.1", 2240);

    // send vtable header and data
    PacketHeader vtable_header = {
        .len = vtable_bin_len + 8,
        .ty = PacketType::MSG,
        .packet_id = {224, 0, 0},
    };

    sock.write_all(&vtable_header, sizeof(vtable_header));
    sock.write_all(vtable_bin, vtable_bin_len);

    // send sin wave data continuously  
    double val = 1.0;
    auto table_header = PacketHeader {
      .len = 8 + 8,
      .ty = PacketType::TABLE,
      .packet_id = {1, 0, 0},
    };

    while (true) {
        auto sin_val = std::sin(val);
        sock.write_all(&table_header, sizeof(table_header));
        sock.write_all(&sin_val, sizeof(sin_val));
        val += 0.000001;
    }

    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
}
