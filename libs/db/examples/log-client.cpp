///$(which true);FLAGS="--std=c++23";THIS_FILE="$(cd "$(dirname "$0")"; pwd -P)/$(basename "$0")";OUT_FILE="/tmp/build-cache/$THIS_FILE";mkdir -p "$(dirname "$OUT_FILE")";test "$THIS_FILE" -ot "$OUT_FILE" || $(which clang++ || which g++) $FLAGS "$THIS_FILE" -o "$OUT_FILE" || exit $?;exec bash -c "exec -a \"$0\" \"$OUT_FILE\" $([ $# -eq 0 ] || printf ' "%s"' "$@")"

#include <arpa/inet.h>
#include <cstdint>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstddef>
#include <iostream>
#include <print>
#include <sstream>
#include <system_error>
#include <vector>

#include "db.hpp"

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

    void send_raw(const std::vector<uint8_t>& data)
    {
        write_all(data.data(), data.size());
    }

private:
    int fd_ = -1;
};

static constexpr const char* LOG_NAME = "fsw.log";

int main(int argc, char* argv[])
try {
    const char* ip = "127.0.0.1";
    uint16_t port = 2240;

    if (argc >= 2) ip = argv[1];
    if (argc >= 3) port = static_cast<uint16_t>(std::stoi(argv[2]));

    std::println("log-client: connecting to {}:{}", ip, port);
    auto sock = Socket(ip, port);

    // Register the log metadata once at startup
    auto metadata_pkt = set_log_metadata_packet(LOG_NAME);
    sock.send_raw(metadata_pkt);
    std::println("log-client: registered log metadata for '{}'", LOG_NAME);

    // Simulated flight software log messages
    struct SimLog {
        LogLevel level;
        const char* message;
        uint32_t delay_ms;
    };

    // Boot sequence
    static const SimLog boot_sequence[] = {
        { LOG_INFO,  "FSW v2.4.1 starting up",              200 },
        { LOG_INFO,  "Board: Aleph Orin NX rev3",           100 },
        { LOG_DEBUG, "Loading flight parameters from EEPROM", 150 },
        { LOG_INFO,  "IMU driver initialized (ICM-42688)",   100 },
        { LOG_INFO,  "Magnetometer calibrated",              100 },
        { LOG_INFO,  "GPS module initialized, searching...", 200 },
        { LOG_INFO,  "Barometer online: 101325 Pa",          100 },
        { LOG_INFO,  "Actuator self-test: 4/4 passed",       300 },
        { LOG_INFO,  "GPS lock acquired (8 satellites)",     100 },
        { LOG_INFO,  "System ready, entering IDLE state",    500 },
    };

    // Periodic telemetry cycle
    static const SimLog flight_messages[] = {
        { LOG_INFO,  "State: ARMED",                                 500 },
        { LOG_INFO,  "State: FLIGHT",                                200 },
        { LOG_INFO,  "Altitude: 142m AGL, V_up: 82.3 m/s",          300 },
        { LOG_DEBUG, "Nav filter update: sigma_pos=0.8m",            200 },
        { LOG_INFO,  "Altitude: 487m AGL, V_up: 156.1 m/s",         300 },
        { LOG_WARN,  "Battery voltage low: 11.2V (threshold: 11.5V)", 500 },
        { LOG_INFO,  "Altitude: 1024m AGL, max-Q passed",           300 },
        { LOG_DEBUG, "Fin trim adjusted: +0.3 deg",                  200 },
        { LOG_INFO,  "MECO confirmed, coasting phase",              500 },
        { LOG_INFO,  "Apogee detected: 2847m AGL",                  300 },
        { LOG_INFO,  "Drogue deploy command sent",                   100 },
        { LOG_INFO,  "Drogue deployment confirmed",                  200 },
        { LOG_INFO,  "Altitude: 500m AGL, descent rate: -12.4 m/s", 500 },
        { LOG_INFO,  "Main chute deploy at 300m",                    300 },
        { LOG_INFO,  "Main deployment confirmed",                    200 },
        { LOG_WARN,  "GPS signal degraded (3 satellites)",           500 },
        { LOG_INFO,  "GPS reacquired (6 satellites)",                300 },
        { LOG_INFO,  "Touchdown detected, impact: 3.2g",             200 },
        { LOG_INFO,  "State: RECOVERY",                              500 },
        { LOG_INFO,  "Beacon active on 433.92 MHz",                  1000 },
        { LOG_ERROR, "Simulated anomaly: sensor timeout on ADC ch3", 2000 },
        { LOG_WARN,  "Watchdog reset counter: 1",                    1000 },
        { LOG_INFO,  "System nominal after watchdog recovery",       2000 },
    };

    std::println("log-client: sending boot sequence...");
    for (const auto& entry : boot_sequence) {
        send_log(sock, LOG_NAME, entry.level, entry.message);
        usleep(entry.delay_ms * 1000);
    }

    std::println("log-client: entering flight message loop");
    uint32_t cycle = 0;
    while (true) {
        for (const auto& entry : flight_messages) {
            std::ostringstream msg;
            msg << "[cycle " << cycle << "] " << entry.message;
            send_log(sock, LOG_NAME, entry.level, msg.str());
            usleep(entry.delay_ms * 1000);
        }
        cycle++;
    }

    return 0;
} catch (const std::exception& e) {
    std::cerr << "log-client error: " << e.what() << std::endl;
    return 1;
}
