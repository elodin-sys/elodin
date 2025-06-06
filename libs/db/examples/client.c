///$(which true);THIS_FILE="$(cd "$(dirname "$0")"; pwd -P)/$(basename "$0")";OUT_FILE="/tmp/build-cache/$THIS_FILE";mkdir -p "$(dirname "$OUT_FILE")";test "$THIS_FILE" -ot "$OUT_FILE" || $(which clang || which gcc) "$THIS_FILE" -o "$OUT_FILE" || exit $?;exec bash -c "exec -a \"$0\" \"$OUT_FILE\" $([ $# -eq 0 ] || printf ' "%s"' "$@")"

#include <arpa/inet.h>
#include <assert.h>
#include <math.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

enum PacketType {
    MSG = 0,
    TABLE = 1,
    TIME_SERIES = 2
};

typedef struct packet_header_t packet_header_t;
struct packet_header_t {
    uint32_t len;
    uint8_t ty;
    uint8_t packet_id[2];
    uint8_t request_id;
};

typedef struct sensor_data_t sensor_data_t;
struct sensor_data_t {
    float mag[3];
    float gyro[3];
    float accel[3];
    float temp;
    float pressure;
    float humidity;
};

ssize_t write_all(int fd, const void* buf, size_t count)
{
    size_t written = 0;
    while (written < count) {
        ssize_t n = write(fd, buf + written, count - written);
        written += n;
    }
    return written;
}

int main()
{
    // Create TCP socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("Failed to create socket");
        return 1;
    }

    // Connect to server
    struct sockaddr_in server_addr = {
        .sin_family = AF_INET,
        .sin_port = htons(2240),
        .sin_addr.s_addr = inet_addr("127.0.0.1")
    };

    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("failed to connect");
        return 1;
    }

    sensor_data_t sensor_data = {
        .mag = { 0.0, 0.0, 0.0 },
        .gyro = { 0.0, 0.0, 0.0 },
        .accel = { 0.0, 0.0, 0.0 },
        .temp = 0.0,
        .pressure = 2.0,
        .humidity = 3.0
    };
    packet_header_t sensor_data_header = {
        .len = 4 + sizeof(sensor_data),
        .ty = TABLE,
        .packet_id = { 1, 0 },
        .request_id = 0,
    };

    // Send sin wave data continuously
    double time = 0.0;
    while (1) {
        write_all(sock, &sensor_data_header, sizeof(sensor_data_header));
        write_all(sock, &sensor_data, sizeof(sensor_data));

        time += 1.0 / 1000.0;
        sensor_data.temp = sin(time);
        usleep(1000);
    }
    close(sock);
    return 0;
}
