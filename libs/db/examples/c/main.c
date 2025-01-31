#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <errno.h>
#include <assert.h>

#include "./vtable.h"

typedef struct {
    int64_t time;
    float mag[3];
    float gyro[3];
    float accel[3];
    float temp;
    float pressure;
    float humidity;
} sensor_data_t;

ssize_t write_all(int fd, const void *buf, size_t count) {
    size_t written = 0;
    while (written < count) {
        ssize_t n = write(fd, buf + written, count - written);
        written += n;
    }
    return written;
}

int main() {
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
        .time = 0,
        .mag = {0.0, 0.0, 0.0},
        .gyro = {0.0, 0.0, 0.0},
        .accel = {0.0, 0.0, 0.0},
        .temp = 1.0,
        .pressure = 2.0,
        .humidity = 3.0
    };

    // Send initialization message (vtable + component metadata)
    write_all(sock, init_msg, init_msg_len);

    // Send sin wave data continuously
    assert(sizeof(sensor_data_t) == SENSOR_DATA_LEN);
    while (1) {
        write_all(sock, &sensor_data_header, sensor_data_header_len);
        write_all(sock, &sensor_data, SENSOR_DATA_LEN);

        double sin_val = sin((double)sensor_data.time / 100000.0);
        sensor_data.time += 1;
        sensor_data.temp = sin_val;
    }
    close(sock);
    return 0;
}
