#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <errno.h>
#include "./vtable.h"

enum PacketType {
    PACKET_MSG = 0,
    PACKET_TABLE = 1,
    PACKET_TIME_SERIES = 2
};

struct PacketHeader {
    uint64_t len;
    uint8_t ty;
    uint8_t packet_id[3];
    uint32_t req_id;
};

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

    struct PacketHeader vtable_header = {
        .len = vtable_bin_len + 8,
        .ty = PACKET_MSG,
        .packet_id = {224, 0, 0},
        .req_id = 0
    };

    // Send vtable header and data
    if (write(sock, &vtable_header, sizeof(vtable_header)) != sizeof(vtable_header) ||
        write(sock, vtable_bin, vtable_bin_len) != vtable_bin_len) {
        perror("failed to send vtable");
        return 1;
    }


    // Send sin wave data continuously
    double val = 1.0;
    while (1) {
        double sin_val = sin(val);
        
        struct PacketHeader table_header = {
            .len = 8 + 8,
            .ty = PACKET_TABLE,
            .packet_id = {1, 0, 0},
            .req_id = 0
        };

        if (write(sock, &table_header, sizeof(table_header)) != sizeof(table_header) ||
            write(sock, &sin_val, sizeof(sin_val)) != sizeof(sin_val)) {
            perror("failed to send data");
            return 1;
        }

        val += 0.000001;
    }

    close(sock);
    return 0;
}
