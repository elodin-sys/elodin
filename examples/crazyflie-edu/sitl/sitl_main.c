/**
 * sitl_main.c - SITL (Software-In-The-Loop) Main Entry Point
 *
 * This is the SITL wrapper that:
 *   1. Receives sensor data from Elodin via UDP (port 9003)
 *   2. Calls user_main_loop() at 500 Hz
 *   3. Sends motor commands back via UDP (port 9002)
 *
 * The same user_code.c runs here and on Crazyflie hardware.
 *
 * Build: ./build.sh
 * Run:   (automatically spawned by main.py)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <signal.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>

#include "../user_code.h"

// =============================================================================
// Configuration
// =============================================================================

// UDP ports (matching Betaflight SITL convention)
#define PORT_SENSORS  9003   // Elodin -> SITL (sensor data)
#define PORT_MOTORS   9002   // SITL -> Elodin (motor commands)

// Default host
#define DEFAULT_HOST "127.0.0.1"

// Timeout for receiving sensor packets (ms)
#define RECV_TIMEOUT_MS 1000

// =============================================================================
// Packet Structures (must match Python side)
// =============================================================================

/**
 * Sensor packet from Elodin (Python -> C)
 *
 * Layout: timestamp(8) + gyro(12) + accel(12) + buttons(4) = 36 bytes
 */
typedef struct __attribute__((packed)) {
    double timestamp;       // Simulation time (seconds)
    float gyro[3];          // Angular velocity (rad/s), body frame
    float accel[3];         // Acceleration (g units), body frame
    uint8_t is_armed;       // Armed state
    uint8_t button_blue;    // Blue button
    uint8_t button_yellow;  // Yellow button
    uint8_t button_green;   // Green button
    uint8_t button_red;     // Red button
    uint8_t _padding[3];    // Padding to 8-byte boundary
} sensor_packet_t;

/**
 * Motor packet to Elodin (C -> Python)
 *
 * Layout: timestamp(8) + motors(8) = 16 bytes
 */
typedef struct __attribute__((packed)) {
    double timestamp;       // Echo back timestamp for sync
    uint16_t motor_pwm[4];  // Motor PWM commands (0-65535)
} motor_packet_t;

// Verify packet sizes at compile time
_Static_assert(sizeof(sensor_packet_t) == 40, "sensor_packet_t must be 40 bytes");
_Static_assert(sizeof(motor_packet_t) == 16, "motor_packet_t must be 16 bytes");

// =============================================================================
// Global State
// =============================================================================

static volatile bool g_running = true;
static int g_sensor_sock = -1;
static int g_motor_sock = -1;
static struct sockaddr_in g_elodin_addr;
static uint64_t g_step_count = 0;

// =============================================================================
// Signal Handler
// =============================================================================

static void signal_handler(int sig) {
    (void)sig;
    printf("\n[SITL] Received signal, shutting down...\n");
    g_running = false;
}

// =============================================================================
// Socket Setup
// =============================================================================

static int setup_sensor_socket(uint16_t port) {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        perror("[SITL] Failed to create sensor socket");
        return -1;
    }

    // Allow address reuse
    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // Bind to receive sensor packets
    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons(port),
        .sin_addr.s_addr = INADDR_ANY,
    };

    if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("[SITL] Failed to bind sensor socket");
        close(sock);
        return -1;
    }

    // Set receive timeout
    struct timeval tv = {
        .tv_sec = RECV_TIMEOUT_MS / 1000,
        .tv_usec = (RECV_TIMEOUT_MS % 1000) * 1000,
    };
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    return sock;
}

static int setup_motor_socket(const char* host, uint16_t port) {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        perror("[SITL] Failed to create motor socket");
        return -1;
    }

    // Store destination address
    g_elodin_addr.sin_family = AF_INET;
    g_elodin_addr.sin_port = htons(port);
    if (inet_pton(AF_INET, host, &g_elodin_addr.sin_addr) <= 0) {
        fprintf(stderr, "[SITL] Invalid host address: %s\n", host);
        close(sock);
        return -1;
    }

    return sock;
}

// =============================================================================
// Main Loop
// =============================================================================

static void run_sitl_loop(void) {
    user_state_t state = {0};
    sensor_packet_t sensor_pkt;
    motor_packet_t motor_pkt;

    printf("[SITL] Entering main loop...\n");

    while (g_running) {
        // Receive sensor packet (blocking with timeout)
        ssize_t recv_len = recvfrom(g_sensor_sock, &sensor_pkt, sizeof(sensor_pkt),
                                    0, NULL, NULL);

        if (recv_len < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // Timeout - check if still running
                continue;
            }
            perror("[SITL] Receive error");
            break;
        }

        if (recv_len != sizeof(sensor_pkt)) {
            fprintf(stderr, "[SITL] Invalid packet size: %zd (expected %zu)\n",
                    recv_len, sizeof(sensor_pkt));
            continue;
        }

        // Update state from sensor packet
        state.time = sensor_pkt.timestamp;
        state.dt = CONTROL_LOOP_DT;
        state.sensors.gyro.x = sensor_pkt.gyro[0];
        state.sensors.gyro.y = sensor_pkt.gyro[1];
        state.sensors.gyro.z = sensor_pkt.gyro[2];
        state.sensors.accel.x = sensor_pkt.accel[0];
        state.sensors.accel.y = sensor_pkt.accel[1];
        state.sensors.accel.z = sensor_pkt.accel[2];
        state.is_armed = sensor_pkt.is_armed != 0;
        state.button_blue = sensor_pkt.button_blue != 0;
        state.button_yellow = sensor_pkt.button_yellow != 0;
        state.button_green = sensor_pkt.button_green != 0;
        state.button_red = sensor_pkt.button_red != 0;

        // Clear motor commands before user code
        state.motor_pwm[0] = 0;
        state.motor_pwm[1] = 0;
        state.motor_pwm[2] = 0;
        state.motor_pwm[3] = 0;

        // Call user code
        user_main_loop(&state);

        // Build motor response packet
        motor_pkt.timestamp = sensor_pkt.timestamp;
        motor_pkt.motor_pwm[0] = state.motor_pwm[0];
        motor_pkt.motor_pwm[1] = state.motor_pwm[1];
        motor_pkt.motor_pwm[2] = state.motor_pwm[2];
        motor_pkt.motor_pwm[3] = state.motor_pwm[3];

        // Send motor packet
        ssize_t sent = sendto(g_motor_sock, &motor_pkt, sizeof(motor_pkt), 0,
                              (struct sockaddr*)&g_elodin_addr, sizeof(g_elodin_addr));
        if (sent < 0) {
            perror("[SITL] Send error");
        }

        g_step_count++;

        // Periodic status (every 500 steps = 1 second at 500 Hz)
        if (g_step_count % 500 == 0) {
            printf("[SITL] t=%.1fs | armed=%d blue=%d | PWM=[%u,%u,%u,%u]\n",
                   state.time, state.is_armed, state.button_blue,
                   state.motor_pwm[0], state.motor_pwm[1],
                   state.motor_pwm[2], state.motor_pwm[3]);
        }
    }
}

// =============================================================================
// Main Entry Point
// =============================================================================

int main(int argc, char* argv[]) {
    const char* host = DEFAULT_HOST;
    uint16_t sensor_port = PORT_SENSORS;
    uint16_t motor_port = PORT_MOTORS;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            host = argv[++i];
        } else if (strcmp(argv[i], "--sensor-port") == 0 && i + 1 < argc) {
            sensor_port = (uint16_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--motor-port") == 0 && i + 1 < argc) {
            motor_port = (uint16_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --host HOST         Elodin host (default: %s)\n", DEFAULT_HOST);
            printf("  --sensor-port PORT  Sensor receive port (default: %d)\n", PORT_SENSORS);
            printf("  --motor-port PORT   Motor send port (default: %d)\n", PORT_MOTORS);
            printf("  -h, --help          Show this help\n");
            return 0;
        }
    }

    printf("================================================\n");
    printf("Crazyflie SITL - Software In The Loop\n");
    printf("================================================\n");
    printf("Host: %s\n", host);
    printf("Sensor port: %d (receive)\n", sensor_port);
    printf("Motor port: %d (send)\n", motor_port);
    printf("Control rate: %d Hz\n", CONTROL_LOOP_HZ);
    printf("================================================\n");

    // Set up signal handler
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Create sockets
    g_sensor_sock = setup_sensor_socket(sensor_port);
    if (g_sensor_sock < 0) {
        return 1;
    }

    g_motor_sock = setup_motor_socket(host, motor_port);
    if (g_motor_sock < 0) {
        close(g_sensor_sock);
        return 1;
    }

    printf("[SITL] Sockets ready, waiting for Elodin...\n");

    // Initialize user code
    user_init();

    // Run main loop
    run_sitl_loop();

    // Cleanup
    printf("[SITL] Shutting down after %llu steps\n", (unsigned long long)g_step_count);
    close(g_sensor_sock);
    close(g_motor_sock);

    return 0;
}

