# Flight software

Index of this folder. Follow the linked README for build steps and how a program is used.

## Aleph Orin Flight Computer

The Jetson Orin, and the STM32 image it flashes over the board-to-board link. NixOS modules and deploy are in [aleph/README.md](../aleph/README.md).

### Basic

The computer, a first-light MCU image, and the bridges that land logs and SoC telemetry in Elodin-DB.

| Dir | What |
| --- | --- |
| [aleph-setup](aleph-setup/) | First-boot wizard: WiFi and a user account |
| [aleph-status](aleph-status/) | Terminal status for the OS, hardware, sensors, and SoC |
| [c-blinky](c-blinky/README.md) | STM32 blink image with UART logs |
| [openocd](openocd/) | Host SWD flash of an STM32 ELF (`flash.sh`, `aleph.cfg`) |
| [serial-bridge](serial-bridge/) | MCU UART frames into Elodin-DB |
| [tegrastats-bridge](tegrastats-bridge/) | Orin CPU, GPU, and temperature into Elodin-DB |

### Advanced

Sensors, estimation, control, and video.

| Dir | What |
| --- | --- |
| [sensor-fw](sensor-fw/README.md) | STM32 firmware: IMUs, magnetometer, barometer, and GPS, out over UART and onto the SD card |
| [blackbox](blackbox/) | IMU, GPS, and compass records shared by `sensor-fw` and `serial-bridge` |
| [mekf](mekf/README.md) | Attitude filter on the Orin, and the spec for the sensor pipeline |
| [lqr](lqr/) | Attitude controller. Torque from the MEKF estimate |
| [msp-osd](msp-osd/README.md) | MSP DisplayPort OSD fed from Elodin-DB |
| [gstreamer](gstreamer/README.md) | GStreamer plugin that writes H.264 into Elodin-DB |
| [video-streamer](video-streamer/README.md) | A video file, re-encoded to AV1, into Elodin-DB |
| [rtsp-streamer](rtsp-streamer/) | An RTSP H.264 camera into Elodin-DB |

## Aleph Flight Controller

Bring-up for revision 6. Built images are in `fc-firmware.zip`.

| Dir | What |
| --- | --- |
| [probe](probe/README.md) | Debug probe: CMSIS-DAP plus the UART3 and UART6 consoles. `blinky` is first light |
| [selftest](selftest/README.md) | H747 self-test and ExpressLRS passthrough |
| [elrs](elrs/) | ExpressLRS 4.1.0 for the onboard ESP32-C3 (`./elrs/build.sh`, `./elrs/flash.sh`) |

Flash the probe (`./probe/flash.sh dfu dap`), then the self-test (`./selftest/flash.sh`), then ExpressLRS. Probe host tests: `./probe/tests/run.sh`.

## Other

| Dir | What |
| --- | --- |
| [roci](roci/README.md) | Flight-software framework. `mekf` and `lqr` use `roci/adcs` |
| [udp_component_broadcast](udp_component_broadcast/README.md) | UDP relay of Elodin-DB components between two simulations |
