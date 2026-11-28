#!/bin/sh -eu
# Flash ExpressLRS onto the onboard ESP32-C3 through the debug probe:
# load the self-test onto the H747, put it in ELRS download passthrough
# (probe UART6 <-> H747 UART4 <-> ESP32-C3), then run esptool on UART6.
#
# usage: ./flash.sh UART3_PORT UART6_PORT [PROBE_SERIAL]
#   macOS: /dev/cu.usbmodem<N>3 and /dev/cu.usbmodem<N>5 (same <N>)
#   Linux: /dev/serial/by-id/usb-*<serial>*-if02 and ...-if04

[ $# -ge 2 ] || { sed -n 2,9p "$0"; exit 2; }
U3=$1; U6=$2; SERIAL=${3:-}
cd "$(dirname "$0")"
OPENOCD=${OPENOCD:-openocd}
ESPTOOL=${ESPTOOL:-esptool}
command -v "$ESPTOOL" >/dev/null 2>&1 || ESPTOOL=esptool.py

ocd() {
    "$OPENOCD" -f interface/cmsis-dap.cfg \
        -c "cmsis_dap_vid_pid 0x1209 0x0001; cmsis_dap_backend usb_bulk; ${SERIAL:+adapter serial $SERIAL;} transport select swd; adapter speed 4000" \
        -f target/stm32h7x_dual_bank.cfg -c "$1"
}

# Wait for the self-test summary on UART3, then send one key.
key() {
    python3 - "$U3" "$1" <<'PY'
import os, sys, termios, time
fd = os.open(sys.argv[1], os.O_RDWR | os.O_NOCTTY)
a = termios.tcgetattr(fd); a[0] = a[1] = a[3] = 0
a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL; a[4] = a[5] = termios.B115200
a[6][termios.VMIN] = 0; a[6][termios.VTIME] = 1; termios.tcsetattr(fd, termios.TCSANOW, a)
buf, t = b'', time.time()
while time.time() - t < 30 and b'passed,' not in buf: buf += os.read(fd, 4096)
if b'passed,' not in buf: sys.exit('no self-test output on ' + sys.argv[1])
os.write(fd, sys.argv[2].encode()); t = time.time()
while time.time() - t < 1: buf += os.read(fd, 4096)
print('\n'.join(l for l in buf.decode(errors='replace').splitlines() if 'passed,' in l or 'passthrough' in l))
PY
}

echo "== self-test onto the H747"
ocd "program ../selftest/build/selftest.elf verify reset exit" 2>&1 | grep -E 'Verified|rror' || true
key p
echo "== ESP32-C3 download over $U6"
"$ESPTOOL" --chip esp32c3 -p "$U6" -b 115200 --before no_reset --after no_reset write_flash -z \
    --flash_mode dio --flash_freq 40m --flash_size detect \
    0x0000 build/bootloader.bin 0x8000 build/partitions.bin 0xe000 build/boot_app0.bin 0x10000 build/firmware.bin
echo "== reset (ESP32-C3 boots ELRS)"
ocd "init; reset run; exit" >/dev/null 2>&1
echo "done. Flash the flight firmware onto the H747 next; the self-test is still on it."
