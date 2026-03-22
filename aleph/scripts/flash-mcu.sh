#!/bin/sh -e

flash_port="${ALEPH_FLASH_MCU_PORT:-/dev/ttyTHS1}"
flash_baud="${ALEPH_FLASH_MCU_BAUD:-19200}"
flash_addr="${ALEPH_FLASH_MCU_ADDR:-0x08000000}"
bridge_unit="${ALEPH_FLASH_MCU_BRIDGE_UNIT:-serial-bridge.service}"

restarted_bridge=0
bootloader_entered=0
tmpdir="$(mktemp -d)"

usage() {
  echo "Usage: $0 [--elf|--bin] <firmware file>"
}

cleanup() {
  if [ "$bootloader_entered" -eq 1 ]; then
    reset-mcu --app || true
  fi

  if [ "$restarted_bridge" -eq 1 ] && command -v systemctl >/dev/null 2>&1; then
    # Queue the bridge restart without waiting on the current flash unit to finish.
    systemctl start --no-block "$bridge_unit" || true
  fi

  rm -rf "$tmpdir"
}

trap cleanup EXIT INT TERM

configure_uart() {
  # Work around stm32flash failing to apply termios directly on ttyTHS*.
  stty -F "$flash_port" raw "$flash_baud" parenb -parodd cs8 -cstopb 2>/dev/null \
    || stty -F "$flash_port" "$flash_baud" parenb -parodd cs8 -cstopb 2>/dev/null \
    || true
}

probe_bootloader() {
  configure_uart
  stm32flash -b "$flash_baud" -m 8e1 "$flash_port"
}

flash_firmware() {
  configure_uart
  stm32flash -b "$flash_baud" -m 8e1 -w "$fw_bin" -S "$flash_addr" "$flash_port"
}

stop_bridge_if_active() {
  if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet "$bridge_unit"; then
    systemctl stop "$bridge_unit"
    restarted_bridge=1
  fi
}

if [ "$#" -ne 2 ]; then
  usage >&2
  exit 1
fi

fw_src="$2"
if [ ! -f "$fw_src" ]; then
  echo "Firmware file not found: $fw_src" >&2
  exit 1
fi

case "$1" in
  --elf)
    fw_bin="$tmpdir/$(basename "$fw_src").bin"
    arm-none-eabi-objcopy -O binary "$fw_src" "$fw_bin"
    ;;
  --bin)
    fw_bin="$tmpdir/$(basename "$fw_src")"
    cp "$fw_src" "$fw_bin"
    ;;
  --help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac

stop_bridge_if_active

if lsof "$flash_port" >/dev/null 2>&1; then
  echo "Serial port is busy: $flash_port" >&2
  lsof "$flash_port" >&2 || true
  exit 1
fi

echo "Entering STM32 bootloader on $flash_port"
reset-mcu --bootloader
bootloader_entered=1

attempts=0
max_attempts=5
while ! probe_bootloader; do
  attempts=$((attempts + 1))
  if [ "$attempts" -ge "$max_attempts" ]; then
    echo "Timed out waiting for STM32 bootloader on $flash_port" >&2
    exit 1
  fi
  sleep 1
done

echo "Flashing $fw_bin to $flash_addr via $flash_port"
flash_firmware

reset-mcu --app
bootloader_entered=0
