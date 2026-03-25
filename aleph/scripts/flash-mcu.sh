#!/bin/sh -e

flash_method="${ALEPH_FLASH_MCU_METHOD:-swd}"
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
    systemctl start --no-block "$bridge_unit" || true
  fi

  rm -rf "$tmpdir"
}

trap cleanup EXIT INT TERM

stop_bridge_if_active() {
  if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet "$bridge_unit"; then
    systemctl stop "$bridge_unit"
    restarted_bridge=1
  fi
}

swd_flash() {
  cat > "$tmpdir/openocd.cfg" << 'OPENOCD_EOF'
source [find interface/cmsis-dap.cfg]
transport select swd
adapter speed 12000
set CHIPNAME stm32h747xit6
source [find target/stm32h7x_dual_bank.cfg]
reset_config none
$_CHIPNAME.cpu0 cortex_m reset_config sysresetreq
OPENOCD_EOF

  openocd -f "$tmpdir/openocd.cfg" \
    -c "program $1 verify reset exit"
}

configure_uart() {
  stty -F "$flash_port" "$flash_baud" parenb -parodd cs8 -cstopb 2>/dev/null || true
}

uart_probe_bootloader() {
  configure_uart
  stm32flash -b "$flash_baud" -m 8e1 "$flash_port"
}

uart_flash_firmware() {
  configure_uart
  stm32flash -b "$flash_baud" -m 8e1 -w "$1" -S "$flash_addr" "$flash_port"
}

# Enter bootloader with timing that ensures stm32flash's 0x7F sync byte
# arrives during the ROM bootloader's interface detection window.
# The STM32H747 ROM scans for UART activity in a narrow window after
# reset; if it misses the sync, it falls to a dual-core handler and hangs.
uart_enter_bootloader() {
  reset-mcu --nrst-low
  sleep 0.5
  reset-mcu --boot0-high
  sleep 0.5
  configure_uart
  reset-mcu --nrst-high
  sleep 0.1
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
    fw_elf="$fw_src"
    fw_bin="$tmpdir/$(basename "$fw_src").bin"
    arm-none-eabi-objcopy -O binary "$fw_src" "$fw_bin"
    ;;
  --bin)
    fw_elf=""
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

case "$flash_method" in
  swd)
    if [ -n "$fw_elf" ]; then
      echo "Flashing $fw_elf via SWD"
      swd_flash "$fw_elf"
    else
      echo "Flashing $fw_bin to $flash_addr via SWD"
      swd_flash "$fw_bin $flash_addr"
    fi
    ;;
  uart)
    if lsof "$flash_port" >/dev/null 2>&1; then
      echo "Serial port is busy: $flash_port" >&2
      lsof "$flash_port" >&2 || true
      exit 1
    fi

    try_probe_bootloader() {
      attempts=0
      max_attempts=5
      while ! uart_probe_bootloader; do
        attempts=$((attempts + 1))
        if [ "$attempts" -ge "$max_attempts" ]; then
          return 1
        fi
        sleep 0.1
      done
      return 0
    }

    echo "Entering STM32 bootloader on $flash_port (GPIO reset)"
    export ALEPH_NRST_METHOD=gpio
    uart_enter_bootloader
    bootloader_entered=1

    if ! try_probe_bootloader; then
      echo "GPIO reset did not reach bootloader, trying I2C expander reset"
      ALEPH_NRST_METHOD=gpio reset-mcu --app || true
      bootloader_entered=0

      export ALEPH_NRST_METHOD=i2c
      reset-mcu --nrst-low
      sleep 1
      reset-mcu --boot0-high
      sleep 3
      configure_uart
      reset-mcu --nrst-high
      sleep 2
      bootloader_entered=1

      if ! try_probe_bootloader; then
        echo "Both GPIO and I2C reset methods failed on $flash_port" >&2
        exit 1
      fi
    fi

    echo "Flashing $fw_bin to $flash_addr via $flash_port"
    uart_flash_firmware "$fw_bin"

    reset-mcu --app
    bootloader_entered=0
    ;;
  *)
    echo "Unknown flash method: $flash_method (expected 'swd' or 'uart')" >&2
    exit 1
    ;;
esac
