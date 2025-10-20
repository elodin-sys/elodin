#!/bin/sh -e

reset-mcu --bootloader
while ! dfu-util --list | grep -q '0483:df11'; do
  sleep 0.1
done

fw_bin="$2"

if [ "$1" = "--elf" ]; then
  arm-none-eabi-objcopy -O binary "$fw_bin" "$fw_bin.bin"
  fw_bin="$fw_bin.bin"
elif [ "$1" = "--bin" ]; then
  : # do nothing
elif [ "$1" = "--help" ]; then
  echo "Usage: $0 [--elf|--bin] <firmware file>"
  exit 0
else
  echo "Usage: $0 [--elf|--bin] <firmware file>"
  exit 1
fi

dfu-suffix -c "$fw_bin" || dfu-suffix -a "$fw_bin"
dfu-util -a 0 -d 0483:df11 -s 0x08000000:leave -D "$fw_bin"
