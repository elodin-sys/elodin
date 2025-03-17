{pkgs, ...}: {
  environment.systemPackages = with pkgs; [
    libgpiod_1
    dfu-util
    gcc-arm-embedded
    stm32flash
    tio
    fish
    neovim
    git
    uv
    python312
    ripgrep
    (writeShellScriptBin "reset-mcu" ''
      #!/bin/sh -e

      args="$@"
      boot0_pin="gpiochip0 144"
      nrst_pin="gpiochip0 106"

      # if "--bootloader" is in args, then set boot0 to 1
      if echo "$args" | grep -q -- "--bootloader"; then
          gpioset --mode=time --usec=20000 $boot0_pin=1 & sleep 0.001
      fi

      gpioset --mode=time --usec=100 $nrst_pin=0
    '')
    (writeShellScriptBin "flash-mcu" ''
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
    '')
  ];
}
