{
  pkgs,
  lib,
  name,
  cfg,
  timeoutSec ? "90s",
}: let
  flashMcu = pkgs.flash-mcu;

  flashScript = pkgs.writeShellApplication {
    name = "${name}-flash-service";
    runtimeInputs = [
      pkgs.coreutils
      flashMcu
    ];
    text = ''
      marker="$STATE_DIRECTORY/last-flashed-package"
      desired_signature="${lib.concatStringsSep "|" [
        (toString cfg.package)
        (toString flashMcu)
        cfg.serialPort
        (toString cfg.bootloaderBaudRate)
        cfg.boot0GpioChip
        (toString cfg.boot0GpioLine)
        cfg.resetGpioChip
        (toString cfg.resetGpioLine)
      ]}"
      desired_firmware="${cfg.package}/firmware.bin"

      if [ -f "$marker" ] && [ "$(cat "$marker")" = "$desired_signature" ]; then
        echo "${name} firmware already flashed from $desired_firmware"
        exit 0
      fi

      echo "Flashing ${name} firmware from $desired_firmware"
      flash-mcu --bin "$desired_firmware"
      printf '%s\n' "$desired_signature" > "$marker"
    '';
  };

  serviceAttrs = {
    description = "Flash ${name} onto the Aleph STM32H7";
    wantedBy = lib.mkIf cfg.autostart ["multi-user.target"];
    before = ["serial-bridge.service"];
    restartIfChanged = true;
    restartTriggers = [cfg.package];

    serviceConfig = {
      Type = "oneshot";
      RemainAfterExit = true;
      User = "root";
      Group = "root";
      StateDirectory = "mcu-flash";
      ExecStart = "${flashScript}/bin/${name}-flash-service";
      TimeoutStartSec = timeoutSec;
      Environment = [
        "ALEPH_FLASH_MCU_METHOD=uart"
        "ALEPH_FLASH_MCU_ADDR=0x08000000"
        "ALEPH_FLASH_MCU_PORT=${cfg.serialPort}"
        "ALEPH_FLASH_MCU_BAUD=${toString cfg.bootloaderBaudRate}"
        "ALEPH_FLASH_MCU_BRIDGE_UNIT=serial-bridge.service"
        "ALEPH_BOOT0_GPIOCHIP=${cfg.boot0GpioChip}"
        "ALEPH_BOOT0_GPIOLINE=${toString cfg.boot0GpioLine}"
        "ALEPH_NRST_GPIOCHIP=${cfg.resetGpioChip}"
        "ALEPH_NRST_GPIOLINE=${toString cfg.resetGpioLine}"
      ];
      StandardOutput = "journal";
      StandardError = "journal";
    };
  };
in {
  inherit flashScript serviceAttrs;
}
