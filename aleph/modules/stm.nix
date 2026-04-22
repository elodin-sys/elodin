{
  config,
  lib,
  ...
}: let
  cfg = config.aleph.stm;
in {
  imports = [
    ./sensor-fw.nix
    ./c-blinky.nix
  ];

  options.aleph.stm.firmware = lib.mkOption {
    type = lib.types.enum ["sensor-fw" "c-blinky"];
    default = "sensor-fw";
    description = ''
      Selects which STM32 firmware Aleph deploys and flashes by default. Options are:
      "sensor-fw" which sets up streaming from expansion board IMU sensors into elodin-db.
      "c-blinky" analagous to arduino blink sketch; blinks an LED on the expansion board.
    '';
  };

  config = {
    assertions = [
      {
        assertion = !(config.services.sensor-fw.enable && config.services.c-blinky.enable);
        message = "services.sensor-fw and services.c-blinky are mutually exclusive (STM32 can only load one program at a time).";
      }
    ];

    services.sensor-fw.enable = lib.mkDefault (cfg.firmware == "sensor-fw");
    services.c-blinky.enable = lib.mkDefault (cfg.firmware == "c-blinky");
  };
}
