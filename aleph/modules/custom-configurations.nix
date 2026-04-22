# Sample Aleph configuration presets for STM consumed by `flake.nix` when
# building named `nixosConfigurations` variants. These are added to base config.
#
rec {
  # Config preset for blinking led (similar to blink sketch)
  preset-c-blinky = {...}: {
    imports = [./stm.nix];

    aleph.stm.firmware = "c-blinky";
  };

  # Config preset for streaming sensor data from expansion board to elodin-db
  preset-sensor-fw = {...}: {
    imports = [./stm.nix];

    aleph.stm.firmware = "sensor-fw";
  };

  # Sensor-FW config preset + u-blox SAM-M10Q support
  preset-m10q = {...}: {
    imports = [preset-sensor-fw];

    services.sensor-fw.gps.model = "m10q";
  };

  # Sensor-FW config preset + u-blox NEO-M9N support
  preset-m9n = {...}: {
    imports = [preset-sensor-fw];

    services.sensor-fw.gps.model = "m9n";
  };
}
