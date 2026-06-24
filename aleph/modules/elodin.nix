{
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.elodin;
in {
  options.services.elodin = {
    enable = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Whether to install the Elodin simulation CLI and packaged examples.
      '';
    };

    package = lib.mkOption {
      type = lib.types.package;
      default = pkgs.elodin-cli;
      description = "The Elodin simulation CLI package to install.";
    };

    examplesPackage = lib.mkOption {
      type = lib.types.package;
      default = pkgs.elodin-examples;
      description = "Packaged Elodin examples for on-device testing.";
    };

    enableRenderer = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Whether to enable graphics runtime support for the headless sensor camera renderer.
      '';
    };
  };

  config = lib.mkIf cfg.enable (let
    # EGM08 spherical-harmonic gravity coefficients used by the cube-sat example.
    # The SDK downloads these into el._get_cache_dir() on first run; pre-fetch
    # them so the example runs offline and reproducibly on-device.
    egm08C = pkgs.fetchurl {
      url = "https://assets.elodin.systems/assets/C_normal.npy";
      hash = "sha256-sZKrOEr/1MzAs7OMR5DreWcu/16E3hurVoFNwLIUTWU=";
    };
    egm08S = pkgs.fetchurl {
      url = "https://assets.elodin.systems/assets/S_normal.npy";
      hash = "sha256-Y6ophu4G7hitgrO9QxYYmpyC/1u2nkbTPqWYpo71ojk=";
    };
    elodinRunExample = pkgs.writeShellScriptBin "elodin-run-example" ''
      set -euo pipefail

      export PATH="${lib.makeBinPath [
        pkgs.coreutils
        pkgs.gnugrep
        pkgs.iproute2
        pkgs.systemd
      ]}:$PATH"

      usage() {
        echo "Usage: elodin-run-example {ball|sensor-camera|video-stream|drone|cube-sat|three-body}"
      }

      if [ "$#" -ne 1 ]; then
        usage
        exit 2
      fi

      example="$1"
      examples_root="${cfg.examplesPackage}"
      db_path="/db/example-$example"
      work_dir="/tmp/elodin-example-$example"
      run_timeout="''${ELODIN_EXAMPLE_TIMEOUT:-60s}"

      case "$example" in
        ball)
          main="$examples_root/ball/main.py"
          run_timeout="''${ELODIN_EXAMPLE_TIMEOUT:-30s}"
          ;;
        sensor-camera)
          main="$examples_root/sensor-camera/main.py"
          run_timeout="''${ELODIN_EXAMPLE_TIMEOUT:-90s}"
          export ELODIN_SENSOR_CAMERA_MAX_TICKS="''${ELODIN_SENSOR_CAMERA_MAX_TICKS:-1800}"
          export ELODIN_SENSOR_CAMERA_DB="$db_path"
          ;;
        video-stream)
          main="$examples_root/video-stream/main.py"
          work_dir="/tmp/elodin-example-video-stream"
          run_timeout="''${ELODIN_EXAMPLE_TIMEOUT:-45s}"
          ;;
        drone)
          main="$examples_root/drone/main.py"
          run_timeout="''${ELODIN_EXAMPLE_TIMEOUT:-90s}"
          ;;
        cube-sat)
          main="$examples_root/cube-sat/main.py"
          run_timeout="''${ELODIN_EXAMPLE_TIMEOUT:-240s}"
          # Seed the EGM08 gravity coefficients into the SDK cache so the
          # example does not need network access at runtime.
          cache_dir="''${HOME:-/root}/.cache/elodin-cli"
          mkdir -p "$cache_dir"
          cp -n ${egm08C} "$cache_dir/C_normal.npy" || true
          cp -n ${egm08S} "$cache_dir/S_normal.npy" || true
          ;;
        three-body)
          main="$examples_root/three-body/main.py"
          run_timeout="''${ELODIN_EXAMPLE_TIMEOUT:-30s}"
          ;;
        *)
          usage
          exit 2
          ;;
      esac

      port_in_use() {
        ss -H -tln 'sport = :2240' | grep -q .
      }

      stop_elodin_db() {
        systemctl stop elodin-db-default.service 2>/dev/null || true
        systemctl list-units --type=service --state=active --plain --no-legend 'elodin-db@*.service' |
          while read -r unit _; do
            [ -n "$unit" ] && systemctl stop "$unit" || true
          done

        for _ in $(seq 1 40); do
          if ! port_in_use; then
            return 0
          fi
          sleep 0.25
        done

        echo "ERROR: port 2240 is still in use after stopping elodin-db services" >&2
        ss -tlnp 'sport = :2240' >&2 || true
        exit 1
      }

      mkdir -p /db /root/.local/share/cli "$work_dir"
      rm -rf "$db_path"
      rm -rf "$work_dir/video-stream-db"
      log_file="$work_dir/$example.log"

      echo "Stopping system elodin-db services so the simulation can bind :2240..."
      stop_elodin_db

      echo "Running $example with DB path $db_path"
      cd "$work_dir"

      set +e
      ELODIN_DB_PATH="$db_path" timeout --signal=INT --kill-after=10s "$run_timeout" "${cfg.package}/bin/elodin" run "$main" 2>&1 | tee "$log_file"
      status="''${PIPESTATUS[0]}"
      set -e

      if [ "$example" = "video-stream" ] && [ -d "$work_dir/video-stream-db" ]; then
        rm -rf "$db_path"
        mv "$work_dir/video-stream-db" "$db_path"
      fi

      if [ "$status" -ne 0 ]; then
        if { [ "$status" -eq 124 ] || [ "$status" -eq 130 ] || [ "$status" -eq 143 ]; } && [ -d "$db_path" ]; then
          echo "$example stopped after timeout; DB was created, so the smoke run succeeded."
        else
          echo "ERROR: $example failed with status $status" >&2
          exit "$status"
        fi
      fi

      if [ "$example" = "sensor-camera" ] && ! grep -q "verification: PASS" "$log_file"; then
        echo "ERROR: sensor-camera did not report verification: PASS" >&2
        exit 1
      fi

      echo "DB saved at $db_path"
    '';
  in {
    # jetpack's graphics module already sets hardware.graphics.package to
    # l4t-3d-core (the Jetson Vulkan/GL driver) when graphics is enabled, so we
    # only need to ensure graphics is on for the headless renderer.
    hardware.graphics.enable = lib.mkDefault cfg.enableRenderer;

    environment.systemPackages = [
      cfg.package
      cfg.examplesPackage
      elodinRunExample
    ];

    systemd.services.elodin-assets-seed = {
      description = "Seed the writable Elodin asset root with packaged defaults";
      wantedBy = ["multi-user.target"];
      path = with pkgs; [
        coreutils
        gnugrep
      ];
      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
      };
      script = ''
        mkdir -p /root/assets

        # Migrate the tiny auto-created empty skybox manifest so the packaged
        # default skybox is selectable. Preserve real customer manifests.
        manifest=/root/assets/skyboxes/manifest.ron
        if [ -f "$manifest" ] && [ "$(wc -c < "$manifest")" -le 128 ] && ! grep -q "desert_night" "$manifest"; then
          rm -f "$manifest"
        fi

        cp -rn --no-preserve=mode,ownership ${pkgs.elodin-assets}/. /root/assets/
        chmod -R u+rwX /root/assets
      '';
    };

    systemd.tmpfiles.rules = [
      "d /root 0700 root root - -"
      "d /root/.local 0755 root root - -"
      "d /root/.local/share 0755 root root - -"
      "d /root/.local/share/cli 0755 root root - -"
      "L+ /root/examples - - - - ${cfg.examplesPackage}"
    ];
  });
}
