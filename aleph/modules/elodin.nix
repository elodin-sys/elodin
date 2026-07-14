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
      description = "Whether to install the Elodin simulation CLI.";
    };

    examples = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Whether to seed the packaged examples and their default assets. The
        shared asset root is created regardless of this flag so customers can
        deploy their own assets.
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
      description = "Packaged Elodin examples, symlinked into /var/lib/elodin.";
    };

    assetsPackage = lib.mkOption {
      type = lib.types.package;
      default = pkgs.elodin-assets;
      description = "Default assets seeded into ELODIN_ASSETS.";
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
    assetsDir = "/var/lib/elodin/assets";
  in {
    # jetpack's graphics module already sets hardware.graphics.package to
    # l4t-3d-core (the Jetson Vulkan/GL driver) when graphics is enabled, so we
    # only need to ensure graphics is on for the headless renderer.
    hardware.graphics.enable = lib.mkDefault cfg.enableRenderer;

    environment.systemPackages = [cfg.package];

    # Every user resolves assets from the shared, writable asset root.
    environment.sessionVariables.ELODIN_ASSETS = assetsDir;

    # Pre-create each user's CLI data dir so the first-launch notice does not
    # short-circuit an interactive `elodin run`.
    environment.loginShellInit = ''
      mkdir -p "''${XDG_DATA_HOME:-$HOME/.local/share}/cli"
    '';

    systemd.services.elodin-assets-seed = lib.mkIf cfg.examples {
      description = "Seed the shared Elodin asset root with packaged defaults";
      wantedBy = ["multi-user.target"];
      path = with pkgs; [
        coreutils
        findutils
        gnugrep
      ];
      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
      };
      script = ''
        install -d -m 2775 -g wheel ${assetsDir}

        # Migrate the tiny auto-created empty skybox manifest so the packaged
        # default skybox is selectable. Preserve real customer manifests.
        manifest=${assetsDir}/skyboxes/manifest.ron
        if [ -f "$manifest" ] && [ "$(wc -c < "$manifest")" -le 128 ] && ! grep -q "desert_night" "$manifest"; then
          rm -f "$manifest"
        fi

        cp -rn --no-preserve=mode,ownership ${cfg.assetsPackage}/. ${assetsDir}/
        chgrp -R wheel ${assetsDir}
        chmod -R g+rwX ${assetsDir}
        find ${assetsDir} -type d -exec chmod g+s {} +
      '';
    };

    systemd.tmpfiles.rules =
      [
        "d /var/lib/elodin 0755 root root - -"
        # setgid + group-writable so any wheel user can add assets.
        "d ${assetsDir} 2775 root wheel - -"
      ]
      ++ lib.optionals cfg.examples [
        "L+ /var/lib/elodin/examples - - - - ${cfg.examplesPackage}"
      ];
  });
}
