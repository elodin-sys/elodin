{
  nixConfig = {
    extra-substituters = ["https://elodin-nix-cache.s3.us-west-2.amazonaws.com"];
    extra-trusted-public-keys = [
      "elodin-cache-1:vvbmIQvTOjcBjIs8Ri7xlT2I3XAmeJyF5mNlWB+fIwM="
    ];
    fallback = true;
  };
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    nixpkgs-unstable.url = "github:nixos/nixpkgs/nixos-unstable";
    jetpack.url = "github:anduril/jetpack-nixos/2c98c9d6c326d67ae5f4909db61238d31352e18c";
    rust-overlay.url = "github:oxalica/rust-overlay";

    jetpack.inputs.nixpkgs.follows = "nixpkgs";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
    # Monorepo: parent is the Elodin flake root. For an `aleph/`-only tree use
    # `github:elodin-sys/elodin` (root) instead of path:..
    elodin = {
      url = "path:..";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.rust-overlay.follows = "rust-overlay";
    };
  };
  outputs = {
    self,
    nixpkgs,
    nixpkgs-unstable,
    jetpack,
    rust-overlay,
    elodin,
  }: let
    # Separate Aleph's fixed NixOS target from host-scoped flake outputs
    alephSystem = "aarch64-linux";
    supportedSystems = ["x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin"];
    forAllSystems = f:
    # Equivalent to using flake-utils
      nixpkgs.lib.genAttrs supportedSystems (system: f nixpkgs.legacyPackages.${system});

    rustToolchain = p: p.rust-bin.fromRustupToolchainFile ../rust-toolchain.toml;
    gitJSONOverlay = builtins.fromJSON (builtins.readFile ./gitrepos.json);
    gitReposOverlay = final: prev: {
      cudaPackages = prev.cudaPackages_12; # CUDA 12 for JP6
      nvidia-jetpack = prev.nvidia-jetpack6.overrideScope (jetpackFinal: jetpackPrev: {
        gitRepos =
          jetpackPrev.gitRepos
          // (final.lib.mapAttrs (_: info:
            final.fetchgit {
              inherit (info) url rev sha256;
            })
          gitJSONOverlay);
      });
    };
    # JAX pulls jax-cuda12-plugin when cudaSupport is true; that hits NCCL stubs
    # jetpack marks unavailable on Orin. Elodin-py only needs CPU JAX on Aleph.
    jetsonJaxCpuOverlay = final: prev:
      prev.lib.optionalAttrs (prev.stdenv.hostPlatform.system == "aarch64-linux") {
        python313Packages = prev.python313Packages.override {
          overrides = self: super: {
            jax = super.jax.override {cudaSupport = false;};
          };
        };
      };
    overlay = final: prev:
      (prev.lib.packagesFromDirectoryRecursive {
        directory = ./pkgs;
        callPackage = path: args: final.callPackage path (args // {inherit rustToolchain;});
      })
      // (rust-overlay.overlays.default final prev)
      // (gitReposOverlay final prev)
      // (jetsonJaxCpuOverlay final prev)
      // (elodin.overlays.default final prev);

    elodinPythonStack = [
      ./modules/elodin-py.nix
      ({lib, ...}: {
        services.elodin-py.enable = lib.mkDefault true;
        services.elodin-py.editor = false;
      })
    ];

    baseModules = {
      default = defaultModule;
      jetpack = jetpack.nixosModules.default;
      usb-eth = ./modules/usb-eth.nix;
      hardware = ./modules/hardware.nix;
      minimal = ./modules/minimal.nix;
      fs = ./modules/fs.nix;
      aleph-base = ./modules/aleph-base.nix;
      aleph-setup = ./modules/aleph-setup.nix;
      wifi = ./modules/wifi.nix;
    };
    fswModules = {
      elodin-db = ./modules/elodin-db.nix;
      aleph-serial-bridge = ./modules/aleph-serial-bridge.nix;
      tegrastats-bridge = ./modules/tegrastats-bridge.nix;
      mekf = ./modules/mekf.nix;
      msp-osd = ./modules/msp-osd.nix;
      udp-component-broadcast = ./modules/udp-component-broadcast.nix;
      udp-component-receive = ./modules/udp-component-receive.nix;
      elodinsink = ./modules/elodinsink.nix;
    };
    stmModules = {
      sensor-fw = ./modules/sensor-fw.nix;
      c-blinky = ./modules/c-blinky.nix;
    };
    stmConfigurationModules = {
      stm = ./modules/stm.nix;
    };
    devModules = {
      aleph-cuda = ./modules/aleph-cuda.nix;
      aleph-dev = ./modules/aleph-dev.nix;
    };
    defaultModule = {config, ...}: {
      imports = [
        "${nixpkgs}/nixos/modules/profiles/minimal.nix"
      ];
      nixpkgs.overlays = [
        jetpack.overlays.default
        overlay
      ];
      system.stateVersion = "25.05";
      i18n.supportedLocales = [(config.i18n.defaultLocale + "/UTF-8")];
      services.openssh.settings.PasswordAuthentication = true;
      services.openssh.enable = true;
      services.openssh.settings.PermitRootLogin = "yes";
      # services.nvpmodel.enable = false;
      # services.nvfancontrol.enable = false;
      security.sudo.wheelNeedsPassword = false;
      users.users.root.password = "root";
    };

    # A set of presets that flash different programs onto the expansion board STM
    configurationPresets = import ./modules/custom-configurations.nix;

    installerSystem = module: let
      baseNixosConfig = nixpkgs.lib.nixosSystem {
        system = alephSystem;
        modules = [module];
      };
    in
      nixpkgs.lib.nixosSystem {
        system = alephSystem;
        modules = [
          module
          ({...}: {
            imports = [./modules/installer.nix];
            aleph.sd.enable = true;
            aleph.installer.system = baseNixosConfig.config.system.build.toplevel;
          })
        ];
      };
    baseConfigurationModules =
      builtins.attrValues baseModules
      ++ builtins.attrValues fswModules
      ++ builtins.attrValues devModules
      ++ elodinPythonStack;

    # Create a NixOS system configuration from the shared base modules plus any extra modules.
    mkConfiguration = extraModules:
      nixpkgs.lib.nixosSystem {
        system = alephSystem;
        modules = baseConfigurationModules ++ extraModules;
      };
    customConfigurations = {
      base = mkConfiguration [];
      c-blinky = mkConfiguration [configurationPresets.preset-c-blinky];
      sensor-fw = mkConfiguration [configurationPresets.preset-sensor-fw];
      m10q = mkConfiguration [configurationPresets.preset-m10q];
      m9n = mkConfiguration [configurationPresets.preset-m9n];
    };
  in
    {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          buildInputs = with pkgs; [
            just
            zstd
          ];
        };
      });
      apps = forAllSystems (pkgs: {
        deploy = {
          type = "app";
          program = "${pkgs.writeScript "deploy" (builtins.readFile ./deploy.sh)}";
        };
      });
    }
    // rec {
      nixosModules =
        baseModules
        // fswModules
        // stmModules
        // stmConfigurationModules
        // devModules
        // configurationPresets
        // {elodin-py = ./modules/elodin-py.nix;};
      overlays.default = overlay;
      overlays.jetpack = jetpack.overlays.default;
      overlays.gitRepos = gitReposOverlay;
      nixosConfigurations =
        {
          # .#nixosConfigurations.default.config.system.build.toplevel will apply the
          # sensor-fw sketch onto the expansion board STM by default
          default = mkConfiguration [configurationPresets.preset-sensor-fw];
          installer = installerSystem ({...}: {
            imports = builtins.attrValues baseModules;
          });
        }
        // customConfigurations;
      packages.aarch64-linux = {
        toplevel = nixosConfigurations.default.config.system.build.toplevel;
        sdimage = nixosConfigurations.installer.config.system.build.sdImage;
      };
      packages.x86_64-linux = let
        flash-cross = jetpack.nixosConfigurations."orin-nx-devkit".extendModules {
          modules = [
            {nixpkgs.buildPlatform.system = "x86_64-linux";}
          ];
        };
      in {
        flash-uefi = nixpkgs.legacyPackages.x86_64-linux.runCommand "flash-uefi" {} ''
          mkdir -p $out
          cp ${flash-cross.config.system.build.legacyFlashScript}/bin/flash-orin-nx-devkit $out/flash-uefi
          sed -i '46i\cp ${./tegra234-mb2-bct-misc-p3767-0000.dts} bootloader/generic/BCT/tegra234-mb2-bct-misc-p3767-0000.dts' $out/flash-uefi
          chmod +x $out/flash-uefi
        '';
      };
      lib.installerSystem = installerSystem;
      templates.default = {
        path = ./template;
        description = "custom nixos image";
      };
    };
}
