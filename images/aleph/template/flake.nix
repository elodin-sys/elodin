{
  nixConfig = {
    extra-substituters = ["http://ci-arm1.elodin.dev:5000"];
    extra-trusted-public-keys = [
      "builder-cache-1:q7rDGIQgkg1nsxNEg7mHN1kEDuxPmJhQpuIXCCwLj8E="
    ];
  };

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.11";
    aleph.url = "github:elodin-sys/elodin/main?dir=images/aleph";
    aleph.inputs.nixpkgs.follows = "nixpkgs";
  };
  outputs = {
    nixpkgs,
    aleph,
    self,
    ...
  }: rec {
    nixosModules.default = {config, ...}: {
      imports = with aleph.nixosModules; [
        "${nixpkgs}/nixos/modules/profiles/minimal.nix"

        # hardware modules
        jetpack # core module required to make jetpack-nixos work
        hardware # aleph specific hardware module, brings in the forked-kernel and device tree
        sd-image # module that allows building sd-card images compatible with aleph

        # networking modules
        usb-eth # sets up the usb ethernet gadget present on aleph
        wifi # sets up wifi using iwd

        minimal # strips down nixos to an even more minimal set of defaults

        # elodin-db and integrations
        elodin-db # brings in elodin-db as a default service
        aleph-serial-bridge # pushes sensor data into elodin-db from the default expansion board firmware
        tegrastats-bridge # pushes telemetry form the Orin SoC into elodin-db (i.e cpu usage, temps, etc)

        # default tooling
        aleph-setup # a setup tool that guides you through setting up wifi and a user on first login
        aleph-dev # a default set of packages like cuda, opencv, and git that make developing on aleph easier

        # default fsw
        mekf # a basic attitude mekf that runs on the sensor data from the expansion board
      ];

      # overlays required to get elodin and nvidia packages
      nixpkgs.overlays = [
        aleph.overlays.default
        aleph.overlays.jetpack
      ];

      system.stateVersion = "24.11";

      i18n.supportedLocales = [(config.i18n.defaultLocale + "/UTF-8")];

      services.openssh.enable = true; # enable ssh
      services.openssh.settings = {
        PasswordAuthentication = true;
        PermitRootLogin = "yes";
      };
      security.sudo.wheelNeedsPassword = false;
      nix.settings.trusted-users = ["@wheel"];
    };
    # sets up two different nixos systems default and installer
    # installer is setup to be flashed to a usb drive, and contains the
    # aleph-installer tool. This tool lets you install the system to the nvme
    # drive
    nixosConfigurations = aleph.lib.installerSystem nixosModules.default;
    packages.aarch64-linux = {
      default = nixosConfigurations.installer.config.system.build.sdImage;
      toplevel = nixosConfigurations.default.config.system.build.toplevel; # the toplevel config, this allows
      # you to use the deploy.sh script
    };
  };
}
