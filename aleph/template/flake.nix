{
  nixConfig = {
    extra-substituters = ["https://elodin-nix-cache.s3.us-west-2.amazonaws.com"];
    extra-trusted-public-keys = [
      "elodin-cache-1:vvbmIQvTOjcBjIs8Ri7xlT2I3XAmeJyF5mNlWB+fIwM="
    ];
    fallback = true;
  };

  inputs = {
    aleph.url = "github:elodin-sys/elodin/main?dir=aleph";
    nixpkgs.follows = "aleph/nixpkgs";
  };
  outputs = {
    nixpkgs,
    aleph,
    self,
    ...
  }: rec {
    system = "aarch64-linux";
    nixosModules.default = {config, ...}: {
      imports = with aleph.nixosModules; [
        "${nixpkgs}/nixos/modules/profiles/minimal.nix"

        # hardware modules
        jetpack # core module required to make jetpack-nixos work
        hardware # aleph specific hardware module, brings in the forked-kernel and device tree
        fs # module that allows building sd-card images compatible with aleph

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
        aleph-base # a set of default configuration options that make developing on aleph easier
        aleph-dev # a default set of packages like cuda, opencv, and git that make developing on aleph easier

        # default fsw
        mekf # a basic attitude mekf that runs on the sensor data from the expansion board
        msp-osd # MSP DisplayPort OSD for FPV goggles, displays attitude from MEKF

        # udp component broadcast (for sharing component data between flight computers)
        udp-component-broadcast # broadcasts component data over UDP
        udp-component-receive # receives UDP broadcasts and writes to local elodin-db
      ];

      # overlays required to get elodin and nvidia packages
      nixpkgs.overlays = [
        aleph.overlays.default
        aleph.overlays.jetpack
      ];

      system.stateVersion = "25.05";

      i18n.supportedLocales = [(config.i18n.defaultLocale + "/UTF-8")];

      services.openssh.enable = true; # enable ssh
      services.openssh.settings = {
        PasswordAuthentication = true;
        PermitRootLogin = "yes";
      };
      security.sudo.wheelNeedsPassword = false;
      nix.settings.trusted-users = ["@wheel"];

      # Customize the kernel source (current options are default and no_otg)
      aleph.kernel.source = "default";

      # Elodin-DB configuration
      # services.elodin-db = {
      #   enable = true;                    # Enable elodin-db (default: true)
      #   autostart = true;                 # Set to false to configure but not auto-start
      #   dbUniqueOnBoot = true;            # Create unique db folder on each boot
      #   openFirewall = true;              # Open ports 2240 and 2248
      # };

      # Enable MSP OSD service (uses MEKF attitude output by default)
      services.msp-osd = {
        enable = true;

        # Override input mappings for a different data source
        inputs = {
          position = {
            component = "my.PositionComponent";
            x = 0;
            y = 1;
            z = 2;
          };
          orientation = {
            component = "my.OrientationComponent";
            qx = 3;
            qy = 4;
            qz = 5;
            qw = 6;
          };
          velocity = {
            component = "my.VelocityComponent";
            x = 0;
            y = 1;
            z = 2;
          };
          # Optional: target position for tracking indicator (e.g., another aircraft)
          # target = {
          #   component = "target.world_pos";
          #   x = 4;
          #   y = 5;
          #   z = 6;
          # };
        };

        # Other overridable options:
        # autostart = true;         # Set to false to configure but not auto-start
        # coordinateFrame = "enu";  # or "ned" (default)
        # serialPort = "/dev/ttyTHS7";
        # baudRate = 115200;
        # mode = "serial";  # or "debug" for terminal output
        # refreshRateHz = 20.0;
        # osdRows = 18;
        # osdCols = 50;

        # Horizon display calibration:
        # charAspectRatio = 1.5;    # Character height/width ratio (1.5 for Walksnail Avatar)
        # pitchScale = 5.0;         # Degrees per row (~camera_vfov / osd_rows)

        # Auto-recording (Walksnail Avatar):
        # autoRecord = true;        # Start VTX recording when service starts
      };

      # UDP Component Broadcast service - broadcasts component data to other flight computers
      # Uncomment and configure to enable broadcasting
      # services.udp-component-broadcast = {
      #   enable = true;
      #   component = "bdx.world_pos";      # Component to broadcast (required)
      #   rename = "target.world_pos";      # Rename for receivers (optional)
      #   sourceId = "bdx-plane";           # Identifier for this broadcaster
      #   broadcastRate = 20.0;             # Broadcast rate in Hz
      #   broadcastPort = 41235;            # UDP broadcast port
      #   # autostart = true;                # Set to false to configure but not auto-start
      #   # dbAddr = "127.0.0.1:2240";      # Elodin-DB address (default)
      #   # verbose = false;                 # Enable verbose logging
      # };

      # UDP Component Receive service - receives broadcasts from other flight computers
      # Uncomment and configure to enable receiving
      # services.udp-component-receive = {
      #   enable = true;
      #   listenPort = 41235;               # UDP port to listen on
      #   # autostart = true;               # Set to false to configure but not auto-start
      #   # filter = ["target.world_pos"];  # Only accept specific components (empty = all)
      #   # dbAddr = "127.0.0.1:2240";      # Elodin-DB address to write to
      #   # timestampMode = "sender";       # Timestamp mode: "sender", "local", or "monotonic"
      #   # verbose = false;                # Enable verbose logging
      # };
    };
    # sets up two different nixos systems default and installer
    # installer is setup to be flashed to a usb drive, and contains the
    # aleph-installer tool. This tool lets you install the system to the nvme
    # drive
    nixosConfigurations = {
      default = nixpkgs.lib.nixosSystem {
        inherit system;
        modules = [nixosModules.default];
      };
    };
    packages.aarch64-linux = {
      sdimage = aleph.packages.aarch64-linux.sdimage;
      # the toplevel config, this allows you to use the deploy.sh script:
      default = nixosConfigurations.default.config.system.build.toplevel;
    };
  };
}
