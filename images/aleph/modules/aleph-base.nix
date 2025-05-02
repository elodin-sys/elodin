{...}: {
  nix.settings.extra-substituters = [
    "https://cache.nixos.org"
    "http://ci-arm1.elodin.dev:5000"
  ];
  nix.settings.trusted-public-keys = [
    "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
    "builder-cache-1:q7rDGIQgkg1nsxNEg7mHN1kEDuxPmJhQpuIXCCwLj8E="
  ];
  nix.settings.experimental-features = ["nix-command" "flakes"];
  nix.settings.trusted-users = ["@wheel"];
  security.pam.loginLimits = [
    {
      domain = "*";
      type = "soft";
      item = "nofile";
      value = "65536";
    }
    {
      domain = "*";
      type = "hard";
      item = "nofile";
      value = "524288";
    }
  ];
  environment.etc."elodin-version" = let
    cargoToml = builtins.fromTOML (builtins.readFile ../../../Cargo.toml);
    version = cargoToml.workspace.package.version;
  in {
    text = version;
    enable = true;
  };
}
