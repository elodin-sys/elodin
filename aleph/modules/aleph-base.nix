{...}: {
  nix.settings.extra-substituters = [
    "https://cache.nixos.org"
    "s3://elodin-nix-cache?region=us-west-2"
  ];
  nix.settings.trusted-public-keys = [
    "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
    "elodin-cache-1:vvbmIQvTOjcBjIs8Ri7xlT2I3XAmeJyF5mNlWB+fIwM="
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
  # Always create podman group, even if podman isn't enabled.
  users.groups.podman = {};
  environment.etc."elodin-version" = let
    cargoToml = builtins.fromTOML (builtins.readFile ../../Cargo.toml);
    version = cargoToml.workspace.package.version;
  in {
    text = version;
    enable = true;
  };
}
