{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/aba830385baac22d06a22085f5b5baeb88c88b46";
    cargo2nix.url = "github:cargo2nix/cargo2nix/release-0.11.0";
    rust-overlay.url = "github:oxalica/rust-overlay/b7a041430733fccaa1ffc3724bb9454289d0f701";
    flake-utils.url = "github:numtide/flake-utils";
    get-flake.url = "github:ursi/get-flake";
    nix2container.url = "github:nlewo/nix2container";
  };

  outputs = {
    self,
    nixpkgs,
    ...
  } @ inputs: rec {
    packages.rust-overrides = pkgs:
      pkgs.rustBuilder.overrides.all
      ++ [
        (pkgs.rustBuilder.rustLib.makeOverride {
          name = "paracosm-types";
          overrideAttrs = drv: {
            propagatedNativeBuildInputs =
              drv.propagatedNativeBuildInputs
              or []
              ++ [
                pkgs.buildPackages.protobuf
              ]
              ++ pkgs.lib.optional pkgs.buildPackages.hostPlatform.isDarwin [
                pkgs.buildPackages.libiconv
              ];
          };
        })
        (pkgs.rustBuilder.rustLib.makeOverride {
          name = "ring";
          overrideAttrs = drv: {
            propagatedNativeBuildInputs =
              (drv.propagatedNativeBuildInputs
                or [])
              ++ pkgs.lib.optional pkgs.buildPackages.hostPlatform.isDarwin [
                pkgs.buildPackages.libiconv
              ];
          };
        })
        (pkgs.rustBuilder.rustLib.makeOverride {
          name = "libsqlite3-sys";
          overrideAttrs = drv: {
            propagatedNativeBuildInputs =
              (drv.propagatedNativeBuildInputs
                or [])
              ++ pkgs.lib.optional pkgs.buildPackages.hostPlatform.isDarwin [
                pkgs.buildPackages.libiconv
              ];
          };
        })
        (pkgs.rustBuilder.rustLib.makeOverride {
          name = "alsa-sys";
          overrideAttrs = drv: {
            buildInputs =
              (drv.buildInputs
                or [])
              ++ [pkgs.alsa-lib];
          };
        })
      ];
  };
}
