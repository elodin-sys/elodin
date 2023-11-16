{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/aba830385baac22d06a22085f5b5baeb88c88b46";
    cargo2nix.url = "github:cargo2nix/cargo2nix/release-0.11.0";
    rust-overlay.url = "github:oxalica/rust-overlay/b7a041430733fccaa1ffc3724bb9454289d0f701";
    flake-utils.url = "github:numtide/flake-utils";
    get-flake.url = "github:ursi/get-flake";
  };

  outputs = {
    self,
    nixpkgs,
    get-flake,
    flake-utils,
    ...
  } @ inputs: let
    types = get-flake ./libs/paracosm-types;
  in rec {
    packages.rust-overrides = pkgs: let
      protos = types.packages.${pkgs.system}.default;
    in
      pkgs.rustBuilder.overrides.all
      ++ [
        (pkgs.rustBuilder.rustLib.makeOverride {
          name = "paracosm-types";
          overrideAttrs = drv: {
            propagatedBuildInputs =
              drv.propagatedBuildInputs
              or []
              ++ [
                protos
                pkgs.protobuf
                pkgs.libiconv
              ];
            propagatedNativeBuildInputs = drv.propagatedNativeBuildInputs or [] ++ [
              pkgs.libiconv
            ];
            ELODIN_PROTOBUFS = "${protos}/protobufs";
          };
        })
      ];
  };
}
