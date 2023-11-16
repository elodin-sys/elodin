{
  inputs = {
    paracosm.url = "path:../../.";
    nixpkgs.follows = "paracosm/nixpkgs";
    cargo2nix.follows = "paracosm/cargo2nix";
    rust-overlay.follows = "paracosm/rust-overlay";
    flake-utils.follows = "paracosm/flake-utils";
  };

  outputs = inputs:
    with inputs;
      flake-utils.lib.eachDefaultSystem (
        system: let
          pkgs = import nixpkgs {
            inherit system;
          };
        in rec {
          packages.default = pkgs.stdenv.mkDerivation rec {
            pname = "paracosm-protos";
            version = "0.1.0";
            src = ./protobufs;
            installPhase = ''
              mkdir -p $out/protobuf
              cp -R . $out/protobuf
            '';
          };
        }
      );
}
