{
  pkgs,
  rustToolchain,
  ...
}: let
  pname = "memserve";
  version = "0.1.0";
in
  pkgs.rustPlatform.buildRustPackage {
    inherit pname version;

    src = ../../docs/memserve;

    cargoLock = {
      lockFile = ../../docs/memserve/Cargo.lock;
    };

    nativeBuildInputs = [
      (rustToolchain pkgs)
    ];

    doCheck = false;

    meta = with pkgs.lib; {
      description = "Memory server for documentation";
    };
  }
