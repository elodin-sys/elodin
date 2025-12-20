{
  pkgs,
  rustToolchain,
  lib,
  pkg-config,
  systemd,
  ...
}: let
  pname = "msp-osd";
  workspaceToml = builtins.fromTOML (builtins.readFile ../../Cargo.toml);
  version = workspaceToml.workspace.package.version;

  common = import ./common.nix {inherit lib;};
  src = common.src;
in
  pkgs.rustPlatform.buildRustPackage {
    inherit pname version src;

    cargoLock = {
      lockFile = ../../Cargo.lock;
      allowBuiltinFetchGit = true;
    };

    buildAndTestSubdir = "fsw/msp-osd";

    nativeBuildInputs = [
      pkg-config
      (rustToolchain pkgs)
    ];

    buildInputs = [
      systemd
    ];

    doCheck = false;

    meta = with lib; {
      description = "MSP OSD Service - MSP DisplayPort OSD for VTX";
      homepage = "https://github.com/elodin-sys/elodin";
      license = licenses.mit;
      maintainers = with maintainers; [];
      mainProgram = "msp-osd";
    };
  }
