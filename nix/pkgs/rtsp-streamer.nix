{
  pkgs,
  lib,
  rustToolchain,
  pkg-config,
  ...
}: let
  pname = "rtsp-streamer";
  workspaceToml = builtins.fromTOML (builtins.readFile ../../Cargo.toml);
  version = workspaceToml.workspace.package.version;

  common = pkgs.callPackage ./common.nix {};
  src = common.src;
in
  pkgs.rustPlatform.buildRustPackage {
    inherit pname version src;

    cargoLock = {
      lockFile = ../../Cargo.lock;
      allowBuiltinFetchGit = true;
    };

    buildAndTestSubdir = "fsw/rtsp-streamer";

    nativeBuildInputs = [
      (rustToolchain pkgs)
      pkg-config
    ];

    buildInputs = lib.optionals pkgs.stdenv.isDarwin [
      pkgs.libiconv
    ];

    doCheck = false;

    meta = with lib; {
      description = "Pulls an H.264 RTSP stream and streams it into Elodin-DB";
      homepage = "https://github.com/elodin-sys/elodin";
      license = licenses.mit;
      mainProgram = "rtsp-streamer";
    };
  }
