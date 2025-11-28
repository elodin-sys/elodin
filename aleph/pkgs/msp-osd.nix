{
  lib,
  pkgs,
  rustPlatform,
  rustToolchain,
  pkg-config,
  systemd,
}:
rustPlatform.buildRustPackage rec {
  pname = "msp-osd";
  version = "0.1.0";

  src = ../../fsw/msp-osd;

  cargoLock = {
    lockFile = ../../Cargo.lock;
  };

  nativeBuildInputs = [
    pkg-config
    (rustToolchain pkgs)
  ];

  buildInputs = [
    systemd
  ];

  # Use the workspace's Cargo.lock
  postPatch = ''
    ln -sf ${../../Cargo.lock} Cargo.lock
  '';

  meta = with lib; {
    description = "MSP OSD Service - MSP DisplayPort OSD for VTX";
    homepage = "https://github.com/elodin-sys/elodin";
    license = licenses.mit;
    maintainers = with maintainers; [];
    mainProgram = "msp-osd";
  };
}
