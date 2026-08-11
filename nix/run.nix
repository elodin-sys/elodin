{
  pkgs,
  rustToolchain,
}: let
  common = pkgs.callPackage ./pkgs/common.nix {};
  python = pkgs.elodin.elodin-py.python.withPackages (ps: [
    pkgs.elodin.elodin-py.py
    ps.grpcio
    ps.grpcio-tools
    ps.grpcio-health-checking
    ps.grpcio-reflection
  ]);
in
  pkgs.mkShell {
    name = "elodin-run";
    packages =
      [
        python
        pkgs.elodin.elodin-cli
        pkgs.elodin.elodin-db
        (rustToolchain pkgs)
        common.ktxTools
        pkgs.which
        pkgs.ripgrep
        pkgs.lesspipe
      ]
      ++ common.commonNativeBuildInputs
      ++ common.commonBuildInputs
      ++ pkgs.lib.optionals pkgs.stdenv.isLinux common.linuxGraphicsAudioDeps
      ++ pkgs.lib.optionals pkgs.stdenv.isDarwin common.darwinDeps;
    TOKTX = "${common.ktxTools}/bin/toktx";

    shellHook = ''
    '';
  }
