{
  pkgs,
  rustToolchain,
}: let
  common = pkgs.callPackage ./pkgs/common.nix {};
  lib = pkgs.lib;
  python = pkgs.elodin.elodin-py.python.withPackages (ps: [
    pkgs.elodin.elodin-py.py
    ps.grpcio
    ps.grpcio-tools
    ps.grpcio-health-checking
    ps.grpcio-reflection
  ]);
  basePackages = [
    python
    pkgs.elodin.elodin-cli
    pkgs.elodin.elodin-db
    (rustToolchain pkgs)
    common.ktxTools
    pkgs.which
    pkgs.ripgrep
    pkgs.lesspipe
  ];
  linuxAttrs = lib.optionalAttrs pkgs.stdenv.isLinux (
    (common.linuxGraphicsEnv {inherit pkgs;})
    // {
      GST_PLUGIN_PATH = common.makeGstPluginPath {inherit pkgs;};
      LD_LIBRARY_PATH = common.makeLinuxLibraryPath {inherit pkgs;};
    }
  );
in
  pkgs.mkShell (linuxAttrs
    // {
      name = "elodin-run";
      packages =
        basePackages
        ++ common.commonNativeBuildInputs
        ++ common.commonBuildInputs
        ++ lib.optionals pkgs.stdenv.isLinux (
          common.linuxGraphicsAudioDeps
          ++ common.linuxCaptureTools
          ++ common.linuxGstreamerPackages
          ++ [pkgs.ffmpeg-full]
        )
        ++ lib.optionals pkgs.stdenv.isDarwin common.darwinDeps;
      TOKTX = "${common.ktxTools}/bin/toktx";
      shellHook = lib.optionalString pkgs.stdenv.isLinux common.linuxEditorShellHook;
    })
