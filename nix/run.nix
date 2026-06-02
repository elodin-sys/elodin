{pkgs}: let
  common = pkgs.callPackage ./pkgs/common.nix {};
in
  pkgs.mkShell {
    name = "elodin-run";
    packages = [
      pkgs.elodin.elodin-py.py
      pkgs.elodin.elodin-cli
      pkgs.elodin.elodin-db
      common.ktxTools
      pkgs.gcc.cc.lib
      pkgs.gfortran.cc.lib
      pkgs.which
      pkgs.lesspipe
    ];
    TOKTX = "${common.ktxTools}/bin/toktx";

    shellHook = ''
    '';
  }
