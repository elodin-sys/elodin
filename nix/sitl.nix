{ pkgs }:

pkgs.mkShell {
  name = "sitl";
  packages = [
    pkgs.elodin.elodin-py.py
    pkgs.elodin.elodin-cli
    pkgs.elodin.elodin-db
    pkgs.gcc.cc.lib
    pkgs.gfortran.cc.lib
    pkgs.which
    pkgs.lesspipe
  ];
  shellHook = ''
  '';
}

