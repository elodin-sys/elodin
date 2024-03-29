{ config, pkgs, ... }:
let
  pytest = pkgs.runCommand "pytest"
    rec {
      src = ../libs/nox-py;
      doCheck = true;
      nativeBuildInputs = with pkgs; [
        (python3.withPackages (ps: with ps; [pytest pytest-json-report]))
        config.packages.elodin-py
      ];
    }
    ''
      pytest $src
      mkdir $out
    '';
in
{
  checks.pytest = pytest;
}
