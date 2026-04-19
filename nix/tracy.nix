{pkgs}:
pkgs.mkShell {
  name = "elodin-tracy";
  GLIBC_TUNABLES = "glibc.rtld.optional_static_tls=65536";
  packages = [
    pkgs.elodin.elodin-py.py
    pkgs.elodin.elodin-cli-tracy
    pkgs.elodin.elodin-db
    pkgs.tracy
    pkgs.gcc.cc.lib
    pkgs.gfortran.cc.lib
    pkgs.which
    pkgs.lesspipe
  ];
}
