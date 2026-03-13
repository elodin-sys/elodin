{pkgs}:
pkgs.mkShell {
  name = "elodin-tracy";
  GLIBC_TUNABLES = "glibc.rtld.optional_static_tls=4096";
  packages =
    [
      pkgs.elodin.elodin-py-tracy.py
      pkgs.elodin.elodin-cli-tracy
      pkgs.elodin.elodin-db
      pkgs.tracy
      pkgs.gcc.cc.lib
      pkgs.gfortran.cc.lib
      pkgs.which
      pkgs.lesspipe
    ]
    ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
      pkgs.elodin.iree_runtime_tracy
    ];
}
