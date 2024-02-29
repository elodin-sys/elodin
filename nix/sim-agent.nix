{ config, self', pkgs, lib, flakeInputs, rustToolchain, system, ... }:
let
  xla_sha256_map = {
    "aarch64-darwin" = "sha256:0ykfnp6d78vp2yrhmr8wa3rlv6cri6mdl0fg034za839j7i7xqkz";
    "aarch64-linux" = "sha256:0sy53r6qhw0n3n342s013nq5rnzlg1qdbmgpvawh3p35a21qy8xr";
    "x86_64-linux"   = "sha256:103mybbnz6fm2i3r0fy0nf23ffdjxb37wd4pzvmwn0dpczr6dkw1";
  };
  xla_ext = fetchTarball {
    url = "https://github.com/elodin-sys/xla/releases/download/v0.5.4/xla_extension-${system}-gnu-cpu.tar.gz";
    sha256 = builtins.getAttr system xla_sha256_map;
  };
  craneLib = (flakeInputs.crane.mkLib pkgs).overrideToolchain rustToolchain;
  crateName = craneLib.crateNameFromCargoToml { cargoToml = ../services/sim-agent/Cargo.toml; };
  src = pkgs.nix-gitignore.gitignoreSource [] ../.;
  arch = builtins.elemAt (lib.strings.splitString "-" system) 0;
  commonArgs = {
    inherit (crateName) pname version;
    inherit src;
    doCheck = false;
    cargoExtraArgs = "--package=${crateName.pname} --package=nox-py";
    nativeBuildInputs = with pkgs; [ maturin ];
    buildInputs = with pkgs;
      [
        systemdMinimal
        alsa-lib
        pkg-config
        (python3.withPackages (ps: with ps; [numpy jax jaxlib]))
        clang
        protobuf
        pango
        gtk3
      ]
      ++ lib.optionals stdenv.isDarwin [ pkgs.libiconv ];
    XLA_EXTENSION_DIR = "${xla_ext}";
    LIBCLANG_PATH = "${pkgs.llvmPackages_14.libclang.lib}/lib";
    BINDGEN_EXTRA_CLANG_ARGS = with pkgs; ''${lib.optionalString stdenv.cc.isGNU "-isystem ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc} -isystem ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc}/${stdenv.hostPlatform.config} -idirafter ${stdenv.cc.cc}/lib/gcc/${stdenv.hostPlatform.config}/${lib.getVersion stdenv.cc.cc}/include"}'';
  };
  cargoArtifacts = craneLib.buildDepsOnly commonArgs;
  bin = craneLib.buildPackage (commonArgs // {
    inherit cargoArtifacts;
  });

  wheelName = "elodin";
  wheelVersion = (craneLib.crateNameFromCargoToml { cargoToml = ../libs/nox-py/Cargo.toml; }).version;
  wheelSuffix = "cp310-abi3-linux_${arch}";
  wheel = craneLib.buildPackage (commonArgs // {
    inherit cargoArtifacts;
    pname = "elodin";
    buildPhase = "maturin build --offline --target-dir ./target -m libs/nox-py/Cargo.toml --release";
    installPhase = "install -D target/wheels/${wheelName}-${wheelVersion}-${wheelSuffix}.whl -t $out/";
  });
  elodin = ps: ps.buildPythonPackage rec {
    pname = wheelName;
    format = "wheel";
    version = wheelVersion;
    src = "${wheel}/${wheelName}-${wheelVersion}-${wheelSuffix}.whl";
    doCheck = false;
    propagatedBuildInputs = with ps; [ jax jaxlib typing-extensions ];
    pythonImportsCheck = [ wheelName ];
  };

  mkrootfs =
    {
      init,
      copyToRoot
    }:
    pkgs.runCommand "copy-contents"
      {
        init = init;
        contents = copyToRoot;
        closureInfo = pkgs.closureInfo {
          rootPaths = copyToRoot;
        };
        nativeBuildInputs = with pkgs; [ rsync zstd squashfsTools ];
      }
      ''
        mkdir -p rootfs/nix/store
        cd rootfs
        mkdir -p nix/store proc sys tmp dev
        cp $init init
        for item in $contents; do
          rsync -aK $item/ .
        done
        for item in $(< $closureInfo/store-paths); do
          rsync -aK $item ./nix/store/
        done
        mkdir $out
        mksquashfs . $out/root.squashfs -comp zstd
      '';

  # TODO(Akhil): use s6 supervision suite instead as running workload as pid 1 is not recommended
  init = pkgs.writeScript "init"
    ''
    #!/bin/sh

    mount -n -t proc none /proc
    mount -n -t sysfs none /sys
    mount -n -t tmpfs none /tmp
    mdev -s

    setsid sh -c 'sh </dev/ttyS0 >/dev/ttyS0 2>&1'
    poweroff -f
    '';
  vm = mkrootfs {
    init = init;
    copyToRoot = with pkgs; [
      cacert
      busybox
      (python3.withPackages (ps: with ps; [(elodin ps)]))
    ];
  };

  image = pkgs.dockerTools.buildLayeredImage {
    name = "elo-sim-agent";
    tag = "latest";
    contents = with pkgs; [
      cacert
      busybox
      (python3.withPackages (ps: with ps; [(elodin ps)]))
    ];
    config = {
      Env = ["SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt"];
      Cmd = ["${bin}/bin/${crateName.pname}"];
    };
  };
in
{
  packages.sim-agent-image = image;
  packages.sandbox-vm = vm;
}
