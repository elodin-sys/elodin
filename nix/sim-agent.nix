{
  config,
  self',
  pkgs,
  lib,
  flakeInputs,
  rustToolchain,
  system,
  ...
}: let
  xla_sha256_map = {
    "aarch64-darwin" = "sha256:0ykfnp6d78vp2yrhmr8wa3rlv6cri6mdl0fg034za839j7i7xqkz";
    "aarch64-linux" = "sha256:0sy53r6qhw0n3n342s013nq5rnzlg1qdbmgpvawh3p35a21qy8xr";
    "x86_64-linux" = "sha256:103mybbnz6fm2i3r0fy0nf23ffdjxb37wd4pzvmwn0dpczr6dkw1";
  };
  xla_ext = fetchTarball {
    url = "https://github.com/elodin-sys/xla/releases/download/v0.5.4/xla_extension-${system}-gnu-cpu.tar.gz";
    sha256 = builtins.getAttr system xla_sha256_map;
  };
  craneLib = (flakeInputs.crane.mkLib pkgs).overrideToolchain rustToolchain;
  crateName = craneLib.crateNameFromCargoToml {cargoToml = ../services/sim-agent/Cargo.toml;};

  pyFilter = path: _type: builtins.match ".*py$" path != null;
  mdFilter = path: _type: builtins.match ".*nox-py.*md$" path != null;
  protoFilter = path: _type: builtins.match ".*proto$" path != null;
  assetFilter = path: _type: builtins.match ".*assets.*$" path != null;
  cppFilter = path: _type: builtins.match ".*[h|(cpp)]$" path != null;
  srcFilter = path: type: (pyFilter path type) || (mdFilter path type) || (protoFilter path type) || (assetFilter path type) || (cppFilter path type) || (craneLib.filterCargoSources path type);
  src = lib.cleanSourceWith {
    src = craneLib.path ./..;
    filter = srcFilter;
  };

  arch = builtins.elemAt (lib.strings.splitString "-" system) 0;
  commonArgs = {
    inherit (crateName) pname version;
    inherit src;
    doCheck = false;
    nativeBuildInputs = with pkgs; [maturin];
    buildInputs = with pkgs;
      [
        systemdMinimal
        alsa-lib
        pkg-config
        python3
        protobuf
        openssl
        gfortran
        gfortran.cc.lib
        cmake
      ]
      ++ lib.optionals stdenv.isDarwin [pkgs.libiconv];
    XLA_EXTENSION_DIR = "${xla_ext}";
    cargoExtraArgs = "--package=sim-agent --package=nox-py";
    OPENSSL_DIR = "${pkgs.openssl.dev}";
    OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
    OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include/";
  };
  cargoArtifacts = craneLib.buildDepsOnly commonArgs;
  bin = craneLib.buildPackage (commonArgs
    // {
      inherit cargoArtifacts;
    });

  wheelName = "elodin";
  wheelVersion = (craneLib.crateNameFromCargoToml {cargoToml = ../Cargo.toml;}).version;
  wheelSuffix = "cp310-abi3-linux_${arch}";
  wheel = craneLib.buildPackage (commonArgs
    // {
      inherit cargoArtifacts;
      pname = "elodin";
      buildPhase = "maturin build --offline --target-dir ./target -m libs/nox-py/Cargo.toml --release";
      installPhase = "install -D target/wheels/${wheelName}-${wheelVersion}-${wheelSuffix}.whl -t $out/";
    });
  elodin = ps:
    ps.buildPythonPackage rec {
      pname = wheelName;
      format = "wheel";
      version = wheelVersion;
      src = "${wheel}/${wheelName}-${wheelVersion}-${wheelSuffix}.whl";
      doCheck = false;
      propagatedBuildInputs = with ps; [jax jaxlib typing-extensions numpy polars pytest];
      pythonImportsCheck = [wheelName];
    };

  kernel = pkgs.linux_6_6.override {
    defconfig = "defconfig kvm_guest.config";
    autoModules = false;
    ignoreConfigErrors = true;
    structuredExtraConfig = with lib.kernel; {
      SQUASHFS = yes;
      VIRTIO_MMIO = yes;
      VSOCKETS = yes;
      VIRTIO_VSOCKETS = yes;
      VIRTIO_VSOCKETS_COMMON = yes;
      VHOST_VSOCK = yes;
    };
  };
  rootfs =
    pkgs.runCommand "rootfs"
    rec {
      contents = with pkgs; [
        busybox
        config.packages.sim-builder
        (python3.withPackages (ps: with ps; [(elodin ps) pytest pytest-json-report]))
        openblasCompat
      ];
      closureInfo = pkgs.closureInfo {rootPaths = contents;};
      nativeBuildInputs = with pkgs; [rsync zstd squashfsTools];
    }
    ''
      mkdir rootfs
      cd rootfs
      mkdir -p nix/store proc sys tmp dev
      cp ${../services/sim-builder/init} init
      for item in $contents; do
        rsync -aK $item/ .
      done
      for item in $(< $closureInfo/store-paths); do
        rsync -aK $item ./nix/store/
      done
      mkdir $out
      mksquashfs . $out/root.squashfs -comp zstd
    '';
  runvm = pkgs.runCommand "runvm" {} "mkdir $out; cp ${../services/sim-builder/runvm} $out/runvm";
  vm = pkgs.buildEnv {
    name = "sim-builder-vm";
    extraPrefix = "/vm";
    paths = [
      pkgs.qboot
      kernel
      rootfs
      runvm
    ];
    postBuild = "rm -rf $out/vm/{bios.bin.elf,lib,System.map}";
  };
  mkfs = pkgs.buildEnv {
    name = "mkfs";
    paths = with pkgs; [e2fsprogs];
    pathsToLink = ["/bin"];
  };

  image = pkgs.dockerTools.buildLayeredImage {
    name = "elo-sim-agent";
    tag = "latest";
    contents = with pkgs; [
      mkfs
      cacert
      busybox
      vm
      qemu_kvm
    ];
    config = {
      Env = ["SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt"];
      Cmd = ["${bin}/bin/${crateName.pname}"];
    };
  };
in {
  packages.sim-agent-image = image;
  packages.sandbox-vm = vm;
  packages.elodin-py = elodin pkgs.python3Packages;
}
