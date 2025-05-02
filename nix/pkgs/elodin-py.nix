{
  pkgs,
  lib,
  crane,
  rustToolchain,
  system,
  ...
}: let
  xla_path_map = {
    "aarch64-darwin" = "xla_extension-aarch64-darwin-cpu.tar.gz";
    "aarch64-linux" = "xla_extension-aarch64-linux-gnu-cpu.tar.gz";
    "x86_64-linux" = "xla_extension-x86_64-linux-gnu-cpu.tar.gz";
  };
  xla_sha256_map = {
    "aarch64-darwin" = "sha256:0ykfnp6d78vp2yrhmr8wa3rlv6cri6mdl0fg034za839j7i7xqkz";
    "aarch64-linux" = "sha256:0sy53r6qhw0n3n342s013nq5rnzlg1qdbmgpvawh3p35a21qy8xr";
    "x86_64-linux" = "sha256:103mybbnz6fm2i3r0fy0nf23ffdjxb37wd4pzvmwn0dpczr6dkw1";
  };
  xla_path = builtins.getAttr system xla_path_map;
  xla_ext = fetchTarball {
    url = "https://github.com/elodin-sys/xla/releases/download/v0.5.4/${xla_path}";
    sha256 = builtins.getAttr system xla_sha256_map;
  };
  craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
  pname = (craneLib.crateNameFromCargoToml {cargoToml = ../../libs/nox-py/Cargo.toml;}).pname;
  version = (craneLib.crateNameFromCargoToml {cargoToml = ../../Cargo.toml;}).version;

  pyFilter = path: _type: builtins.match ".*py$" path != null;
  mdFilter = path: _type: builtins.match ".*nox-py.*md$" path != null;
  protoFilter = path: _type: builtins.match ".*proto$" path != null;
  assetFilter = path: _type: builtins.match ".*assets.*$" path != null;
  cppFilter = path: _type: builtins.match ".*[h|(cpp)|(cpp.jinja)]$" path != null;
  srcFilter = path: type: (pyFilter path type) || (mdFilter path type) || (protoFilter path type) || (assetFilter path type) || (cppFilter path type) || (craneLib.filterCargoSources path type);
  src = lib.cleanSourceWith {
    src = craneLib.path ./../..;
    filter = srcFilter;
  };

  arch = builtins.elemAt (lib.strings.splitString "-" system) 0;
  commonArgs = {
    inherit pname version;
    inherit src;
    doCheck = false;
    nativeBuildInputs = with pkgs; [maturin];
    buildInputs = with pkgs;
      [
        pkg-config
        python3
        openssl
        gfortran
        gfortran.cc.lib
        cmake
      ]
      ++ lib.optionals stdenv.isDarwin [pkgs.libiconv];
    XLA_EXTENSION_DIR = "${xla_ext}";
    cargoExtraArgs = "--package=nox-py";
    OPENSSL_DIR = "${pkgs.openssl.dev}";
    OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
    OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include/";
  };
  cargoArtifacts = craneLib.buildDepsOnly commonArgs;

  wheelName = "elodin";
  wheelSuffix = "cp310-abi3-linux_${arch}";
  wheel = craneLib.buildPackage (commonArgs
    // {
      inherit cargoArtifacts;
      pname = "elodin";
      buildPhase = "maturin build --offline --target-dir ./target -m libs/nox-py/Cargo.toml --release";
      installPhase = "install -D target/wheels/${wheelName}-${version}-${wheelSuffix}.whl -t $out/";
    });
  elodin = ps:
    ps.buildPythonPackage {
      pname = wheelName;
      format = "wheel";
      version = version;
      src = "${wheel}/${wheelName}-${version}-${wheelSuffix}.whl";
      doCheck = false;
      propagatedBuildInputs = with ps; [jax jaxlib typing-extensions numpy polars pytest];
      pythonImportsCheck = [wheelName];
    };
in
  elodin pkgs.python3Packages
