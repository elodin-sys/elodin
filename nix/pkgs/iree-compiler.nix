# IREE Compiler Python package
# Downloads pre-built wheels from PyPI and installs them as a Python package
{
  lib,
  stdenv,
  fetchurl,
  python3,
  autoPatchelfHook,
  ncurses,
  zlib,
  libxml2,
  ...
}: let
  version = "3.11.0";
  wheelTag = "cp312-abi3";

  # Platform-specific wheel URLs and hashes from PyPI
  # Hash format: sha256-<base64> (SRI format)
  wheelInfo =
    if stdenv.isDarwin
    then {
      url = "https://files.pythonhosted.org/packages/4c/1f/ae45ccb7edef9eead14eacb0e8b472ce3d72ee3863a0f9eb6c916e09e6e3/iree_base_compiler-${version}-${wheelTag}-macosx_13_0_universal2.whl";
      hash = "sha256-Lp/IiLkERmKhTuugG2MC11phGPaIdGmdz4+cfpkaTuI=";
    }
    else if stdenv.hostPlatform.system == "x86_64-linux"
    then {
      url = "https://files.pythonhosted.org/packages/47/9c/f7cd82016c869ceb37b787b628608b68225c9e7bc46e0de9f18a9932ef3c/iree_base_compiler-${version}-${wheelTag}-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
      hash = "sha256-mfwzaTdUBUhfEdpUGnXvKxU9Up+jcSwdl3HJijXiZp4=";
    }
    else if stdenv.hostPlatform.system == "aarch64-linux"
    then {
      url = "https://files.pythonhosted.org/packages/24/a3/67dda13e131479d17a5f8c63c0e43bd6674ddc5ff84d49f77b9155a917f8/iree_base_compiler-${version}-${wheelTag}-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl";
      hash = "sha256-GuqTT9wuvh3WTl8gY6lt5nWjXwtIVvndt2st9Y3++4c=";
    }
    else throw "Unsupported platform: ${stdenv.hostPlatform.system}";

  wheelFile = fetchurl {
    inherit (wheelInfo) url hash;
  };
in
  python3.pkgs.buildPythonPackage rec {
    pname = "iree-base-compiler";
    inherit version;
    format = "wheel";

    src = wheelFile;

    nativeBuildInputs = lib.optionals stdenv.isLinux [
      autoPatchelfHook
    ];

    buildInputs = lib.optionals stdenv.isLinux [
      stdenv.cc.cc.lib
      ncurses
      zlib
      libxml2
    ];

    propagatedBuildInputs = with python3.pkgs; [
      numpy
      sympy
    ];

    pythonImportsCheck = ["iree.compiler"];

    meta = with lib; {
      description = "IREE Python Compiler API";
      homepage = "https://iree.dev/";
      license = licenses.asl20;
      platforms = ["x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin"];
    };
  }
