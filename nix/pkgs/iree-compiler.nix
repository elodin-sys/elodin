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
  version = "3.10.0";

  # Platform-specific wheel URLs and hashes from PyPI
  # Hash format: sha256-<base64> (SRI format)
  wheelInfo =
    if stdenv.isDarwin
    then {
      url = "https://files.pythonhosted.org/packages/2e/47/fcf68acc6b34ca3b92b08d171f4a5f676346c0c1ee65efeba1bc00c7efb7/iree_base_compiler-3.10.0-cp312-cp312-macosx_13_0_universal2.whl";
      hash = "sha256-S8YV/QmsuXSzEvj7A2YWy2PWnUv8uCT4VowZ8/JYWiY=";
    }
    else if stdenv.hostPlatform.system == "x86_64-linux"
    then {
      url = "https://files.pythonhosted.org/packages/0c/b9/ac4156d5c99afb799c07a06066e3db5c72ef2e10107a46da78e532c678fe/iree_base_compiler-3.10.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
      hash = "sha256-SjUrvnu6z7KdAVSQhD71YA7FONNV6Ay01JUhJkVzHgM=";
    }
    else if stdenv.hostPlatform.system == "aarch64-linux"
    then {
      url = "https://files.pythonhosted.org/packages/6a/d9/99eb4cd2bdbe77bc0597a4cc32d3a4c6adb2af0cf0f1c3bbffa97be43f66/iree_base_compiler-3.10.0-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl";
      hash = "sha256-FWOgaBNu3GhSwV8zbovpTjYgFZ78JDj/On2n22iXmm0=";
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
