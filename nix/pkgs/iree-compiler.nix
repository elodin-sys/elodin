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
      url = "https://files.pythonhosted.org/packages/6f/ab/0a36a64a5f38b04ee8ada47780d0893a77c9c4741158fbb878d3cbcbc5a1/iree_base_compiler-3.10.0-cp313-cp313-macosx_13_0_universal2.whl";
      hash = "sha256-rRlZjGd9CpxsHpr4DmE5CL7BHgIPKzANVRH6lksFXjA=";
    }
    else if stdenv.hostPlatform.system == "x86_64-linux"
    then {
      url = "https://files.pythonhosted.org/packages/df/22/6590fb777060c870106f58234646c328a3675ca6e5b5463682b850fc8a27/iree_base_compiler-3.10.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
      hash = "sha256-vjAqXFOHa9y9qUJbiI25cIlX0ROtxYvu1vg6RFFbtBI=";
    }
    else if stdenv.hostPlatform.system == "aarch64-linux"
    then {
      url = "https://files.pythonhosted.org/packages/33/13/c14d9378c4a15ac80aa1fa6fb4e78457ab8f0f5dd88c205954e5965c5c5e/iree_base_compiler-3.10.0-cp313-cp313-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl";
      hash = "sha256-4kajF+UDU5uzmg/8Mv8pshbqeN0tRyOB0Z2vKdYIix8=";
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
