{
  lib,
  stdenv,
  fetchFromGitHub,
  gitUpdater,
  cmake,
  static ? stdenv.hostPlatform.isStatic,
  ...
}:

stdenv.mkDerivation rec {
  pname = "yaml-cpp";
  version = "0.6.3";

  src = fetchFromGitHub {
    owner = "jbeder";
    repo = "yaml-cpp";
    rev = "yaml-cpp-${version}";
    hash = "sha256-Ggx+ybQyArIMPfXEffMUqFAbIPbvJJMRRTtyzvrvc3o=";
  };

  strictDeps = true;

  nativeBuildInputs = [
    cmake
  ];

  cmakeFlags = [
    "-DYAML_CPP_BUILD_TOOLS=false"
    (lib.cmakeBool "YAML_BUILD_SHARED_LIBS" (!static))
    "-DINSTALL_GTEST=false"
  ];

  doCheck = stdenv.buildPlatform.canExecute stdenv.hostPlatform;

  passthru.updateScript = gitUpdater { };

  meta = with lib; {
    description = "YAML parser and emitter for C++";
    homepage = "https://github.com/jbeder/yaml-cpp";
    license = licenses.mit;
    platforms = platforms.all;
    maintainers = with maintainers; [ OPNA2608 ];
  };
}
