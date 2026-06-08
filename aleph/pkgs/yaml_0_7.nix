{
  lib,
  stdenv,
  fetchFromGitHub,
  cmake,
  static ? stdenv.hostPlatform.isStatic,
  ...
}:
stdenv.mkDerivation rec {
  pname = "yaml-cpp";
  version = "0.7.0";

  src = fetchFromGitHub {
    owner = "jbeder";
    repo = "yaml-cpp";
    rev = "yaml-cpp-${version}";
    hash = "sha256-2tFWccifn0c2lU/U1WNg2FHrBohjx8CXMllPJCevaNk=";
  };

  strictDeps = true;

  nativeBuildInputs = [
    cmake
  ];

  cmakeFlags = [
    "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
    "-DYAML_CPP_BUILD_TOOLS=false"
    "-DYAML_CPP_BUILD_TESTS=false"
    (lib.cmakeBool "YAML_BUILD_SHARED_LIBS" (!static))
    "-DINSTALL_GTEST=false"
  ];

  # newer GCC versions are stricter about source files including the headers they use
  # yaml-cpp 0.7.0 uses uint16_t/uint32_t here but did not include <cstdint>
  # probably fixed upstream, but we have to stick with yaml-cpp 0.7.0 because deepstream 7.1 targets it
  postPatch = ''
        substituteInPlace src/emitterutils.cpp \
          --replace-fail "#include <sstream>" "#include <sstream>
    #include <cstdint>"
  '';

  doCheck = stdenv.buildPlatform.canExecute stdenv.hostPlatform;

  # Fix double slashes in .pc file (CMake issue)
  # Replace ${prefix}// with ${prefix}/ and ${exec_prefix}// with ${exec_prefix}/
  # Must be postInstall (not postFixup) because Nix's fixupPhase checks for broken
  # .pc files BEFORE postFixup runs
  postInstall = ''
    sed -i 's|''${prefix}//|''${prefix}/|g' $out/share/pkgconfig/yaml-cpp.pc
    sed -i 's|''${exec_prefix}//|''${exec_prefix}/|g' $out/share/pkgconfig/yaml-cpp.pc
  '';

  meta = with lib; {
    description = "YAML parser and emitter for C++";
    homepage = "https://github.com/jbeder/yaml-cpp";
    license = licenses.mit;
    platforms = platforms.all;
    maintainers = with maintainers; [OPNA2608];
  };
}
