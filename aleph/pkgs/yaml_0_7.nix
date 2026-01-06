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
    "-DYAML_CPP_BUILD_TOOLS=false"
    (lib.cmakeBool "YAML_BUILD_SHARED_LIBS" (!static))
    "-DINSTALL_GTEST=false"
  ];

  doCheck = stdenv.buildPlatform.canExecute stdenv.hostPlatform;

  # Fix double slashes in .pc file (CMake issue)
  # Replace ${prefix}// with ${prefix}/ and ${exec_prefix}// with ${exec_prefix}/
  postFixup = ''
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
