{
  lib,
  stdenv,
  fetchFromGitHub,
  cmake,
  ninja,
  python3,
  git,
  ...
}: let
  version = "3.10.0";

  flatcc = fetchFromGitHub {
    owner = "dvidelabs";
    repo = "flatcc";
    rev = "9362cd00f0007d8cbee7bff86e90fb4b6b227ff3";
    hash = "sha256-umZ9TvNYDZtF/mNwQUGuhAGve0kPw7uXkaaQX0EzkBY=";
  };

  benchmark = fetchFromGitHub {
    owner = "google";
    repo = "benchmark";
    rev = "99bdb2127d1fa1cff444bbefb814e105c7d20c45";
    hash = "sha256-d/7BDynAUsH20bGqyh4HLKKgqCeGlTRQRvqX5dmpMLg=";
  };
in
  stdenv.mkDerivation {
    pname = "iree-runtime";
    inherit version;

    src = fetchFromGitHub {
      owner = "iree-org";
      repo = "iree";
      rev = "v${version}";
      hash = "sha256-6phU7ypXdvWBujnIzKGbYyABFAEb7U4miukelxNC4Gw=";
    };

    postUnpack = ''
      # Place required third-party submodules for runtime build
      rmdir $sourceRoot/third_party/flatcc 2>/dev/null || true
      cp -r ${flatcc} $sourceRoot/third_party/flatcc
      chmod -R u+w $sourceRoot/third_party/flatcc

      rmdir $sourceRoot/third_party/benchmark 2>/dev/null || true
      cp -r ${benchmark} $sourceRoot/third_party/benchmark
      chmod -R u+w $sourceRoot/third_party/benchmark
    '';

    nativeBuildInputs = [cmake ninja python3 git];

    cmakeFlags = [
      "-DIREE_BUILD_COMPILER=OFF"
      "-DIREE_BUILD_TESTS=OFF"
      "-DIREE_BUILD_SAMPLES=OFF"

      # HAL drivers for CPU execution
      "-DIREE_HAL_DRIVER_DEFAULTS=OFF"
      "-DIREE_HAL_DRIVER_LOCAL_SYNC=ON"
      "-DIREE_HAL_DRIVER_LOCAL_TASK=ON"

      # Executable loaders
      "-DIREE_HAL_EXECUTABLE_LOADER_DEFAULTS=OFF"
      "-DIREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF=ON"
      "-DIREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE=ON"

      # Static libraries
      "-DBUILD_SHARED_LIBS=OFF"

      # Disable optional features not needed for our use case
      "-DIREE_ENABLE_CPUINFO=OFF"
      "-DIREE_BUILD_TRACY=OFF"
      # Prevent FetchContent from trying to git-clone libbacktrace during
      # the build (nix sandboxes have no network access). Defaults to ON
      # on Linux.
      "-DIREE_ENABLE_LIBBACKTRACE=OFF"
    ];

    # Install headers and static libraries
    installPhase = ''
      runHook preInstall

      mkdir -p $out/lib $out/include

      # Copy all static libraries from the build tree
      find . -name '*.a' -exec cp {} $out/lib/ \;

      # Copy runtime headers from the source tree
      cd $NIX_BUILD_TOP/$sourceRoot
      cp -r runtime/src/iree $out/include/iree

      # Remove non-header files from include dir
      find $out/include -type f ! -name '*.h' -delete
      find $out/include -type d -empty -delete

      runHook postInstall
    '';

    meta = with lib; {
      description = "IREE runtime C library for executing compiled ML models";
      homepage = "https://iree.dev";
      license = licenses.asl20;
    };
  }
