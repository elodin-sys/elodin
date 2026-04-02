{
  lib,
  stdenv,
  fetchFromGitHub,
  cmake,
  ninja,
  python3,
  git,
  lld,
  musl ? null,
  darwin ? {},
  fixDarwinDylibNames ? null,
  patchelf ? null,
  ...
}:
  assert stdenv.isLinux -> musl != null;
  let
  version = "3.11.0";

  # Same IREE source as iree-runtime.nix
  ireeSrc = fetchFromGitHub {
    owner = "openxla";
    repo = "iree";
    rev = "v${version}";
    hash = "sha256-2NHWLhhCoCHJGWN+ZPZm44pOeksr3fTpLH7X0hkThWw=";
  };

  # Compiler-specific submodules (not needed for runtime-only build)
  llvmProject = fetchFromGitHub {
    owner = "iree-org";
    repo = "llvm-project";
    rev = "66395ad94c3283d938a2957e7c7b439711764680";
    hash = "sha256-gAEahFTP3UkE9mwEIvsNg4AnjxAs/GibvUVTZnTFjbg=";
  };

  stablehlo = fetchFromGitHub {
    owner = "iree-org";
    repo = "stablehlo";
    rev = "46af9d3590bc712d1c9e2f8ef408d2ecbb6108a3";
    hash = "sha256-yeDKEteSDyNOCRvxsMRhp8RbgDMr6qVm+voMNs9Vs8M=";
  };

  # Shared submodules (same as runtime)
  flatcc = fetchFromGitHub {
    owner = "dvidelabs";
    repo = "flatcc";
    rev = "9362cd00f0007d8cbee7bff86e90fb4b6b227ff3";
    hash = "sha256-umZ9TvNYDZtF/mNwQUGuhAGve0kPw7uXkaaQX0EzkBY=";
  };

  benchmark = fetchFromGitHub {
    owner = "google";
    repo = "benchmark";
    rev = "192ef10025eb2c4cdd392bc502f0c852196baa48";
    hash = "sha256-Mm4pG7zMB00iof32CxreoNBFnduPZTMp3reHMCIAFPQ=";
  };

  printf = fetchFromGitHub {
    owner = "eyalroz";
    repo = "printf";
    rev = "f1b728cbd5c6e10dc1f140f1574edfd1ccdcbedb";
    hash = "sha256-HHp6uKEJv3HWEGgIBjeMsCXUSIPYTLwofjcDTswgSuA=";
  };
in
  stdenv.mkDerivation {
    pname = "iree-compiler";
    inherit version;

    src = ireeSrc;

    patches = [
      ./iree-fix-i1-hex-parsing.patch
      ./iree-fix-scalar-concat.patch
      ./iree-fix-lapack-custom-call.patch
      ./iree-fix-power-zero.patch
      ./iree-fix-large-constant-promotion.patch
      ./iree-fix-case-to-if.patch
      ./iree-fix-scf-to-cf.patch
    ];

    postUnpack = ''
      # Compiler-specific submodules
      rmdir $sourceRoot/third_party/llvm-project 2>/dev/null || true
      cp -r ${llvmProject} $sourceRoot/third_party/llvm-project
      chmod -R u+w $sourceRoot/third_party/llvm-project

      rmdir $sourceRoot/third_party/stablehlo 2>/dev/null || true
      cp -r ${stablehlo} $sourceRoot/third_party/stablehlo
      chmod -R u+w $sourceRoot/third_party/stablehlo

      # Shared submodules (same as runtime)
      rmdir $sourceRoot/third_party/flatcc 2>/dev/null || true
      cp -r ${flatcc} $sourceRoot/third_party/flatcc
      chmod -R u+w $sourceRoot/third_party/flatcc

      rmdir $sourceRoot/third_party/benchmark 2>/dev/null || true
      cp -r ${benchmark} $sourceRoot/third_party/benchmark
      chmod -R u+w $sourceRoot/third_party/benchmark
      # Force POSIX regex to succeed in benchmark (nix sandbox has limited headers)
      sed -i 's/message(FATAL_ERROR "Failed to determine the source files for the regular expression backend")/set(HAVE_STD_REGEX ON)/' \
        $sourceRoot/third_party/benchmark/CMakeLists.txt

      rmdir $sourceRoot/third_party/printf 2>/dev/null || true
      cp -r ${printf} $sourceRoot/third_party/printf
      chmod -R u+w $sourceRoot/third_party/printf
    '';

    nativeBuildInputs =
      [cmake ninja python3 git]
      ++ lib.optionals stdenv.isLinux [lld patchelf]
      ++ lib.optionals stdenv.isDarwin [
        (darwin.cctools or null)
        fixDarwinDylibNames
      ];

    # Host tools built during compilation need libstdc++ (Linux) or libc++ (macOS)
    LD_LIBRARY_PATH = lib.optionalString stdenv.isLinux (lib.makeLibraryPath [stdenv.cc.cc.lib]);

    cmakeFlags = [
      "-DIREE_BUILD_COMPILER=ON"
      "-DIREE_BUILD_TESTS=OFF"
      "-DIREE_BUILD_SAMPLES=OFF"

      "-DIREE_TARGET_BACKEND_DEFAULTS=OFF"
      "-DIREE_TARGET_BACKEND_LLVM_CPU=ON"

      "-DIREE_HAL_DRIVER_DEFAULTS=OFF"
      "-DIREE_HAL_DRIVER_LOCAL_SYNC=ON"
      "-DIREE_HAL_DRIVER_LOCAL_TASK=ON"

      "-DIREE_HAL_EXECUTABLE_LOADER_DEFAULTS=OFF"
      "-DIREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF=ON"
      "-DIREE_HAL_EXECUTABLE_LOADER_SYSTEM_LIBRARY=ON"
      "-DIREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE=ON"

      "-DIREE_INPUT_STABLEHLO=ON"
      "-DIREE_INPUT_TORCH=OFF"
      "-DIREE_INPUT_TOSA=OFF"

      "-DIREE_ENABLE_CPUINFO=OFF"
      "-DIREE_BUILD_TRACY=OFF"
      "-DIREE_ENABLE_LIBBACKTRACE=OFF"
      "-DIREE_BUILD_BENCHMARKS=OFF"
      "-DBENCHMARK_ENABLE_TESTING=OFF"
      "-DBENCHMARK_ENABLE_GTEST_TESTS=OFF"
      "-DBENCHMARK_USE_BUNDLED_GTEST=OFF"

      "-DBUILD_SHARED_LIBS=OFF"

      # X86 for CI/dev machines, AArch64 for Apple Silicon and Jetson Orin.
      "-DLLVM_TARGETS_TO_BUILD=X86;AArch64"
      "-DLLVM_ENABLE_ASSERTIONS=OFF"
      "-DIREE_ENABLE_ASSERTIONS=OFF"
      "-DIREE_TARGET_BACKEND_VMVX=OFF"
      "-DIREE_OUTPUT_FORMAT_C=OFF"
      "-DIREE_BUILD_BINDINGS_TFLITE=OFF"
      "-DIREE_BUILD_BINDINGS_TFLITE_JAVA=OFF"
    ]
    ++ lib.optional stdenv.isLinux "-DIREE_ENABLE_LLD=ON";

    # Note: we don't restrict --target because partial builds have dependency
    # ordering issues with ukernel bitcode generation. The full build is needed.

    installPhase = ''
      runHook preInstall

      mkdir -p $out/bin $out/lib $out/libexec

      cp tools/iree-compile $out/bin/
      ${
        if stdenv.isDarwin
        then ''find lib -name "libIREECompiler.dylib*" -exec cp -P {} $out/lib/ \;''
        else ''find lib -name "libIREECompiler.so*" -exec cp -P {} $out/lib/ \;''
      }

      find . -name "iree-lld" -type f -executable | head -1 | while read f; do
        cp "$f" $out/libexec/iree-lld
      done
      ${
        if stdenv.isLinux
        then ''
          # Wrapper appends musl libc.a so f64 math symbols (sin, cos, log, exp)
          # are resolved during embedded ELF linking.
          printf '#!/bin/sh\nexec "%s" "$@" "%s"\n' \
            "$out/libexec/iree-lld" "${musl}/lib/libc.a" \
            > $out/bin/iree-lld
        ''
        else ''
          # macOS system linker provides libm; no musl wrapper needed.
          printf '#!/bin/sh\nexec "%s" "$@"\n' \
            "$out/libexec/iree-lld" \
            > $out/bin/iree-lld
        ''
      }
      chmod +x $out/bin/iree-lld

      ${
        if stdenv.isLinux
        then ''
          for f in $out/bin/iree-compile $out/libexec/iree-lld; do
            patchelf --set-rpath "$out/lib:${stdenv.cc.cc.lib}/lib" "$f" || true
          done
          for f in $out/lib/*.so*; do
            patchelf --set-rpath "${stdenv.cc.cc.lib}/lib" "$f" || true
          done
        ''
        else ''
          # Fix RPATH for macOS: point binaries at $out/lib for libIREECompiler.dylib
          for f in $out/bin/iree-compile $out/libexec/iree-lld; do
            install_name_tool -add_rpath "$out/lib" "$f" 2>/dev/null || true
            install_name_tool -add_rpath "${stdenv.cc.cc.lib}/lib" "$f" 2>/dev/null || true
          done
          for f in $out/lib/*.dylib*; do
            install_name_tool -id "$f" "$f" 2>/dev/null || true
          done
        ''
      }

      runHook postInstall
    '';

    meta = with lib; {
      description = "IREE compiler (iree-compile) built from source for targeted patching";
      homepage = "https://iree.dev";
      license = licenses.asl20;
    };
  }
