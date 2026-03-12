{
  lib,
  stdenv,
  fetchFromGitHub,
  fetchurl,
  runCommand,
  cmake,
  ninja,
  python3,
  git,
  pkg-config,
  zstd,
  gnused,
  enableTracing ? false,
  tracySrc ? null,
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

  capstoneSrc = fetchFromGitHub {
    owner = "capstone-engine";
    repo = "capstone";
    rev = "97db712c91e964718f9cc300e81b9cf76b31a22e";
    hash = "sha256-oKRu3P1inWueEMIpL0uI2ayCMHZ9FIVotil4sqwLqH4=";
  };

  ppqsortRaw = fetchFromGitHub {
    owner = "GabTux";
    repo = "PPQSort";
    rev = "4b964020d67b435dae7ebac7b8f5ecea1f421c58";
    hash = "sha256-myMOKIq7veA/p+mRsMuecWPUI1Xh7z+38sJF1J7bGYM=";
  };

  cpmSrc = fetchurl {
    url = "https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.38.7/CPM.cmake";
    hash = "sha256-g+XrcbK7uLHyrTjxlQKHoFdiTjhcI49gh/lM38RK+cU=";
  };

  packageProjectSrc = fetchFromGitHub {
    owner = "TheLartians";
    repo = "PackageProject.cmake";
    rev = "v1.11.1";
    hash = "sha256-E7WZSYDlss5bidbiWL1uX41Oh6JxBRtfhYsFU19kzIw=";
  };

  ppqsortSrc =
    runCommand "ppqsort-iree-patched" {
      nativeBuildInputs = [gnused python3];
    } ''
      cp -R ${ppqsortRaw} $out
      chmod -R u+w $out
      sed -i 's#https://github.com/cpm-cmake/CPM.cmake/releases/download/v''${CPM_DOWNLOAD_VERSION}/CPM.cmake#file://${cpmSrc}#' $out/cmake/CPM.cmake
      ${lib.optionalString stdenv.isDarwin ''
                python <<'PY'
        import os
        from pathlib import Path

        header = Path(os.environ["out"]) / "include" / "ppqsort.h"
        text = header.read_text()

        text = text.replace(
            """#ifdef FORCE_CPP
            #include "ppqsort/parallel/cpp/mainloop_par.h"
        #else
            #ifdef _OPENMP
                #include "ppqsort/parallel/openmp/mainloop_par.h"
            #else
                #include "ppqsort/parallel/cpp/mainloop_par.h"
            #endif
        #endif
        """,
            """#ifdef FORCE_CPP
            #include "ppqsort/parallel/cpp/mainloop_par.h"
        #else
            #ifdef __APPLE__
                // libc++ on current Darwin targets does not expose the C++20
                // jthread/stop_token API needed by PPQSort's parallel backend.
                // Keep Tracy enabled, but fall back to the sequential sorter here.
            #elif defined(_OPENMP)
                #include "ppqsort/parallel/openmp/mainloop_par.h"
            #else
                #include "ppqsort/parallel/cpp/mainloop_par.h"
            #endif
        #endif
        """,
        )

        text = text.replace(
            """       } else if constexpr (execution::_is_same_decay_v<ExecutionPolicy, decltype(execution::par)>) {
                   par_ppqsort(std::forward<T>(args)...);
               } else if constexpr (execution::_is_same_decay_v<ExecutionPolicy, decltype(execution::seq_force_branchless)>) {
                   seq_ppqsort<true>(std::forward<T>(args)...);
               } else if constexpr (execution::_is_same_decay_v<ExecutionPolicy, decltype(execution::par_force_branchless)>) {
                   par_ppqsort<true>(std::forward<T>(args)...);
        """,
            """       } else if constexpr (execution::_is_same_decay_v<ExecutionPolicy, decltype(execution::par)>) {
        #ifdef __APPLE__
                   seq_ppqsort(std::forward<T>(args)...);
        #else
                   par_ppqsort(std::forward<T>(args)...);
        #endif
               } else if constexpr (execution::_is_same_decay_v<ExecutionPolicy, decltype(execution::seq_force_branchless)>) {
                   seq_ppqsort<true>(std::forward<T>(args)...);
               } else if constexpr (execution::_is_same_decay_v<ExecutionPolicy, decltype(execution::par_force_branchless)>) {
        #ifdef __APPLE__
                   seq_ppqsort<true>(std::forward<T>(args)...);
        #else
                   par_ppqsort<true>(std::forward<T>(args)...);
        #endif
        """,
        )

        header.write_text(text)
        PY
      ''}
    '';
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

      ${lib.optionalString (enableTracing && tracySrc != null) ''
        rmdir $sourceRoot/third_party/tracy 2>/dev/null || true
        cp -r ${tracySrc} $sourceRoot/third_party/tracy
        chmod -R u+w $sourceRoot/third_party/tracy
      ''}
    '';

    nativeBuildInputs =
      [cmake ninja python3 git]
      ++ lib.optionals enableTracing [pkg-config];

    buildInputs = lib.optionals enableTracing [zstd];

    # Tracy's capture server code triggers _FORTIFY_SOURCE buffer overflow
    # detection with Nix's default hardening flags.
    hardeningDisable = lib.optionals enableTracing ["fortify"];

    cmakeFlags =
      [
        "-DIREE_BUILD_COMPILER=OFF"
        "-DIREE_BUILD_TESTS=OFF"
        "-DIREE_BUILD_SAMPLES=OFF"

        # HAL drivers for CPU execution
        "-DIREE_HAL_DRIVER_DEFAULTS=OFF"
        "-DIREE_HAL_DRIVER_LOCAL_SYNC=ON"
        "-DIREE_HAL_DRIVER_LOCAL_TASK=ON"

        # Executable loaders (all three needed for full platform support)
        "-DIREE_HAL_EXECUTABLE_LOADER_DEFAULTS=OFF"
        "-DIREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF=ON"
        "-DIREE_HAL_EXECUTABLE_LOADER_SYSTEM_LIBRARY=ON"
        "-DIREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE=ON"

        # Executable plugins
        "-DIREE_HAL_EXECUTABLE_PLUGIN_SYSTEM_LIBRARY=ON"

        # Static libraries
        "-DBUILD_SHARED_LIBS=OFF"

        # Disable optional features not needed for our use case
        "-DIREE_ENABLE_CPUINFO=OFF"
        "-DIREE_BUILD_TRACY=OFF"
        # Prevent FetchContent from trying to git-clone libbacktrace during
        # the build (nix sandboxes have no network access). Defaults to ON
        # on Linux.
        "-DIREE_ENABLE_LIBBACKTRACE=OFF"
      ]
      ++ lib.optionals enableTracing [
        "-DIREE_ENABLE_RUNTIME_TRACING=ON"
        "-DIREE_TRACING_MODE=2"
        "-DIREE_TRACING_PROVIDER=tracy"
        "-DIREE_BUILD_TRACY=ON"
        "-DFETCHCONTENT_SOURCE_DIR_CAPSTONE=${capstoneSrc}"
        "-DFETCHCONTENT_SOURCE_DIR_PPQSORT=${ppqsortSrc}"
        "-DCPM_PackageProject.cmake_SOURCE=${packageProjectSrc}"
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

      ${lib.optionalString enableTracing ''
        # Install iree-tracy-capture built from the same Tracy source
        find $NIX_BUILD_TOP -name iree-tracy-capture -type f -executable | head -1 | while read f; do
          mkdir -p $out/bin
          cp "$f" $out/bin/
        done
      ''}

      runHook postInstall
    '';

    meta = with lib; {
      description = "IREE runtime C library for executing compiled ML models";
      homepage = "https://iree.dev";
      license = licenses.asl20;
    };
  }
