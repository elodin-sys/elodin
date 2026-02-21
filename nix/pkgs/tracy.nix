{
  lib,
  stdenv,
  runCommand,
  patch,
  gnused,
  fetchurl,
  fetchFromGitHub,
  tracy,
  pugixml,
  curl,
  xorg,
}:
let
  tracySrc = fetchFromGitHub {
    owner = "wolfpld";
    repo = "tracy";
    rev = "v0.13.1";
    hash = "sha256-D4aQ5kSfWH9qEUaithR0W/E5pN5on0n9YoBHeMggMSE=";
  };

  zstdSrc = fetchFromGitHub {
    owner = "facebook";
    repo = "zstd";
    rev = "v1.5.7";
    hash = "sha256-tNFWIT9ydfozB8dWcmTMuZLCQmQudTFJIkSr0aG7S44=";
  };

  imguiSrc = fetchFromGitHub {
    owner = "ocornut";
    repo = "imgui";
    rev = "v1.92.5-docking";
    hash = "sha256-/jVT7+874LCeSF/pdNVTFoSOfRisSqxCJnt5/SGCXPQ=";
  };

  nfdSrc = fetchFromGitHub {
    owner = "btzy";
    repo = "nativefiledialog-extended";
    rev = "v1.2.1";
    hash = "sha256-GwT42lMZAAKSJpUJE6MYOpSLKUD5o9nSe9lcsoeXgJY=";
  };

  ppqsortSrc = fetchFromGitHub {
    owner = "GabTux";
    repo = "PPQSort";
    rev = "v1.0.6";
    hash = "sha256-HgM+p2QGd9C8A8l/VaEB+cLFDrY2HU6mmXyTNh7xd0A=";
  };

  packageProjectSrc = fetchFromGitHub {
    owner = "TheLartians";
    repo = "PackageProject.cmake";
    rev = "v1.11.1";
    hash = "sha256-E7WZSYDlss5bidbiWL1uX41Oh6JxBRtfhYsFU19kzIw=";
  };

  jsonSrc = fetchFromGitHub {
    owner = "nlohmann";
    repo = "json";
    rev = "v3.12.0";
    hash = "sha256-cECvDOLxgX7Q9R3IE86Hj9JJUxraDQvhoyPDF03B2CY=";
  };

  md4cSrc = fetchFromGitHub {
    owner = "mity";
    repo = "md4c";
    rev = "release-0.5.2";
    hash = "sha256-2/wi7nJugR8X2J9FjXJF1UDnbsozGoO7iR295/KSJng=";
  };

  base64Src = fetchFromGitHub {
    owner = "aklomp";
    repo = "base64";
    rev = "v0.5.2";
    hash = "sha256-dIaNfQ/znpAdg0/vhVNTfoaG7c8eFrdDTI0QDHcghXU=";
  };

  tidySrc = fetchFromGitHub {
    owner = "htacg";
    repo = "tidy-html5";
    rev = "5.8.0";
    hash = "sha256-vzVWQodwzi3GvC9IcSQniYBsbkJV20iZanF33A0Gpe0=";
  };

  usearchSrc = fetchFromGitHub {
    owner = "unum-cloud";
    repo = "usearch";
    rev = "v2.21.3";
    fetchSubmodules = true;
    hash = "sha256-7IylunAkVNceKSXzCkcpp/kAsI3XoqniHe10u3teUVA=";
  };

  cpmSrc = fetchurl {
    url = "https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.38.7/CPM.cmake";
    hash = "sha256-g+XrcbK7uLHyrTjxlQKHoFdiTjhcI49gh/lM38RK+cU=";
  };

  capstoneSrc = fetchFromGitHub {
    owner = "capstone-engine";
    repo = "capstone";
    rev = "6.0.0-Alpha5";
    hash = "sha256-18PTj4hvBw8RTgzaFGeaDbBfkxmotxSoGtprIjcEuVg=";
  };

  imguiPatched = runCommand "imgui-tracy-0.13.1-patched" {
    nativeBuildInputs = [patch];
  } ''
    cp -R ${imguiSrc} $out
    chmod -R u+w $out
    patch -d $out -p1 < ${tracySrc}/cmake/imgui-emscripten.patch
    patch -d $out -p1 < ${tracySrc}/cmake/imgui-loader.patch
  '';

  ppqsortPatched = runCommand "ppqsort-tracy-0.13.1-patched" {
    nativeBuildInputs = [patch gnused];
  } ''
    cp -R ${ppqsortSrc} $out
    chmod -R u+w $out
    sed -i 's#https://github.com/cpm-cmake/CPM.cmake/releases/download/v''${CPM_DOWNLOAD_VERSION}/CPM.cmake#file://${cpmSrc}#' $out/cmake/CPM.cmake
    patch -d $out -p1 < ${tracySrc}/cmake/ppqsort-nodebug.patch
  '';

  tidyPatched = runCommand "tidy-tracy-0.13.1-patched" {
    nativeBuildInputs = [patch];
  } ''
    cp -R ${tidySrc} $out
    chmod -R u+w $out
    patch -d $out -p1 < ${tracySrc}/cmake/tidy-cmake.patch
  '';
in
  (tracy.override {withWayland = false;}).overrideAttrs (oldAttrs: {
    version = "0.13.1";
    src = tracySrc;

    buildInputs =
      (oldAttrs.buildInputs or [])
      ++ [
        pugixml
        curl
      ]
      ++ lib.optionals stdenv.isLinux [
        xorg.libX11
        xorg.libXrandr
        xorg.libXcursor
        xorg.libXi
      ];

    postPatch =
      (oldAttrs.postPatch or "")
      + ''
        sed -i '/imgui-emscripten.patch/d' cmake/vendor.cmake
        sed -i '/imgui-loader.patch/d' cmake/vendor.cmake
        sed -i '/ppqsort-nodebug.patch/d' cmake/vendor.cmake
        sed -i '/tidy-cmake.patch/d' cmake/vendor.cmake
      '';

    cmakeFlags =
      (builtins.filter (f: f != "-DDOWNLOAD_CAPSTONE=off") (oldAttrs.cmakeFlags or []))
      ++ [
        (lib.cmakeFeature "CPM_zstd_SOURCE" "${zstdSrc}")
        (lib.cmakeFeature "CPM_ImGui_SOURCE" "${imguiPatched}")
        (lib.cmakeFeature "CPM_nfd_SOURCE" "${nfdSrc}")
        (lib.cmakeFeature "CPM_PPQSort_SOURCE" "${ppqsortPatched}")
        (lib.cmakeFeature "CPM_PackageProject.cmake_SOURCE" "${packageProjectSrc}")
        (lib.cmakeFeature "CPM_json_SOURCE" "${jsonSrc}")
        (lib.cmakeFeature "CPM_md4c_SOURCE" "${md4cSrc}")
        (lib.cmakeFeature "CPM_base64_SOURCE" "${base64Src}")
        (lib.cmakeFeature "CPM_tidy_SOURCE" "${tidyPatched}")
        (lib.cmakeFeature "CPM_usearch_SOURCE" "${usearchSrc}")
        (lib.cmakeFeature "CPM_capstone_SOURCE" "${capstoneSrc}")
        (lib.cmakeFeature "DOWNLOAD_CAPSTONE" "ON")
      ];
  })
