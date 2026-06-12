{
  description = "World Mesh — large-scale Bevy terrain renderer";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {inherit system;};

      commonNativeBuildInputs = with pkgs; [
        rustc
        cargo
        rust-analyzer
        clippy
        rustfmt
        pkg-config
        clang
        llvmPackages.libclang
        cmake
      ];

      commonBuildInputs = with pkgs;
        [
          openssl
          zlib
        ]
        ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
          libiconv
        ]
        ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
          alsa-lib
          udev
          vulkan-loader
          libxkbcommon
          wayland
          xorg.libX11
          xorg.libXcursor
          xorg.libXi
          xorg.libXrandr
        ];
    in {
      devShells.default = pkgs.mkShell {
        nativeBuildInputs = commonNativeBuildInputs;
        buildInputs = commonBuildInputs;

        LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

        BINDGEN_EXTRA_CLANG_ARGS = "-I${pkgs.llvmPackages.libclang.lib}/lib/clang/${pkgs.llvmPackages.libclang.version}/include";

        shellHook = ''
          ${pkgs.lib.optionalString pkgs.stdenv.isLinux ''
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath commonBuildInputs}:$LD_LIBRARY_PATH"
          ''}
          echo "world_mesh dev shell"
          echo "  rust:  $(rustc --version)"
          echo "  cargo: $(cargo --version)"
        '';
      };

      formatter = pkgs.nixpkgs-fmt;
    });
}
