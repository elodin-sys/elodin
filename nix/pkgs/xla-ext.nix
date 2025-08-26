{
  pkgs,
  stdenv,
  lib,
  bazelisk,
  apple-sdk_15,
  darwinMinVersionHook,
  ...
}:
with pkgs; let
  extSrc = ../../libs/noxla/extension/.;
in
  stdenv.mkDerivation {
    name = "xla-ext";

    src = fetchzip {
      url = "https://github.com/openxla/xla/archive/2a6015f068e4285a69ca9a535af63173ba92995b.tar.gz";
      sha256 = "sey2yXF3tofTgmS1wXJZS6HwngzBYzktl/QRbMZfrYE=";
    };

    outputs = ["out"];

    buildInputs = [
      bazelisk
      python3
    ];

    nativeBuildInputs = [
    ];

    dontConfigure = true;

    unpackPhase =
      ''
        # Our build directory
        export HOME=$TMP
        # The unpacked directory is read-only so we need to
        # set things up in our build directory.
        mkdir xla
        # We only modify the xla sub-directory so symlink the rest.
        ln -s $src/* xla/
        # And don't forget to symlink the dot files!
        ln -s $src/.??* xla/
        # We need a fresh directory to symlink our extension into
        # since the linked xla directory will be read-only.
        rm xla/xla
        mkdir xla/xla
        # Symlink the contents of the xla sub-directory
        ln -s $src/xla/* xla/xla
        # And add ourselves
        ln -s ${extSrc} xla/xla/extension
      ''
      + lib.optionalString stdenv.isDarwin
      ''
        # Pre-fetch dependencies for patching
        cd xla
        bazelisk fetch @build_bazel_apple_support//:apple_support
        cd `bazelisk info output_base`/external/build_bazel_apple_support
      '';

    patches =
      if stdenv.isDarwin
      then [
        # Failure to patch Apple support code results in C++ build failures.
        # This is because the command-line tools SDK location does not have
        # the required C++ headers.
        ./build_bazel_apple_support.patch
      ]
      else [];

    buildPhase =
      lib.optionalString stdenv.isDarwin
      ''
        # We must use the non-Nix macOS SDK and XCode installation
        # since XLA insists on it. This requires reverting relevant
        # environment variables from their Nix values.
        export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
        export SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
        export SDK_VER=`xcrun --show-sdk-version -sdk macosx`
        echo "macOS SDK version: $SDK_VER"
      ''
      + ''
        cd $HOME/xla
        bazelisk build \
          --enable_workspace \
          --experimental_cc_static_library \
          --macos_sdk_version=$SDK_VER \
          --verbose_failures \
          xla/extension:tarball
        find . -name xla_extension.tar.gz
      '';

    installPhase = ''
      runHook preInstall
      mkdir -p $out
      cd $out
      tar zxf $HOME/xla/bazel-bin/xla/extension/xla_extension.tar.gz
      runHook postInstall
    '';
  }
