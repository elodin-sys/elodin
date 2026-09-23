#!/usr/bin/env bash
set -Eeuo pipefail
[[ "$(uname -s)" == Darwin ]] || {
    echo 'Expected macOS' >&2
    exit 1
}
[[ "$(uname -m)" == arm64 ]] || {
    echo 'Expected native Apple silicon' >&2
    exit 1
}

sw_vers
xcodebuild -version
xcode-select -p
test -d /Applications/Xcode.app/Contents/Developer
brew --prefix ffmpeg@8
brew --prefix protobuf
command -v pkg-config
PKG_CONFIG_PATH="$(brew --prefix ffmpeg@8)/lib/pkgconfig" pkg-config --modversion libavutil
command -v cargo
cargo --version
command -v rustc
rustc --version
command -v dist
dist --version
command -v git-lfs
git lfs version
echo "macOS runner environment: PASS ($(uname -m))"
