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
command -v cargo
cargo --version
command -v rustc
rustc --version
command -v dist
dist --version
command -v git-lfs
git lfs version
command -v nix
nix --version
echo "macOS runner environment: PASS ($(uname -m))"
