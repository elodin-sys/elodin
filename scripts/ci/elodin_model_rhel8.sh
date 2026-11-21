#!/usr/bin/env bash
# Build the passive AOT model runtime against glibc 2.28 and consume both link variants.
set -euo pipefail

cd "$(dirname "$0")/../.."
model=${1:-target/elodin-model-rhel8/model}
output=${2:-target/elodin-model-rhel8/output}
image=${ELODIN_RHEL8_IMAGE:-rockylinux:8}
platform=${ELODIN_RHEL8_PLATFORM:-linux/amd64}

case "$model" in
  /*) model_abs=$model ;;
  *) model_abs=$PWD/$model ;;
esac
case "$output" in
  /*) output_abs=$output ;;
  *) output_abs=$PWD/$output ;;
esac
mkdir -p "$output_abs"
mkdir -p "${HOME}/.cargo/registry"

docker run --rm \
  --platform "$platform" \
  -v "$PWD:/src:ro" \
  -v "$model_abs:/model:ro" \
  -v "$output_abs:/out" \
  -v "${HOME}/.cargo/registry:/root/.cargo/registry" \
  "$image" bash -lc '
    set -euo pipefail
    dnf install -y gcc gcc-c++ git curl make perl python3 binutils >/dev/null
    curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.98.0
    export PATH=/root/.cargo/bin:$PATH
    export RUSTUP_TOOLCHAIN=1.98.0
    export CARGO_TARGET_DIR=/out/target
    cd /src
    mkdir -p /out/model
    cp /model/step.stablehlo.mlir /model/state.bin /model/schema.json /model/model-manifest.json /out/model/
    workers=$(python3 -c "import json; print(json.load(open(\"/out/model/schema.json\"))[\"workers\"])")
    cargo run --release --locked -p cranelift-mlir --example compile-object -- \
      /out/model/step.stablehlo.mlir /out/model/step.o elodin_model_kernel "$workers" generic
    python3 -c "
import hashlib,json,os
root=\"/out/model\"
p=os.path.join(root,\"schema.json\")
s=json.load(open(p))
s[\"target\"]=json.load(open(os.path.join(root,\"step.o.target.json\")))
open(p,\"w\").write(json.dumps(s,indent=2))
m=json.load(open(os.path.join(root,\"model-manifest.json\")))
m[\"target\"]=s[\"target\"]
for name in (\"step.stablehlo.mlir\",\"step.o\",\"state.bin\",\"schema.json\"):
    data=open(os.path.join(root,name),\"rb\").read()
    m[\"files\"][name]={\"bytes\":len(data),\"sha256\":hashlib.sha256(data).hexdigest()}
open(os.path.join(root,\"model-manifest.json\"),\"w\").write(json.dumps(m,indent=2))
build={\"format\":\"elodin.model-build\",\"version\":1,\"model\":m,
       \"build\":{\"backend\":\"aot\",\"target\":\"x86_64-unknown-linux-gnu\",\"glibc\":\"2.28\"}}
open(os.path.join(root,\"build-info.json\"),\"w\").write(json.dumps(build,separators=(\",\",\":\")))
"
    export ELODIN_MODEL_DIR=/out/model
    export ELODIN_MODEL_BACKEND=aot
    export ELODIN_MODEL_BUILD_INFO_PATH=/out/model/build-info.json
    cargo build --release --locked -p elodin-model
    cp /out/target/release/libelodin_model.so /out/
    cp /out/target/release/libelodin_model.a /out/
    python3 scripts/ci/check_elodin_model_elf.py /out/libelodin_model.so

    g++ -std=c++17 libs/elodin-model/tests/cpp/passive.cpp \
      -Ilibs/elodin-model/include -L/out -lelodin_model \
      -Wl,-rpath,/out -o /out/passive_shared
    /out/passive_shared

    g++ -std=c++17 libs/elodin-model/tests/cpp/passive.cpp \
      -Ilibs/elodin-model/include /out/libelodin_model.a \
      -ldl -lpthread -lm -lrt -lutil -o /out/passive_static
    /out/passive_static
  '
