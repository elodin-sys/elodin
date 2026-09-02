#!/usr/bin/env bash

set -euo pipefail

mkdir -p ./nasa_spice_data

curl -fL https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls \
    -o ./nasa_spice_data/naif0012.tls
curl -fL https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp \
    -o ./nasa_spice_data/de440.bsp
curl -fL https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc \
    -o ./nasa_spice_data/gm_de440.tpc
curl -fL https://naif.jpl.nasa.gov/pub/naif/VOYAGER/kernels/spk/Voyager_1.a54206u_V0.2_merged.bsp \
    -o ./nasa_spice_data/Voyager_1.a54206u_V0.2_merged.bsp
curl -fL https://naif.jpl.nasa.gov/pub/naif/VOYAGER/kernels/spk/Voyager_2.m05016u.merged.bsp \
    -o ./nasa_spice_data/Voyager_2.m05016u.merged.bsp
curl -fL https://naif.jpl.nasa.gov/pub/naif/VOYAGER/kernels/spk/vgr1_jup230.bsp \
    -o ./nasa_spice_data/vgr1_jup230.bsp

encounter_checksum="e1ea3f72f19b15508bc45979771a36a97d02f33056b76867d444304cb82205c9"
encounter_kernel="./nasa_spice_data/vgr1_jup230.bsp"

if command -v sha256sum >/dev/null 2>&1; then
    echo "${encounter_checksum}  ${encounter_kernel}" | sha256sum -c -
elif command -v shasum >/dev/null 2>&1; then
    echo "${encounter_checksum}  ${encounter_kernel}" | shasum -a 256 -c -
else
    echo "error: need sha256sum or shasum to verify ${encounter_kernel}" >&2
    exit 1
fi
