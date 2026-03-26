#!/usr/bin/env bash

set -euo pipefail

mkdir -p ./nasa_spice_data

curl -fL https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls \
    -o ./nasa_spice_data/naif0012.tls
curl -fL https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp \
    -o ./nasa_spice_data/de440.bsp
curl -fL https://naif.jpl.nasa.gov/pub/naif/VOYAGER/kernels/spk/Voyager_1.a54206u_V0.2_merged.bsp \
    -o ./nasa_spice_data/Voyager_1.a54206u_V0.2_merged.bsp
curl -fL https://naif.jpl.nasa.gov/pub/naif/VOYAGER/kernels/spk/Voyager_2.m05016u.merged.bsp \
    -o ./nasa_spice_data/Voyager_2.m05016u.merged.bsp
