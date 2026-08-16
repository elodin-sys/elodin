#!/usr/bin/env bash

set -euo pipefail

mkdir -p ./nasa_spice_data

download() {
    local name="$1"
    local url="$2"
    local expected_sha256="$3"
    local output="./nasa_spice_data/${name}"
    local temporary="${output}.download"

    if [[ -f "${output}" ]] && echo "${expected_sha256}  ${output}" | sha256sum --check --status; then
        echo "verified ${name}"
        return
    fi

    curl -fL "${url}" -o "${temporary}"
    echo "${expected_sha256}  ${temporary}" | sha256sum --check --status
    mv "${temporary}" "${output}"
    echo "downloaded and verified ${name}"
}

download \
    naif0012.tls \
    https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls \
    678e32bdb5a744117a467cd9601cd6b373f0e9bc9bbde1371d5eee39600a039b
download \
    de440.bsp \
    https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp \
    a4ce9bf9b3282becc9f4b2ac3cebe03a2ae7599981aabd7265fd8482fff7c4b5
download \
    Voyager_1.a54206u_V0.2_merged.bsp \
    https://naif.jpl.nasa.gov/pub/naif/VOYAGER/kernels/spk/Voyager_1.a54206u_V0.2_merged.bsp \
    47c6f2be03668b50a1efb5f96978a2b68b2b501dae6f585841b1569baa3f4311
download \
    Voyager_2.m05016u.merged.bsp \
    https://naif.jpl.nasa.gov/pub/naif/VOYAGER/kernels/spk/Voyager_2.m05016u.merged.bsp \
    ce66cba12cf77bf3a1097f44142ef978f46656788ed08f9052238a102ed2e706
download \
    vgr1_jup230.bsp \
    https://naif.jpl.nasa.gov/pub/naif/VOYAGER/kernels/spk/vgr1_jup230.bsp \
    e1ea3f72f19b15508bc45979771a36a97d02f33056b76867d444304cb82205c9
download \
    vgr2_jup230.bsp \
    https://naif.jpl.nasa.gov/pub/naif/VOYAGER/kernels/spk/vgr2_jup230.bsp \
    9c00be3c83915f6c1fd8448d9266420e3c462e9d78e4c32b17145dac81529d5a
download \
    vgr1_sat337.bsp \
    https://naif.jpl.nasa.gov/pub/naif/VOYAGER/kernels/spk/vgr1_sat337.bsp \
    f451f9f095bc4ad175cde701367cbf1ccc061dd706ad0008cda75fd939d94efa
download \
    vgr2_sat337.bsp \
    https://naif.jpl.nasa.gov/pub/naif/VOYAGER/kernels/spk/vgr2_sat337.bsp \
    1d15debf7bdc6c6ba4c0462b5f4f6af85ddf9948735f434792a1c41418bb39a8
