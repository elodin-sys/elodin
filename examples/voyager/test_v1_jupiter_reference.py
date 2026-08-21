from pathlib import Path

import pytest

spice = pytest.importorskip("spiceypy")

SPICE_DIR = Path(__file__).resolve().parent / "nasa_spice_data"
LEAP_SECONDS = SPICE_DIR / "naif0012.tls"
ENCOUNTER_KERNEL = SPICE_DIR / "vgr1_jup230.bsp"
START_UTC = "1979-02-01T00:00:00"
CHECKPOINTS_S = (0, 5 * 86400, 10 * 86400)


def test_v1_jupiter_kernel_covers_fixed_validation_checkpoints():
    """Keep the first reconstructed validation case tied to one SPK segment."""
    if not LEAP_SECONDS.is_file() or not ENCOUNTER_KERNEL.is_file():
        pytest.skip("run examples/voyager/download_spice_data.sh first")

    spice.kclear()
    try:
        spice.furnsh(str(LEAP_SECONDS))
        spice.furnsh(str(ENCOUNTER_KERNEL))
        start_et = spice.utc2et(START_UTC)

        for elapsed_s in CHECKPOINTS_S:
            handle, descriptor, segment_id = spice.spksfs(-31, start_et + elapsed_s, 256)
            target, center, frame, segment_type, *_ = spice.spkuds(descriptor)

            assert handle != 0
            assert target == -31
            assert center == 5
            assert spice.frmnam(frame) == "J2000"
            assert segment_type == 1
            assert segment_id.strip() == "vgr1.jup230.nio"
    finally:
        spice.kclear()
