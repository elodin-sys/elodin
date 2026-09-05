"""DE440 gravitational parameters used by the Voyager example."""

from pathlib import Path

import spiceypy as spice

SPICE_DIR = Path(__file__).resolve().parent / "nasa_spice_data"
GM_KERNEL = SPICE_DIR / "gm_de440.tpc"
SPICE_NAMES = (
    "SUN",
    "MERCURY BARYCENTER",
    "VENUS BARYCENTER",
    "EARTH",
    "MARS BARYCENTER",
    "JUPITER BARYCENTER",
    "SATURN BARYCENTER",
    "URANUS BARYCENTER",
    "NEPTUNE BARYCENTER",
)


def _load_de440_gravity_parameters() -> dict[str, float]:
    """Read GM values from NAIF's DE440 text kernel and convert to m^3/s^2."""
    if not GM_KERNEL.exists():
        raise FileNotFoundError(
            "missing DE440 GM kernel; run examples/voyager/download_spice_data.sh first: "
            f"{GM_KERNEL}"
        )

    spice.furnsh(str(GM_KERNEL))
    return {
        name: float(spice.bodvrd(name, "GM", 1)[1][0]) * 1.0e9
        for name in SPICE_NAMES
    }


DE440_GM_M3_S2 = _load_de440_gravity_parameters()
