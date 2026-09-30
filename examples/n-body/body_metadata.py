"""Body metadata shared by the simulation and its Python-authored schematic."""

import csv
from dataclasses import dataclass
from pathlib import Path

AU_IN_KM = 149_597_870.7
CSV_PATHS: tuple[Path, ...] = (
    Path(__file__).with_name("planets_truth.csv"),
    # Path(__file__).with_name("moons_truth.csv"),
)
SUN_MASS_SOLAR = 1.0
SUN_RADIUS_KM = 696_340.0
SUN_COLOR = (255, 220, 120)
TRUTH_COLOR = (180, 180, 180)

PLANET_ICON = "public"
MOON_ICON = "circle"


@dataclass(frozen=True)
class BodyMeta:
    mass_solar: float
    radius_km: float
    color_rgb: tuple[int, int, int]
    icon: str


@dataclass(frozen=True)
class Body:
    name: str
    naif_id: int
    meta: BodyMeta


BODY_META: dict[str, BodyMeta] = {
    "mercury": BodyMeta(1.6605e-7, 2439.7, (185, 185, 185), PLANET_ICON),
    "venus": BodyMeta(2.4478e-6, 6051.8, (240, 200, 120), PLANET_ICON),
    "earth": BodyMeta(3.0035e-6, 6371.0, (90, 150, 255), PLANET_ICON),
    "mars": BodyMeta(3.2272e-7, 3389.5, (255, 120, 80), PLANET_ICON),
    "jupiter": BodyMeta(9.5459e-4, 69911.0, (210, 170, 130), PLANET_ICON),
    "saturn": BodyMeta(2.8588e-4, 58232.0, (230, 210, 150), PLANET_ICON),
    "uranus": BodyMeta(4.3662e-5, 25362.0, (150, 240, 255), PLANET_ICON),
    "neptune": BodyMeta(5.1514e-5, 24622.0, (90, 120, 255), PLANET_ICON),
    "pluto": BodyMeta(6.55e-9, 1188.3, (190, 180, 170), PLANET_ICON),
    "moon": BodyMeta(3.694e-8, 1737.4, (210, 210, 210), MOON_ICON),
    "mimas": BodyMeta(1.98e-11, 198.2, (170, 170, 170), MOON_ICON),
    "enceladus": BodyMeta(5.4e-10, 252.1, (190, 210, 230), MOON_ICON),
    "tethys": BodyMeta(3.09e-9, 531.1, (170, 170, 180), MOON_ICON),
    "dione": BodyMeta(5.5e-9, 561.7, (180, 180, 190), MOON_ICON),
    "rhea": BodyMeta(1.16e-8, 763.8, (190, 190, 200), MOON_ICON),
    "titan": BodyMeta(6.763e-8, 2574.7, (230, 180, 130), MOON_ICON),
    "hyperion": BodyMeta(2.8e-12, 135.0, (180, 170, 150), MOON_ICON),
    "iapetus": BodyMeta(9.05e-9, 734.5, (200, 180, 150), MOON_ICON),
    "phoebe": BodyMeta(4.2e-12, 106.5, (110, 110, 120), MOON_ICON),
    "helene": BodyMeta(1.3e-16, 17.6, (150, 150, 160), MOON_ICON),
    "telesto": BodyMeta(1.0e-15, 12.4, (150, 150, 160), MOON_ICON),
    "calypso": BodyMeta(6.0e-16, 10.7, (150, 150, 160), MOON_ICON),
    "methone": BodyMeta(1.0e-16, 1.6, (130, 130, 140), MOON_ICON),
    "polydeuces": BodyMeta(1.0e-16, 1.3, (130, 130, 140), MOON_ICON),
    "charon": BodyMeta(7.6e-10, 606.0, (170, 170, 170), MOON_ICON),
    "nix": BodyMeta(2.0e-14, 24.5, (150, 150, 160), MOON_ICON),
    "hydra": BodyMeta(2.0e-14, 31.0, (150, 150, 160), MOON_ICON),
    "kerberos": BodyMeta(1.0e-14, 12.0, (150, 150, 160), MOON_ICON),
    "styx": BodyMeta(1.0e-14, 8.0, (150, 150, 160), MOON_ICON),
}


def normalize_body_name(raw_name: str) -> str:
    body_name = raw_name.split(maxsplit=1)[-1].strip().lower()
    return body_name.replace("-", "_").replace(" ", "_")


def load_bodies(csv_paths: tuple[Path, ...] = CSV_PATHS) -> list[Body]:
    """Discover supported bodies in CSV order without loading JAX truth arrays."""
    names_by_id: dict[int, str] = {}
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                body_id = int(row["naif_id"])
                name = normalize_body_name(row["name"])
                existing = names_by_id.setdefault(body_id, name)
                if existing != name:
                    raise ValueError(
                        f"conflicting body names for naif_id={body_id}: {existing!r} vs {name!r}"
                    )
    bodies = [
        Body(name=name, naif_id=body_id, meta=BODY_META[name])
        for body_id, name in names_by_id.items()
        if name in BODY_META
    ]
    if not bodies:
        raise ValueError("no supported bodies found in configured CSV files")
    return bodies
