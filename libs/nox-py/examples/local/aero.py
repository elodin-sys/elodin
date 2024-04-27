import polars as pl
import jax.numpy as jnp
import jax
from jax.scipy.ndimage import map_coordinates
import polars.selectors as cs
import io
import pathlib

aero_data_path = f"{pathlib.Path(__file__).parent}/aero_data.csv"


def read_aero_csv(data: str) -> pl.DataFrame:
    clean_data = io.StringIO()
    for line in data.split("\n"):
        if "REFERENCE AREA" in line:
            line = line.strip('"').split()
            # Parse "REFERENCE AREA =   14.08130  REFERENCE LENGTH =    4.23400  XMC/LBASE =    0.44997"
            A, L, Xmc = float(line[3]), float(line[7]), float(line[10])
        else:
            clean_data.write(line + "\n")
    clean_data.seek(0)
    df = pl.read_csv(
        clean_data,
        comment_prefix="#",
        dtypes={
            "IJ": pl.Int32,
            "Mach": pl.Float64,
            "Alphac": pl.Float64,
            "Phi": pl.Float64,
            "CA": pl.Float64,
            "CN": pl.Float64,
            "CY": pl.Float64,
            "Cm": pl.Float64,
            "Cn": pl.Float64,
            "Cl": pl.Float64,
            "CZR": pl.Float64,
            "CYR": pl.Float64,
            "CmR": pl.Float64,
            "CnR": pl.Float64,
            "XCP(CN)/L": pl.Float64,
            "XCP(CY)/L": pl.Float64,
            "XCP(CZR)/L": pl.Float64,
            "XCP(CYR)/L": pl.Float64,
            "Delta11": pl.Float64,
            "Delta12": pl.Float64,
            "Delta13": pl.Float64,
            "Delta14": pl.Float64,
        },
    )
    df = df.select(cs.by_dtype([pl.Int32, pl.Float64])).drop_nulls()
    df = df.with_columns(
        pl.Series("A", [A] * len(df)),
        pl.Series("L", [L] * len(df)),
        pl.Series("Xmc", [Xmc] * len(df)),
    )
    df = df.sort(["Mach", "Delta11", "Delta12", "Delta13", "Delta14", "Phi", "Alphac"])
    clean_data.close()
    return df


def aero_interp_table(df: pl.DataFrame) -> jax.Array:
    coefs = ["Cl", "CnR", "CmR", "CA", "CZR", "CYR", "XCP(CZR)/L", "XCP(CYR)/L"]
    aero = jnp.array(
        [
            [
                [
                    [
                        [
                            df.group_by(["Alphac"], maintain_order=True)
                            .agg(pl.col(coefs).min())
                            .select(pl.col(coefs))
                            .to_numpy()
                            for _, df in df.group_by(["Phi"], maintain_order=True)
                        ]
                        for _, df in df.group_by([f2], maintain_order=True)
                    ]
                    for _, df in df.group_by([f1], maintain_order=True)
                ]
                for _, df in df.filter(filter).group_by(["Mach"], maintain_order=True)
            ]
            # generate two different tables for 2/4 and 1/3 fin deflection:
            for f1, f2, filter in [
                (
                    "Delta12",
                    "Delta14",
                    (pl.col("Delta11") == 0.0) & (pl.col("Delta13") == 0.0),
                ),
                (
                    "Delta11",
                    "Delta13",
                    (pl.col("Delta12") == 0.0) & (pl.col("Delta14") == 0.0),
                ),
            ]
        ]
    )
    aero = aero.transpose(6, 0, 1, 2, 3, 4, 5)
    return aero


# coverts `val` to a coordinate along some series `s`
def to_coord(s: pl.Series, val: jax.Array) -> jax.Array:
    s_min = s.min()
    s_max = s.max()
    s_count = len(s.unique())
    return (val - s_min) * (s_count - 1) / jnp.clip(s_max - s_min, 1e-06)


def test_interpolation():
    with open(aero_data_path) as file:
        aero_data = file.read()
    df = read_aero_csv(aero_data)
    aero = aero_interp_table(df)

    mach = 0.8
    angle_of_attack = 2.0
    roll_angle = 0.0
    d12 = 10.0
    d14 = d12

    coords = jnp.array(
        [
            to_coord(df["Mach"], mach),
            to_coord(df["Delta12"], d12),
            to_coord(df["Delta14"], d14),
            to_coord(df["Phi"], jnp.abs(roll_angle)),
            to_coord(df["Alphac"], jnp.abs(angle_of_attack)),
        ]
    )
    coefs = jnp.array(
        [map_coordinates(coef[0], coords, 1, mode="nearest") for coef in aero]
    )
    print(coefs)


if __name__ == "__main__":
    test_interpolation()
