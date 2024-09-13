from typing import cast

import numpy as np
import polars as pl


def test_origin_drift(df: pl.DataFrame):
    df = df.sort("tick").select(["tick", "world_pos"]).drop_nulls()
    df = df.with_columns(
        pl.col("world_pos").arr.get(4).alias("x"),
        pl.col("world_pos").arr.get(5).alias("y"),
        pl.col("world_pos").arr.get(6).alias("z"),
    )
    distance = np.linalg.norm(df.select(["x", "y"]).to_numpy(), axis=1)
    df = df.with_columns(pl.Series(distance).alias("distance"))
    max_dist = cast(int, df["distance"].max())
    assert max_dist < 2
