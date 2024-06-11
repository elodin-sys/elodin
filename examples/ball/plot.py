import polars as pl
from sim import world, system
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--ticks", type=int, default=1200)
parser.add_argument("--export-dir", type=str, default=None)
args = parser.parse_args()

exec = world(args.seed).build(system())
exec.run(args.ticks)

if args.export_dir:
    exec.write_to_dir(args.export_dir)
else:
    fig, ax = plt.subplots()
    df = exec.history()
    df = df.sort("tick").select(["tick", "world_pos"]).drop_nulls()
    df = df.with_columns(
        pl.col("world_pos").arr.get(4).alias("x"),
        pl.col("world_pos").arr.get(5).alias("y"),
        pl.col("world_pos").arr.get(6).alias("z"),
    )
    distance = np.linalg.norm(df.select(["x", "y", "z"]).to_numpy(), axis=1)
    df = df.with_columns(pl.Series(distance).alias("distance"))
    ax.plot(df["tick"], df["distance"])
    plt.show()
