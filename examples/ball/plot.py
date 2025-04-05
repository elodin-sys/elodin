import argparse

import elodin as el
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sim import system, world

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--ticks", type=int, default=1200)
args = parser.parse_args()

exec = world(args.seed).build(system())
exec.run(args.ticks)

fig, ax = plt.subplots()
df = exec.history("world_pos", el.EntityId(1))
df = df.with_columns(
    pl.col("world_pos").arr.get(4).alias("x"),
    pl.col("world_pos").arr.get(5).alias("y"),
    pl.col("world_pos").arr.get(6).alias("z"),
)
distance = np.linalg.norm(df.select(["x", "y", "z"]).to_numpy(), axis=1)
df = df.with_columns(pl.Series(distance).alias("distance"))
ticks = np.arange(df.shape[0])
ax.plot(ticks, df["distance"])
plt.show()
