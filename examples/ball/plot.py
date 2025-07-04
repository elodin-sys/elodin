import argparse

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

df = exec.history("ball.world_pos")
df = df.with_columns(
    pl.col("ball.world_pos").arr.get(4).alias("ball.x"),
    pl.col("ball.world_pos").arr.get(5).alias("ball.y"),
    pl.col("ball.world_pos").arr.get(6).alias("ball.z"),
)
print(df)
distance = np.linalg.norm(df.select(["ball.x", "ball.y", "ball.z"]).to_numpy(), axis=1)
df = df.with_columns(pl.Series(distance).alias("distance"))
ticks = np.arange(df.shape[0])
fig, ax = plt.subplots()
ax.plot(ticks, df["distance"])
plt.show()
