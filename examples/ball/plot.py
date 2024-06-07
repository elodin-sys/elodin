import polars as pl
from sim import world, system
import matplotlib.pyplot as plt
import numpy as np

exec = world().build(system())
exec.run(ticks=1200)
df = exec.history()

fig, ax = plt.subplots()

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
