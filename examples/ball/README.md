# Bouncing Ball Simulation

This demo tracks a steel ball falling through atmosphere. It combines gravity, drag, and bounce to show how Elodin systems can be chained together.

## Run the demo

```
elodin run main.py
```


## Analyze the trajectory

`plot.py` reuses the world and system from `sim.py`, runs them headless, and renders a simple Matplotlib chart:


```
pip install -U matplotlib polars
python3 plot.py --ticks 1200 --seed 0
```
