# Voyager

This example propagates Voyager 1 and Voyager 2 under gravity from the Sun and
major planets, and compares them with NASA SPICE trajectories.

It is an educational model, not a full mission reconstruction. Chapter 1 uses
direct Newtonian gravity. Chapter 2 also accounts for the acceleration of the
Sun-centered origin.

The editor shows the propagated and SPICE trajectories plus
`position_error_km` and `velocity_error_mps` for both probes.

## Setup

Run from the repository root:

```bash
uv venv --python=3.13 python-env
source python-env/bin/activate
uv pip install spiceypy
cd examples/voyager
./download_spice_data.sh
cd ../..
```

## Run

Chapter 1:

```bash
python examples/voyager/main.py run
```

[Chapter 1 notes](docs/chapter_1.md)

Chapter 2:

```bash
python examples/voyager/chapter_2.py run
```

[Chapter 2 notes](docs/chapter_2.md)

## Validation

The interactive chapters use long-span merged Voyager SPKs for visualization.
The separate headless validation uses four reconstructed Jupiter and Saturn
encounter SPKs, initializing and scoring from the same solution. It does not
change the chapter physics or replace the interactive reference data.

[Truth-reference contract](docs/truth_reference.md)

Chapter 2 reduces position disagreement by roughly 72–74% and velocity
disagreement by about 78% in the original 400-day comparison. On the clean
pre-maneuver Saturn checkpoints, the reductions are 97.5% for Voyager 1 and
68.7% for Voyager 2.

These are improvements in the measured comparisons, not navigation-grade
accuracy. The probes still do not reproduce the complete gravity-assist path or
slingshot sequence.

If SPICE reports a missing kernel, rerun `download_spice_data.sh`. Set `DB_PATH`
to a new location when a fresh simulation database is needed.
