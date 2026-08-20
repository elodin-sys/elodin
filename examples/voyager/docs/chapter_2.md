# Chapter 2 — Heliocentric relative dynamics

Chapter 2 keeps Chapter 1 intact and changes one thing: it accounts for the
acceleration of the Sun-centered origin.

## Run it

```bash
VOYAGER_DYNAMICS_CHAPTER=2 elodin editor examples/voyager/main.py
```

Headless:

```bash
python examples/voyager/chapter_2.py run
```

## Why the correction is needed

SPICE supplies Voyager and planetary states relative to the Sun in
`ECLIPJ2000`. The axes are nonrotating, but the Sun-centered origin is not
inertial because the planets accelerate the Sun.

For heliocentric probe position

```text
r = R_probe - R_sun
```

we have

```text
r'' = R_probe'' - R_sun''
```

so a planet at heliocentric position `r_i` contributes

```text
mu_i * ((r_i - r) / |r_i - r|^3 - r_i / |r_i|^3)
```

The first term is the planet's direct acceleration of Voyager; the second
subtracts that planet's acceleration of the Sun. The Sun's own term remains the
usual central attraction.

## Controlled comparison

The Chapter 1/2 comparison keeps the same kernels, constants, initial states,
frame, observer, RK4 integrator, 3,600-second timestep, telemetry, and
visualization. Only the force formulation changes.

For the original 400-day merged-SPK comparison:

| Probe | Chapter 1 position error | Chapter 2 position error | Chapter 1 velocity error | Chapter 2 velocity error |
| --- | ---: | ---: | ---: | ---: |
| Voyager 1 | 122,783.010 km | 32,101.379 km | 9.08717 m/s | 1.94987 m/s |
| Voyager 2 | 121,720.889 km | 34,124.210 km | 8.05398 m/s | 1.75436 m/s |

That corresponds to roughly 72–74% lower position disagreement and 78% lower
velocity disagreement. These values measure agreement with the merged
supertrajectory, not navigation-grade accuracy.

A timestep check at 3,600, 1,800, and 900 seconds changed the 400-day endpoint
by only a few kilometers, far less than the roughly 88,000–91,000 km removed by
the heliocentric correction.

## Reconstructed-arc validation

The separate headless validation initializes and scores against four NAIF
reconstructed Jupiter/Saturn encounter SPKs. The two primary pre-maneuver
Saturn checkpoints improve from:

- Voyager 1: 0.729 km to 0.018 km (97.49% reduction)
- Voyager 2: 0.850 km to 0.266 km (68.71% reduction)

Jupiter and later encounter checkpoints are kept as diagnostic or excluded
where maneuvers and close-encounter physics make attribution ambiguous. The
full scoring rules, kernel precedence, hashes, and residual tables are in the
[truth-reference contract](truth_reference.md).

## Limits

Chapter 2 does **not** reproduce the complete Voyager gravity-assist trajectory.
It still omits effects such as trajectory-correction maneuvers, major moons,
giant-planet harmonics, solar-radiation pressure, and encounter-specific force
modeling.

The result is intentionally narrow: correcting the accelerating heliocentric
origin materially improves the controlled comparisons, but it does not turn the
example into a high-fidelity mission reconstruction.

[Back to Chapter 1](chapter_1.md) · [Back to Voyager setup](../README.md)
