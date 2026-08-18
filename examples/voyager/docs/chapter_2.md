# Chapter 2 — Heliocentric relative dynamics

Chapter 2 keeps Chapter 1's model and adds one correction: the Sun-centered
origin is accelerating.

Run it with:

```bash
python examples/voyager/chapter_2.py run
```

SPICE gives states relative to the Sun in `ECLIPJ2000`. The axes are
nonrotating, but the origin is not inertial because the planets accelerate the
Sun. For `r = R_probe - R_sun`,

```text
r'' = R_probe'' - R_sun''
```

so each planet contributes

```text
mu_i * ((r_i - r) / |r_i - r|^3 - r_i / |r_i|^3)
```

The first term is the planet's pull on Voyager; the second removes the same
planet's acceleration of the Sun.

## Long-span comparison

Using the same kernels, constants, initial states, RK4 integrator, and
3,600-second timestep as Chapter 1:

| Probe | Chapter 1 position | Chapter 2 position | Chapter 1 velocity | Chapter 2 velocity |
| --- | ---: | ---: | ---: | ---: |
| Voyager 1 | 122,783.010 km | 32,101.379 km | 9.08717 m/s | 1.94987 m/s |
| Voyager 2 | 121,720.889 km | 34,124.210 km | 8.05398 m/s | 1.75436 m/s |

This is roughly 72–74% lower position disagreement and 78% lower velocity
disagreement over 400 days. A timestep check changed the endpoint by only a
few kilometers, much less than the 88,000–91,000 km improvement.

## Reconstructed-arc validation

The separate headless harness initializes and scores four NAIF Jupiter/Saturn
encounter arcs. At the clean pre-maneuver Saturn checkpoints:

- Voyager 1: 0.729 km → 0.018 km (97.49% reduction)
- Voyager 2: 0.850 km → 0.266 km (68.71% reduction)

These results are narrow: Chapter 2 improves the pre-encounter comparisons,
but does not reproduce the complete flybys or gravity-assist sequence. The
harness does not reverse-engineer maneuvers; checkpoints near maneuvers and
close approach are marked diagnostic or excluded because those effects are not
modeled.

[Truth-reference contract](truth_reference.md)
