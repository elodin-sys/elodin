# Chapter 2 — Heliocentric relative dynamics

Chapter 2 keeps the Chapter 1 simulation intact and adds one astrodynamics
concept: the acceleration of the Sun-centered origin.

The goal is not to make Voyager "fully accurate." It is to isolate one modeling
assumption, derive the correction, and measure its effect.

## Run it

From the repository root:

```bash
python examples/voyager/chapter_2.py run
```

For comparison, Chapter 1 remains:

```bash
python examples/voyager/main.py run
```

## The subtlety

SPICE supplies the Voyager and planetary states relative to the Sun in
`ECLIPJ2000`.

`ECLIPJ2000` provides nonrotating axes, but that does not make the Sun-centered
origin inertial. The planets gravitationally accelerate the Sun.

Define the probe's heliocentric position as

```text
r = R_probe - R_sun
```

Differentiate twice:

```text
r'' = R_probe'' - R_sun''
```

The acceleration of the probe relative to the Sun therefore needs both the
probe's planetary acceleration and the corresponding acceleration of the Sun.

For a planet at heliocentric position `r_i`, the Chapter 2 contribution is

```text
mu_i * (
    (r_i - r) / |r_i - r|^3
    - r_i / |r_i|^3
)
```

The first term is the planet's direct acceleration of Voyager.

The second term is that planet's acceleration of the Sun. It is subtracted
because the propagated coordinate is `R_probe - R_sun`.

For the Sun's own gravity edge, the source position is the origin, so the
indirect contribution is zero and the usual central attraction remains.

## Original long-span controlled experiment

Chapter 2 changes only this force formulation.

The comparison keeps the same:

- SPICE kernels;
- gravitational constants;
- initial states;
- `ECLIPJ2000` frame;
- `SUN` observer;
- classical RK4 integrator;
- 3,600-second timestep;
- telemetry and visualization.

The original 400-day runs used native Elodin/Cranelift and the long-span merged
Voyager SPKs.

| Probe | Chapter 1 position error | Chapter 2 position error | Chapter 1 velocity error | Chapter 2 velocity error |
| --- | ---: | ---: | ---: | ---: |
| Voyager 1 | 122,783.010 km | 32,101.379 km | 9.08717 m/s | 1.94987 m/s |
| Voyager 2 | 121,720.889 km | 34,124.210 km | 8.05398 m/s | 1.75436 m/s |

That is:

- 73.9% lower Voyager 1 position disagreement;
- 72.0% lower Voyager 2 position disagreement;
- 78.5% lower Voyager 1 velocity disagreement;
- 78.2% lower Voyager 2 velocity disagreement.

These numbers measure agreement with the published merged supertrajectory.
They should not be read as absolute navigation accuracy: the separate
current-best encounter solutions differ materially over overlapping early
mission coverage.

The improvement grows smoothly over the trajectory rather than appearing only
at the final endpoint.

## Why this is not just a timestep effect

The Chapter 2 400-day endpoint was also run at 3,600, 1,800, and 900 seconds.

Across that eightfold increase in step count, the position spread was only
about:

- 5.65 km for Voyager 1;
- 6.33 km for Voyager 2.

That is tiny compared with the roughly 88,000–91,000 km of disagreement removed
by the heliocentric correction.

The result therefore points to a model-form limitation during cruise rather
than ordinary RK4 timestep error being the dominant cause.

## Validation

The focused Chapter 2 tests cover:

- direct-force direction and magnitude;
- the sign of the indirect acceleration;
- accumulation across multiple gravity sources;
- the Sun-at-origin case without division by zero;
- isolation of the propagated dynamics from post-initialization Voyager truth.

The native Chapter 1 runs reproduce the original reference checkpoints, while
Chapter 2 improves both position and velocity disagreement for both probes at
4, 100, and 400 days.

The later truth-reference audit re-ran both chapters against four independent
NAIF reconstructed encounter solutions. Each case initializes from the same
SPK segment that it scores, with explicit kernel hashes and runtime segment
audits. The two primary pre-maneuver Saturn scores improve by 68.71% and
97.49%. Diagnostic Jupiter/Saturn approach points still trend better, from
8.97% to 90.05%, but they are not headline claims. Near Jupiter encounter, the
simple model no longer improves every velocity metric.

[Read the complete truth contract, results, and limitations](truth_reference.md).

Equivalent alternating native runs measured roughly 3% runtime overhead for
Chapter 2. Treat that as an indicative microbenchmark rather than a universal
performance guarantee.

## What Chapter 2 does not claim

Tens of thousands of kilometers of disagreement still remain after 400 days.

Chapter 2 does not add:

- DE440-consistent gravitational parameters;
- new SPICE kernels;
- moons;
- maneuvers;
- solar-radiation pressure;
- thermal recoil;
- alternate or adaptive integration;
- encounter-specific modeling.

Those are separate questions for later chapters so this chapter keeps one
lesson and one controlled change.

## Takeaway

A coordinate system can have nonrotating axes while its origin still
accelerates.

When the state is measured relative to that accelerating origin, the origin's
acceleration must appear in the relative-motion equation.

[Back to Chapter 1](chapter_1.md) · [Back to Voyager setup](../README.md)
