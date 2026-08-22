# Voyager 1 Jupiter validation — one-page report

This is the focused reconstructed-trajectory check for PR #801: initialize Voyager 1 from the Jupiter encounter SPK, propagate Chapter 1 and Chapter 2 from the **same initial state**, and compare both against the **same reconstructed SPICE trajectory**.

## 1. Hypothesis

The Voyager example integrates a Sun-relative state. Chapter 1 applies each planet's direct pull on the spacecraft, but does not subtract that planet's acceleration of the Sun. The hypothesis is that this missing heliocentric correction is a meaningful source of propagation error.

![Chapter 1 versus Chapter 2 gravity](assets/gravity_before_after.svg)

For Chapter 2, each planet contributes

```text
mu_i * ((r_i - r) / |r_i - r|^3 - r_i / |r_i|^3)
```

The second term is the indirect heliocentric correction: it removes the acceleration of the Sun-centered origin caused by that planet.

## 2. Test

| Item | Value |
| --- | --- |
| Spacecraft | Voyager 1 |
| Encounter kernel | `vgr1_jup230.bsp` |
| Start | 1979-02-06 00:00 UTC |
| Checkpoints | 0, 1, 2 days |
| Frame / origin | `ECLIPJ2000` / Sun |
| Integrator | RK4 |
| Step | 1 hour |
| Planet ephemerides | DE440 |

Both chapters use the same start state, masses, ephemerides, timestep, integrator, and SPICE reference. The validation window is the Feb 6–8 interval described in PR #801; the Feb 5 / Feb 9 maneuver-boundary rationale is therefore treated as PR context rather than independently established maneuver history here.

## 3. Result

A fresh local rerun reproduced the improvement reported in the PR:

| Checkpoint | Chapter 1 position error | Chapter 2 position error | Reduction | Chapter 1 velocity error | Chapter 2 velocity error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Day 1 | 0.766 km | 0.166 km | **78.29%** | 0.0177 m/s | 0.0039 m/s |
| Day 2 | 3.060 km | 0.668 km | **78.16%** | 0.0354 m/s | 0.0078 m/s |

**Takeaway:** in this specific validation run, Chapter 2 reduces both the position and velocity residuals by about 78% at the two scored checkpoints.

## 4. What this does — and does not — establish

The result supports the hypothesis that the missing heliocentric indirect term was a real dynamics error in the Chapter 1 model. It does **not** mean the remaining residual is bad SPICE data or that the Voyager trajectory is fully reconstructed. The current propagation is still gravity-only, so historical thrust and other unmodeled effects can remain in the residual.

The useful pattern is the workflow itself: **identify a model gap → form a hypothesis → change one piece of the dynamics → test both models against the same reconstructed reference → keep the improvement only if the residual actually gets better.**

## Sources

- PR #801
- Issue #794
- NAIF Voyager 1 Jupiter encounter SPK: `vgr1_jup230.bsp`
- NASA/JPL Voyager encounter documentation linked from #794
