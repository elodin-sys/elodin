# Voyager validation reference

This document describes the separate headless benchmark for Chapters 1 and 2.
It is not used by the interactive tutorial.

## Two reference data sets

The interactive tutorial uses long-span merged Voyager SPKs for continuous
mission-scale visualization. The benchmark uses four reconstructed encounter
SPKs from the NAIF Voyager archive:

| Case | Kernel | Start | Encounter |
| --- | --- | --- | --- |
| Voyager 1 Jupiter | `vgr1_jup230.bsp` | 1979-02-01 | 1979-03-05 |
| Voyager 2 Jupiter | `vgr2_jup230.bsp` | 1979-06-05 | 1979-07-09 |
| Voyager 1 Saturn | `vgr1_sat337.bsp` | 1980-10-07 | 1980-11-12 |
| Voyager 2 Saturn | `vgr2_sat337.bsp` | 1981-07-15 | 1981-08-26 |

Each case initializes and scores against the same reconstructed spacecraft
segment. `de440.bsp` supplies the planetary states. The manifest records the
kernel hashes, coverage, frames, targets, and checkpoints.

In practice, each validation run does this:

1. Verify the declared kernel files and SHA-256 hashes.
2. Load the merged kernels, the selected encounter kernel, and `de440.bsp`.
3. Initialize the simulated probe from the encounter SPK at the case start.
4. Propagate the probe with Chapter 1 or Chapter 2 and compare it with that
   same SPK at the declared checkpoints.

The SPICE trajectory is only the reference for initialization and scoring. It
does not drive the simulated probe after initialization. Planet positions used
by the force model come from `de440.bsp`.

This is a data-selection and scoring contract, not reverse engineering of
Voyager maneuvers. Checkpoints near maneuvers or close approach are diagnostic
or excluded because the current model does not include those effects.

## Results

At the selected pre-encounter checkpoints, Chapter 2 reduces position error at
all eight points:

| Case | Day | Chapter 1 | Chapter 2 | Reduction |
| --- | ---: | ---: | ---: | ---: |
| V1 Jupiter | 20 | 338.580 km | 33.699 km | 90.05% |
| V1 Jupiter | 30 | 675.890 km | 555.321 km | 17.84% |
| V2 Jupiter | 20 | 227.324 km | 99.366 km | 56.29% |
| V2 Jupiter | 30 | 632.275 km | 575.536 km | 8.97% |
| V1 Saturn | 1 | 0.729 km | 0.018 km | 97.49% |
| V1 Saturn | 3 | 6.509 km | 0.179 km | 97.26% |
| V2 Saturn | 1 | 0.850 km | 0.266 km | 68.71% |
| V2 Saturn | 4 | 13.550 km | 4.257 km | 68.59% |

The Jupiter day-30 velocity comparison is worse in Chapter 2. That result is
kept in the artifact rather than hidden; close-encounter physics and omitted
mission events are becoming important.

The chapters still omit trajectory-correction maneuvers, major moons, planetary
gravity harmonics, planet-center motion within barycentric systems, and
ephemeris updates at RK4 intermediate stages. One-hour steps are also too large
for close flybys. The benchmark therefore measures a controlled model change,
not full flyby reconstruction or absolute navigation accuracy.

## Reproduce it

From the repository root and Nix development shell:

```bash
cd examples/voyager
./download_spice_data.sh
cd ../..
uv run python examples/voyager/validate_truth.py --include-convergence
```

For a focused run:

```bash
uv run python examples/voyager/validate_truth.py \
  --case v1_jupiter --chapter 1 --chapter 2
```

The harness verifies kernel hashes and SPICE segment precedence before writing
`examples/voyager/truth_validation_results.json`.

[NAIF Voyager kernel notes](https://naif.jpl.nasa.gov/pub/naif/VOYAGER/kernels/aareadme.txt)
