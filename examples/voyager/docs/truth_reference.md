# Voyager truth-reference contract

This document defines which Voyager trajectory data this example uses, what
each data set can support, and how Chapters 1 and 2 are validated without
mixing disagreement between trajectory solutions into force-model error.

The machine-readable contract is
[`truth_reference.json`](../truth_reference.json). The checked-in output from
the reference run is
[`truth_validation_results.json`](../truth_validation_results.json).

## Why there are two kinds of reference

The long-span merged Voyager SPKs are useful for the tutorial's mission-scale
visualization. They provide continuous spacecraft coverage from launch through
the outer mission and let a learner see model divergence develop over hundreds
of days.

They are not the best available solution for every early encounter interval.
NAIF identifies separate Jupiter and Saturn SPKs as the current-best encounter
solutions. Those solutions differ materially from the overlapping trajectory
in the merged kernels.

The example therefore keeps the two uses separate:

- **Long-span educational reference:** the merged SPKs remain the default for
  the interactive Chapters 1 and 2 journey.
- **Reconstructed-arc validation:** four short encounter SPKs initialize and
  score the headless validation cases.

This distinction prevents a benchmark from initializing on one trajectory
solution and later measuring the difference to another as if it were entirely
simulation error.

NAIF also cautions that the primary Voyager mission predates SPICE and that the
restored ancillary data should be used carefully. “Current-best” here means the
best solution identified by the official Voyager kernel archive; it does not
remove that archive-level caveat.

Authoritative archive descriptions:

- [NAIF Voyager kernel status and solution notes](https://naif.jpl.nasa.gov/pub/naif/VOYAGER/kernels/aareadme.txt)
- [NAIF Voyager SPK coverage summaries](https://naif.jpl.nasa.gov/pub/naif/VOYAGER/kernels/spk/aa_summaries.txt)

## Reconstructed validation cases

Every case uses geometric states (`NONE`) expressed relative to the Sun in
`ECLIPJ2000`. The spacecraft is initialized from the same reconstructed SPK
segment used at every scoring checkpoint. DE440 supplies the planetary states
used to translate the segment's native center and update the model's gravity
sources.

| Case | Encounter kernel and segment | Native center | Initialization | Closest approach | Scoring end |
| --- | --- | ---: | --- | --- | --- |
| Voyager 1 Jupiter | `vgr1_jup230.bsp` / `vgr1.jup230.nio` | 5, Jupiter barycenter | 1979-02-01 | 1979-03-05 12:04:46.794 | 1979-03-18 |
| Voyager 2 Jupiter | `vgr2_jup230.bsp` / `vgr2.jup230.nio` | 5, Jupiter barycenter | 1979-06-05 | 1979-07-09 22:28:52.806 | 1979-07-20 |
| Voyager 1 Saturn | `vgr1_sat337.bsp` / `vgr1.sat337.nio` | 6, Saturn barycenter | 1980-10-07 | 1980-11-12 23:45:27.463 | 1980-11-18 |
| Voyager 2 Saturn | `vgr2_sat337.bsp` / `vgr2.sat337.nio` | 6, Saturn barycenter | 1981-07-15 | 1981-08-26 03:23:50.181 | 1981-08-29 |

Voyager 1 Saturn intentionally begins after the reconstructed SPK changes from
a Sun-centered segment to a Saturn-barycenter segment. The complete coverage,
URLs, SHA-256 values, target IDs, native frames, segment types, and declared
checkpoints are recorded in the manifest. The closest-approach epochs were
derived by minimizing the spacecraft-to-native-center distance within each
selected reconstructed segment.

## Maneuver-aware checkpoint contract

The validation harness labels every checkpoint as `primary`, `diagnostic`, or
`excluded`. This is executable: the role, maneuver status, and reason are copied
into every metric and into the result summaries.

| Role | Use |
| --- | --- |
| **Anchor** | Initialization audit only. Error is identically zero by construction and is not a Chapter 1/2 accuracy score. |
| **Primary** | Headline Chapter 1/2 score for a documented pre-maneuver approach point. |
| **Diagnostic** | Trend evidence that overlaps a correction boundary or lacks a replayable inertial maneuver vector. |
| **Excluded** | Retained for visibility, but not used for accuracy claims because close-encounter or post-maneuver dynamics are not a controlled replay. |

The contract is conservative. `start` is an initialization **anchor**, not a
score. The only primary approach scores in this baseline are Voyager 1 Saturn
on Oct 8 (pre-A-8) and Voyager 2 Saturn on Jul 16 (pre-B-8). Jupiter
`clean`/`mid`/`late` points remain diagnostics: the Voyager 1 reconstructed
arc begins after the Jan 29 correction and its day-20 point is after the
Feb 20 window, while Voyager 2 starts inside the TCM-3 phase. Saturn
`clean_approach` is the A-8/B-8 boundary and is diagnostic, not a clean
pre-maneuver score. Closest-approach and post-encounter points are excluded
from headline claims for all four arcs.

A later campaign can add a Voyager 1 Jupiter checkpoint strictly before Feb 20
without changing Chapters 1 or 2. That is a contract extension, not a
Chapter 3 force-model change.

The audit uses the NASA/JPL operations and cruise reports cited in the manifest:
Voyager 1 Jupiter correction activity around Jan 29/Feb 20, Voyager 2 Jupiter
TCM-3/4/5 phases, Voyager 1 Saturn A-8/A-9, and Voyager 2 Saturn B-7/B-8/B-9.
Those reports provide dates and phase windows, and sometimes burn magnitudes and
attitude descriptions, but not a complete inertial delta-v vector for every
event. Until that data exists, post-maneuver error cannot be attributed to the
force model alone.

## Kernel precedence

SPICE gives a competing segment from the file loaded later higher priority.
The validation load order is therefore part of the contract:

1. `naif0012.tls`
2. `Voyager_1.a54206u_V0.2_merged.bsp`
3. `Voyager_2.m05016u.merged.bsp`
4. the selected reconstructed encounter SPK
5. `de440.bsp`

The encounter file wins for the selected spacecraft target. DE440 wins for the
planetary targets that are also embedded in the merged kernels. The harness
uses SPICE's segment-selection API to verify the winning spacecraft file,
segment, target, center, native frame, and type at both initialization and the
end of the scoring interval. It also records the winning planetary segment for
every declared planet at those same epochs and fails if that file is not
`de440.bsp`. A run fails instead of producing metrics if either selection
differs from the manifest.

See [NAIF's SPK data-precedence documentation](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html#Data%20Precedence).

## Revalidated Chapters 1 and 2

The baseline uses native Elodin `six_dof` RK4 at a 100-second timestep. The
chapters keep their existing force models and rounded body masses. The truth
source, initialization epoch, output frame, observer, integrator, timestep, and
planet states are identical within every Chapter 1/Chapter 2 pair. Every run
records position and velocity residuals both as `ECLIPJ2000` Cartesian vectors
and in the truth trajectory's radial-transverse-normal basis.

Primary pre-maneuver approach scores:

| Case | Day | Chapter 1 position | Chapter 2 position | Reduction | Chapter 1 velocity | Chapter 2 velocity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 Saturn | 1 | 0.729 km | 0.018 km | **97.49%** | 0.0170 m/s | 0.0004 m/s |
| V2 Saturn | 1 | 0.850 km | 0.266 km | **68.71%** | 0.0197 m/s | 0.0062 m/s |

Diagnostic approach scores, retained for trend inspection:

| Case | Day | Chapter 1 position | Chapter 2 position | Reduction | Chapter 1 velocity | Chapter 2 velocity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 Jupiter | 20 | 338.580 km | 33.699 km | **90.05%** | 0.3619 m/s | 0.0746 m/s |
| V1 Jupiter | 30 | 675.890 km | 555.321 km | **17.84%** | 0.4139 m/s | 1.0137 m/s |
| V2 Jupiter | 20 | 227.324 km | 99.366 km | **56.29%** | 0.2524 m/s | 0.1422 m/s |
| V2 Jupiter | 30 | 632.275 km | 575.536 km | **8.97%** | 0.7133 m/s | 0.9028 m/s |
| V1 Saturn | 3 | 6.509 km | 0.179 km | **97.26%** | 0.0498 m/s | 0.0016 m/s |
| V2 Saturn | 4 | 13.550 km | 4.257 km | **68.59%** | 0.0782 m/s | 0.0246 m/s |

Chapter 2 reduces position disagreement at the two controlled primary approach
checkpoints and at the four legacy `clean_approach` diagnostics. The contract
keeps those classes separate and does not hide the diagnostic mid-arc
regressions. At the earlier checkpoints it also reduces velocity disagreement.
The two Jupiter day-30 velocity values are not improvements, even though their
position values remain smaller. They are reported rather than hidden: the
simple point-mass model is approaching a close-encounter regime where omitted
physics and trajectory events become more important.

The defensible Chapter 2 conclusion is consequently narrow and strong:

> The heliocentric indirect-acceleration correction materially improves the
> controlled pre-maneuver Saturn approach scores and the diagnostic
> clean-approach trend across all four reconstructed arcs. It is not a claim
> that the educational gravity model reconstructs each flyby or every velocity
> component throughout an encounter arc.

The labeled mid-arc, closest-approach, and end checkpoints deliberately remain
in the result artifact even when Chapter 2 is worse. They expose where omitted
maneuvers and close-encounter physics dominate instead of allowing a favorable
approach-only score to stand in for flyby reconstruction:

| Case | Closest-approach change | End-of-arc change |
| --- | ---: | ---: |
| V1 Jupiter | 88.85% worse | 21.19% worse |
| V2 Jupiter | 74.08% worse | 4.02% worse |
| V1 Saturn | 5.20% worse | 4.36% better |
| V2 Saturn | 48.31% better | 0.16% worse |

## Timestep control and the remaining floor

The investigation also ran the Jupiter cases at 3,600, 900, 300, and 100
seconds and the Saturn cases at 3,600, 300, and 100 seconds.

Saturn timestep controls use the **primary** `early_approach` points (V1 Oct 8
pre-A-8, V2 Jul 16 pre-B-8). Jupiter has no primary approach window, so its
controls use the **diagnostic** `clean_approach` points (day 20). Those Jupiter
points are not called clean force-model scores; they sit near documented TCM
windows. Saturn `clean_approach` (day 3 / day 4) is the A-8 / B-8 boundary and
is diagnostic only — it is not used for convergence headlines.

| Case | Checkpoint | Role | Position-error spread |
| --- | ---: | --- | ---: |
| V1 Saturn | day 1 `early_approach` | primary | 3.6e-8 km |
| V2 Saturn | day 1 `early_approach` | primary | 9.2e-8 km |
| V1 Jupiter | day 20 `clean_approach` | diagnostic | 0.0126 km |
| V2 Jupiter | day 20 `clean_approach` | diagnostic | 0.0254 km |

Those spreads are far smaller than the Chapter 1-to-Chapter 2 changes, so the
primary Saturn improvement is a force-model effect rather than ordinary RK4
truncation error. The diagnostic Jupiter day-20 spreads support the same
numerical conclusion without promoting those points to headline scores.

One-hour integration is inadequate through the close flybys. Smaller steps
reduce that numerical error, but a substantial residual remains because the
current chapters deliberately omit:

- documented trajectory-correction maneuvers represented by the reconstructed
  trajectory;
- major satellite gravity, including Titan and the Galilean moons;
- planet-center motion versus a single planet-system barycenter point mass;
- giant-planet gravity harmonics;
- planetary ephemeris evaluation at the RK intermediate stages.

These effects belong to later controlled experiments. They are not reasons to
alter Chapter 2's reference-frame correction.

## Cruise-reference limitation

The January 1978 to February 1979 investigation found useful navigation
reports, maneuver histories, and the merged spacecraft trajectory, but no
separate public tracking-fit state-vector SPK covering that complete cruise
interval in the NAIF Voyager archive or NASA technical material surveyed.

The merged SPKs can therefore support a claim of **agreement with the published
mission-design/supertrajectory reference** during that interval. This work does
not label the same comparison **absolute navigation accuracy**.

## Tutorial database epoch

The interactive example's SPICE start string is `1978-01-01T00:00:00`.
Propagation has always used `spice.utc2et` of that string.

The Elodin-DB `start_timestamp` was previously hardcoded to
`252_452_400_000_000` (1977-12-31 21:40:00 UTC), 8400 seconds earlier than
UTC midnight on 1978-01-01. That offset did not change the force model or
SPICE queries. It only mislabeled sample wall-clock times in the tutorial
database.

`utc_epoch_microseconds` now converts the same UTC string the SPICE query
uses. The tutorial DB epoch is `252_460_800_000_000`. Validation cases use
each reconstructed-arc `initialization_utc` the same way. Existing tutorial
databases recorded under the old constant are 8400 seconds behind UTC; create
a new `DB_PATH` if you need the labels to match.

That is a data limitation, not a force-model diagnosis. If a navigation-grade
cruise reconstruction becomes available, it can be added as another manifest
case without changing the chapter dynamics.

Relevant NASA material includes the
[Voyager navigation strategy record](https://ntrs.nasa.gov/citations/19780047969)
and the
[1978–1979 cruise-phase maneuver summary](https://ntrs.nasa.gov/api/citations/19790009610/downloads/19790009610.pdf).

## Reproduce the baseline

Download and verify all declared kernels:

```bash
cd examples/voyager
./download_spice_data.sh
cd ../..
```

From the repository's Nix development shell, run:

```bash
uv run python examples/voyager/validate_truth.py --include-convergence
```

The command verifies every required SHA-256 before propagation, runs both
chapters across all four cases, adds the declared Chapter 2 timestep-control
matrix, audits spacecraft and DE440 planetary segment precedence, and writes
`examples/voyager/truth_validation_results.json`. The checked-in artifact
contains 18 runs: eight 100-second Chapter 1/2 baselines and ten additional
Chapter 2 convergence runs.

This refresh did not rerun those 18 native propagations. Residuals are the
previous campaign. The contract labels, Saturn `early_approach` timestep
controls, and DE440 planetary segment audits were rebuilt from that artifact
plus a live SPICE load-order check.

For a focused reproduction:

```bash
uv run python examples/voyager/validate_truth.py \
    --case v1_jupiter \
    --chapter 1 \
    --chapter 2
```

No editor, GUI, or Elodin DB query tooling is required to obtain the numerical
measurements.

The harness scores only the existing Chapter 1 and Chapter 2 force models. It
does not accept gravity-parameter, RK-stage ephemeris, or moon-system flags.
Those belong to a later Chapter 3 experiment after this contract is the
scoring gate.
