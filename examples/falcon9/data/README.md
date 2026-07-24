# Falcon 9 Truth Data

Vendored reference data for the Falcon 9 launch-to-landing SITL example. This
is the distilled result of a survey across the public Falcon 9 data landscape
(webcast-telemetry archives, OCR extraction tools, NASA supersonic-
retropropulsion papers, official mission documents, orbital catalogs,
atmosphere archives, and community simulations). The short version: **no
public archive of raw Falcon 9 onboard engineering telemetry exists.** The
best available truth layer combines webcast-derived stage-1 telemetry,
official event timelines, and published vehicle configuration figures — that
is what lives here.

## Reference Missions

Two Falcon 9 RTLS flights, LC-39A to Landing Zone 1, chosen because they are
the only mission class surveyed with **complete public stage-1 telemetry from
liftoff through touchdown** *and* a public payload mass (needed for a
credible mass model):

| | CRS-12 (primary) | CRS-11 (secondary) |
|---|---|---|
| Launch (UTC) | 2017-08-14 16:31:37 | 2017-06-03 21:07:38 |
| Booster | B1039 (Block 4, flight 1) | B1035 (Full Thrust, flight 1) |
| Payload | Dragon C113 + 2,910 kg cargo | Dragon C106 + 2,708 kg cargo |
| Stage-1 telemetry | t = 0–480 s, through touchdown | t = 0–465 s, through touchdown |
| Landing | LZ-1, success | LZ-1, success |

CRS-12 is the calibration target (it also carries partial stage-2 telemetry
and a press-kit timeline). CRS-11 — same pads, same profile, near-identical
payload class — is the held-out validation flight: a calibrated model should
reproduce it without retuning.

## Files

Per mission (`crs12/`, `crs11/`):

| File | Contents |
|---|---|
| `stage1_raw.json` | Stage-1 webcast telemetry at ~30 Hz: `time` (s, from liftoff), `velocity` (m/s), `altitude` (km) |
| `stage2_raw.json` | (CRS-12 only) partial stage-2 telemetry, t = 502–559 s |
| `analysed.json` | Upstream 1 Hz derived channels: vertical/horizontal velocity, acceleration, downrange, velocity angle, dynamic pressure |
| `events.json` | Event times (s, rounded to 1 s) observed in the webcast: Max-Q, throttle bucket, MECO, boostback start/end, apogee, entry start/end, landing start/end |
| `stages.json` | Time offset where each stage's webcast telemetry begins |
| `mission.json` | Our compilation of mission facts (launch time, pads, coordinates, booster, payload masses) with per-field source URLs |
| `presskit_timeline.json` | (CRS-12 only) planned event times from the official press kit, for plan-vs-actual comparison |

`checksums.txt` holds SHA-256 hashes of every vendored JSON.

## Provenance

Telemetry files (`stage1_raw`, `stage2_raw`, `analysed`, `events`, `stages`)
are copied unmodified (filenames normalized) from
[shahar603/Telemetry-Data](https://github.com/shahar603/Telemetry-Data),
commit `b245d3b81aa36b7941ec10f3f4b508999d106a6d` (2020-01-24), directories
`SpaceX CRS-12/JSON/` and `SpaceX CRS-11/JSON/`. The dataset was produced
with [SpaceXtract](https://github.com/shahar603/SpaceXtract) (OCR of official
SpaceX webcast overlays) and is released under the Unlicense (public domain)
— copy vendored as [`LICENSE-telemetry-data`](LICENSE-telemetry-data).

`mission.json` and `presskit_timeline.json` are our own small compilations of
public facts; each carries its source URLs inline (NASA mission overviews,
the CRS-12 press kit, NASASpaceflight coverage, Wikipedia mission pages).

## Data Quality Notes

Measured on the vendored files:

- **This is displayed telemetry, not onboard telemetry.** Values were OCR'd
  from the webcast overlay. The display's internal filtering and latency are
  undocumented; treat both as uncertainty terms (latency appears to be
  ~1–2 s judging by observed-vs-press-kit event times).
- **Quantization:** velocity steps are 1 km/h (0.276 m/s) — the overlay
  displays integer km/h, converted upstream to m/s. Altitude steps are
  0.1 km. Below ~1 km altitude the display resolution dominates: model it,
  don't fight it.
- **Sampling:** nominally 30 Hz video frames; values hold across frames
  between display updates. CRS-12 has one 1.5 s gap (t = 287.9 s), CRS-11 one
  1.3 s gap (t = 70.9 s). No non-monotonic time and no >50 m/s OCR spikes in
  either stage-1 file.
- **Events are ±1 s or worse** (rounded, display-latency included). The
  CRS-12 press-kit timeline is *planned*, not flown — expect small offsets.
- **Recompute all derivatives.** The upstream `analysed.json` channels
  (vertical/horizontal split, downrange, dynamic pressure) embed undocumented
  smoothing and atmosphere assumptions. Use them as sanity references only;
  `reference.py` should derive its own profiles from `stage1_raw.json` and
  state its assumptions. (Example: upstream downrange at CRS-12 touchdown is
  −6.4 km, whereas LC-39A → LZ-1 geodesic distance is ~14.5 km.)
- The repository's per-mission "Payload Mass" README figures disagree with
  NASA's cargo manifests (e.g. 3,310 kg vs. 2,910 kg for CRS-12);
  `mission.json` carries the NASA press-kit numbers.

## Vehicle Configuration Sources (cited, not vendored)

Public configuration references for the vehicle model constants in `sim.py`:

- [Falcon Payload User's Guide](https://www.spacex.com/assets/media/falcon-users-guide-2025-05-09.pdf)
  (SpaceX, May 2025) — stage architecture, 9 Merlin 1D + TVC, grid fins,
  landing legs, restart capability. Copyrighted; cite, don't redistribute.
- [SpaceX Falcon 9 vehicle page](https://www.spacex.com/vehicles/falcon-9/) —
  headline thrust/mass figures. **Block 5 numbers** — the 2017 reference
  flights flew Block 3/4 boosters with lower per-engine thrust (~760 kN SL
  vs. 845 kN) and less propellant (~395–400 t vs. 411 t); see the
  [Falcon 9 Full Thrust Wikipedia article](https://en.wikipedia.org/wiki/Falcon_9_Full_Thrust)
  for the block-by-block breakdown.
- Community-consolidated estimates (stage dry mass, propellant load, Isp,
  throttle range) are calibration parameters with priors, not facts; keep
  them flagged in `spec.toml`.

## Entry-Burn Validation Priors (cited, not vendored)

The NASA–SpaceX supersonic-retropropulsion research is the only public
material based on **actual Falcon 9 onboard data** (F9-10/Orbcomm-OG2 2014,
F9-13/CRS-4 2014). Useful as qualitative bounds for the entry-burn phase:

- Entry burns lasted ~20–40 s, flown on engines 1, 5, and 9.
- Chamber pressure reached steady state ~5 s after ignition command.
- Attitude/rate transients at retropropulsion startup were observable but
  bounded.

Sources: [NASA SRP program overview (NTRS 20170008535)](https://ntrs.nasa.gov/citations/20170008535),
[Sforzo & Braun, "Feasibility of Supersonic Retropropulsion..." (Georgia Tech)](https://repository.gatech.edu/bitstreams/9205d1bc-2225-4249-9f61-4e768f55d89c/download).
Many published values are normalized/redacted — do not read absolute
engineering units off those plots.

## Sources Considered and Not Selected

- **NROL-76, X-37B OTV-5, ZUMA** (same repo, full stage-1 RTLS coverage) —
  classified payload masses break the mass model. ZUMA's near-identical
  profile makes it a possible extra validation flight if ever needed.
- **SES-11, KoreaSat 5A** (droneship missions) — stage-1 telemetry cuts off
  at 20–24 km altitude, well before landing.
- **Launch Dashboard API** — same data as Telemetry-Data behind a REST
  schema; hosted endpoint unverified. Vendoring pinned files is simpler.
- **SpaceXtract / other OCR tools** — extraction tooling, not data. Revisit
  only if we need to re-extract a mission from source video.
- **hamsternz/falcon9_pipeline** (RF/baseband decode) — unverified
  provenance, unresolved legality; excluded.
- **FlightClub, mpopt, community simulators** — simulation priors, not
  truth; consult for guidance ideas only.
- **Space-Track / CelesTrak orbital elements** — validates stage-2 insertion,
  out of scope for the booster-focused demo.
- **NOAA IGRA / ERA5 atmosphere** — deferred; start with a standard
  atmosphere as an explicit calibration input, add the 2017-08-14 Cape
  Canaveral sounding if density error dominates calibration residuals.

## Re-fetching Upstream

```sh
git clone https://github.com/shahar603/Telemetry-Data /tmp/telemetry-data
git -C /tmp/telemetry-data checkout b245d3b81aa36b7941ec10f3f4b508999d106a6d
# compare against checksums.txt (filenames here drop the "raw"/space naming)
shasum -a 256 "/tmp/telemetry-data/SpaceX CRS-12/JSON/stage1 raw.json"
```
