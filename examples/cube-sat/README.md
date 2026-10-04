# CubeSat Attitude Control

Simulates a low-Earth-orbit CubeSat.

## Run in the editor

```
elodin editor main.py
```

## Visuals

The schematic enables the editor's built-in cinematic Earth
(`environment { earth }` + `viewport cinematic=#true`): a true-scale WGS84
globe with atmosphere, star fields, night city lights, and airglow, all
driven by that camera. The
sun is the real ephemeris for the sim clock, pinned to
`2026-03-20T10:21:00Z` (March equinox). The sat starts just west of Southern
California on the night side, about eight minutes before orbital sunrise, so
the 20-minute window covers west-coast city lights through the terminator
into dayside (about 20 wall seconds at the default 60x playback). Scrubbing
the timeline moves the sun with the playhead.

Parked night/day/night screenshots (same camera offset, sun moves):

```sh
for s in night dawn-limb sunrise morning day afternoon dusk sunset sunset-limb night-am; do
  ELODIN_CUBESAT_SCENARIO=$s ELODIN_SCREENSHOT_DELAY=12 \
    elodin editor examples/cube-sat/visual_check.py
done
```

Note: the rendered globe is world-fixed at the ECEF origin while the sim's
`Earth` body spins at Earth rate, so continents lag ~22 degrees per orbit —
the same approximation the previous `world_mesh "globe"` visual made.
