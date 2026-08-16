# Chapter 1 — Simple interplanetary gravity

Chapter 1 is the Voyager example's intentionally simple starting point.

It answers the first question in the educational journey:

> What happens if we initialize Voyager from SPICE and propagate it using
> Newtonian gravity from the Sun and major planets?

## Run it

From the repository root:

```bash
python examples/voyager/main.py run
```

The simulation begins at the same SPICE state as the reference trajectory.
Elodin then propagates the red Voyager bodies while SPICE updates the green
reference bodies.

The interactive run uses the long-span merged Voyager SPKs. Reconstructed
encounter arcs are kept separate for scientific validation so disagreement
between two trajectory solutions is not counted as force-model error. See the
[truth-reference contract](truth_reference.md) for that distinction.

## Model

For one gravity source, Chapter 1 uses the direct Newtonian attraction

```text
r = r_probe - r_source
F = -G M m r / |r|^3
```

where:

- `M` is the source mass;
- `m` is the Voyager mass;
- `r` points from the source to the probe;
- the minus sign points the force back toward the source.

The model includes the Sun and the eight major-planet entries already defined
by the example.

## What stays intentionally simple

Chapter 1 is a useful first interplanetary model, not a high-fidelity
reconstruction of the Voyager missions.

It deliberately keeps the original assumptions so each later chapter can add
one concept and measure what that concept changes.

The existing telemetry records:

- `position_error_km`;
- `velocity_error_mps`.

Those signals turn the difference from SPICE into something we can investigate
rather than hiding it behind two visually similar trajectories.

## Question for Chapter 2

The state vectors are expressed relative to the Sun.

The Chapter 1 model gives the planets' direct gravitational acceleration to
Voyager, while the Sun remains at the coordinate origin.

That raises the next question:

> If the planets pull on Voyager, what happens when those same planets also
> accelerate the Sun that defines our origin?

[Continue to Chapter 2 — Heliocentric relative dynamics](chapter_2.md)
