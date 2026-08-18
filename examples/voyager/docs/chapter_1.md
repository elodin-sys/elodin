# Chapter 1 — Simple interplanetary gravity

Chapter 1 is the intentionally simple starting model.

Run it from the repository root:

```bash
python examples/voyager/main.py run
```

The probes start from SPICE states. Elodin then propagates them while SPICE
updates the reference trajectories.

For each source, the model applies direct Newtonian attraction:

```text
r = r_probe - r_source
F = -G M m r / |r|^3
```

It includes the Sun and the eight major planets. The model intentionally omits
mission-specific effects, so the result is not a high-fidelity reconstruction.

The interactive run uses long-span merged Voyager SPKs. The separate
[truth-reference validation](truth_reference.md) uses reconstructed encounter
SPKs so differences between reference solutions are not counted as simulation
error.

[Continue to Chapter 2](chapter_2.md)
