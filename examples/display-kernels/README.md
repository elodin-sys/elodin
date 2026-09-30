# Display Kernels

A circling craft whose packed 3×3 covariance is turned into a live ellipsoid
and a graph by Python `@ui.kernel` functions. The schematic is authored in
`elodin.ui` and passed to `world.schematic()` — the kernel sidecars travel
into the DB with the KDL so `elodin editor` can fetch and JIT them.

What to look for:

- The cyan ellipsoid is **oblate** (flat vertically) and **stretches along-track**
  as heading changes. Grid lines should stay visible on the dark theme.
- The top graph is a **display kernel** plotting `det(P)` and `trace(P)`.
- The bottom graph is plain **EQL** of the packed 6-vector, for contrast.

Run from the repository root, inside `nix develop` (after `just install`):

```sh
elodin editor examples/display-kernels/main.py
```

## Iterate on the schematic

With the editor connected, rebuild and push the schematic on every save:

```sh
elodin ui watch examples/display-kernels/schematic.py --db 127.0.0.1:2240
```

Print the generated KDL (and compile the kernels) without launching the editor:

```sh
uv run python examples/display-kernels/schematic.py
```
