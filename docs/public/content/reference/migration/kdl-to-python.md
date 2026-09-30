+++
title = "Migrate KDL schematics to Python"
description = "Convert an Elodin KDL schematic to the typed Python UI API, then verify model and visual parity."
draft = false
weight = 105
sort_by = "weight"

[extra]
lead = "Move an existing KDL schematic to typed Python builders without changing its behavior."
toc = true
top = false
order = 8
icon = ""
+++

Python schematics make repeated UI structures easier to maintain, provide typed
builders, and allow display expressions and kernels to live beside the rest of
the simulation code. KDL remains the format consumed by the editor; the Python
API builds the same schematic model and emits KDL for the editor.

This guide treats migration as a parity exercise first. Convert and validate the
existing schematic before introducing loops, helper functions, expressions, or
kernels.

## Prerequisites

Build Elodin from the repository root so the CLI and Python package match:

```sh
nix develop
just install
```

Use the resulting development shell for the rest of this guide.

## 1. Inventory the existing schematic

Before converting, identify:

- The root `.kdl` file passed to the editor.
- Other KDL files referenced by the root document.
- Relative paths to GLB models, textures, terrain, and other assets.
- Saved layout overlays that users expect the editor to apply.
- Repeated nodes that are good candidates for helpers or loops later.

Keep the KDL and Python files next to each other during migration. This preserves
the most common relative asset-path assumptions and gives you a known-good
fallback.

## 2. Generate a Python scaffold

Convert the root document:

```sh
elodin schematic to-python path/to/schematic.kdl \
  --output path/to/schematic.py
```

The converter emits typed `elodin.ui` builders. It does not wrap the original
KDL in `ui.from_kdl`, so the result is a useful starting point for Python
maintenance.

The generated module follows this contract:

```python
import elodin.ui as ui


def build() -> ui.Schematic:
    return ui.schematic(
        # Generated builders.
    )
```

Do not refactor the generated file yet. First make sure that it imports, builds,
and describes the same model as the KDL source.

## 3. Read the generated builders

The Python API closely follows the KDL hierarchy:

- The KDL root becomes `ui.schematic(...)`.
- `tabs`, `hsplit`, and `vsplit` become nested builders with the same names.
- Views such as `viewport`, `graph`, and monitor views become typed builder
  calls.
- Scene nodes become builders such as `object_3d`, `line_3d`, `point_trails`,
  `vector_arrow`, `world_mesh`, and `window`.
- Object primitives such as spheres, ellipsoids, and GLB models become nested
  primitive builders.
- KDL properties become keyword arguments.
- Repeated child nodes become Python lists or repeated positional children.
- KDL booleans become `True` and `False`.
- KDL tuples become Python tuples.
- Colors become a `color=` argument or a `ui.color(...)` value.
- Existing EQL queries remain strings unless you deliberately replace them with
  `ui.Expr` objects.

Compare the generated nesting with the source from the outside in. A misplaced
layout or scene child can still produce valid Python while changing what the
editor displays.

## 4. Check model parity

Start with the cheapest checks.

### Import and build

```sh
uv run python -c \
  'import elodin.ui as ui; import schematic; assert isinstance(schematic.build(), ui.Schematic)'
```

Run this from the directory containing `schematic.py`, or adjust the import.

### Inspect emitted KDL

```sh
uv run python schematic.py > /tmp/schematic.generated.kdl
```

A Python schematic module prints its emitted KDL when executed directly. A text
diff can be informative, but it is not definitive because formatting and
equivalent model representations may differ.

### Compare canonical models

Compare parsed models instead of raw KDL text:

```python
from pathlib import Path

import elodin.ui as ui

from schematic import build


def canonical(schematic: ui.Schematic) -> ui.Schematic:
    return ui.from_kdl(schematic.emit_kdl())


source = ui.from_kdl(Path("schematic.kdl").read_text())
assert canonical(build()) == canonical(source)
```

Run the check with `uv run python`. If it fails, inspect one section at a time:
root options, layout, views, scene objects, and styling. Do not compensate for a
conversion mismatch by redesigning the schematic.

## 5. Load it in the editor

Preload the Python schematic with a simulation or database target:

```sh
elodin editor <target> --schematic path/to/schematic.py
```

Use `--schematic` for new commands. The older `--kdl` option is deprecated.

For active development, watch the Python entry point and publish updates to a
running database:

```sh
elodin ui watch path/to/schematic.py --db 127.0.0.1:2240
```

The watcher supports imports from sibling Python modules. Changes to an imported
module may require touching or resaving the watched entry-point file before the
schematic is rebuilt.

Check the editor at a representative point in the data:

- The same tabs and split panes exist.
- Graphs use the same series, ranges, and colors.
- Viewport objects have the same transforms, assets, and visibility.
- Lines, trails, arrows, and world geometry use the same component data.
- Relative assets load without warnings.
- Saved layout overlays still behave as expected.

Model equality cannot catch rendering, asset loading, or runtime query issues,
so visual parity is a required migration check.

## 6. Refactor after parity

Once the generated scaffold passes model and visual checks, use normal Python to
remove repetition:

```python
def body(position: str) -> ui.Object3D:
    return ui.object_3d(
        position,
        mesh=ui.sphere(radius=0.1),
    )


def build() -> ui.Schematic:
    return ui.schematic(
        ui.viewport(name="Viewport", active=True),
        body("vehicle.world_pos"),
        body("target.world_pos"),
    )
```

Prefer small, deterministic helpers. `build()` can be called repeatedly by the
watcher, tests, and CLI, so keep it side-effect-light:

- Do not start a simulation, network client, or background task from `build()`.
- Do not depend on mutable process-global state.
- Avoid environment-dependent output unless it is an intentional input.
- Keep output order stable.

After each refactor, rerun the model comparison and visual check. If the
refactor intentionally changes the model, update the expected result and review
that change separately from the migration.

## 7. Adopt expressions, schemas, and kernels

These features are optional improvements, not prerequisites for conversion.

### Expressions

Use `ui.Expr` to compose readable EQL expressions instead of interpolating
large strings:

```python
position = ui.Expr("vehicle.world_pos")
speed = ui.Expr("vehicle.world_vel").norm()
```

### Schemas

Use `ui.Schema` when the Python code needs the component's tensor shape and data
type, especially when passing inputs to a display kernel:

```python
schema = ui.Schema(
    {
        "vehicle.world_pos": {
            "shape": [3],
            "prim_type": "f64",
        }
    }
)
world_pos = schema["vehicle.world_pos"]
```

A plain component-name string is convenient for direct bindings, but it does
not communicate a non-scalar tensor shape to kernel compilation.

### Display kernels

Use `@ui.kernel` for display-side transformations that benefit from JAX math,
matrix operations, or typed tensor inputs. Keep kernels pure and deterministic.
See `examples/display-kernels/schematic.py` for complete graph and viewport
examples.

When a schematic contains kernels, validate it through `elodin ui watch` or an
explicit Python publish/write flow in addition to editor preload. Kernel
metadata includes sidecar artifacts that are not covered by a plain emitted-KDL
comparison.

## Layout overlays

The editor's **Save Layout** action writes a KDL overlay. It does not rewrite the
Python source. The watcher reapplies the active overlay after publishing a new
base schematic.

Treat the Python module as the authored base layout and the overlay as user
state. If a saved layout should become the new default, make the equivalent
change in Python deliberately, validate it, and then decide whether the old
overlay should be retained.

## Cut over safely

Before replacing the KDL entry point:

1. Confirm the Python module imports and `build()` returns a schematic.
2. Confirm canonical model equality, or document every intentional difference.
3. Exercise the schematic against representative telemetry.
4. Compare editor screenshots for layout and rendering parity.
5. Verify all relative assets resolve from the intended launch directory.
6. Verify watch updates and saved overlays.
7. Verify every display kernel with its actual component shapes and data types.
8. Keep the old KDL available until downstream launch scripts and users have
   switched to `--schematic`.

Update launch commands, examples, and documentation together. Delete the old KDL
only when it is no longer a runtime input or rollback path.

## Troubleshooting

### `build()` is missing or returns the wrong value

The module must expose a zero-argument `build()` function returning
`ui.Schematic`. Return the result of `ui.schematic(...)`, not a layout node or
raw KDL string.

### Python cannot import `elodin`

Run `just install` in `nix develop`, then execute Python with `uv run`. Confirm
that the CLI and Python package were built from the same checkout.

### A sibling import fails

Keep imported modules beside the schematic entry point or make them part of an
installed package. The CLI adds the schematic's directory to `sys.path`.

### An asset disappeared

Check the asset path before changing the builder. Relative paths are sensitive
to where the schematic is compiled and where assets are served. Keeping the
Python file beside the old KDL during migration avoids many path changes.

### Canonical models differ

Reduce the comparison by temporarily building one top-level section at a time.
Check root properties before layouts, layouts before views, and views before
their scene children. Compare parsed models rather than formatting.

### The model matches but the view does not

Inspect runtime queries, component tensor shapes, asset-server logs, and display
kernel diagnostics. Then compare at a known simulation timestamp. A structural
comparison cannot validate live data or rendering.

### Saving a layout did not edit the Python file

This is expected. Saved layouts are overlays; copy intentional default changes
back into the Python builders manually.

## Examples and reference

- `examples/ball/schematic.py` shows layouts, viewport objects, lines, arrows,
  and coordinate frames.
- `examples/display-kernels/schematic.py` shows expressions, schemas, and
  display kernels.
- [Schematic reference](@/reference/schematic.md) documents the model and
  editor-facing concepts represented by both authoring formats.
