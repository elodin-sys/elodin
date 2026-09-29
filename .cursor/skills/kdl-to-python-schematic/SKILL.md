---
name: kdl-to-python-schematic
description: Migrate Elodin KDL schematics to typed Python elodin.ui builders, validate model and visual parity, and refactor generated scaffolds safely. Use when converting .kdl files, adopting Python-authored schematics, or troubleshooting elodin schematic to-python output.
---

# KDL to Python schematic migration

Convert an existing KDL schematic without silently changing its model or visual
behavior. Establish parity before improving the generated Python.

Read the [migration guide](../../../docs/public/content/reference/migration/kdl-to-python.md)
before making changes. Use
[`examples/ball/schematic.py`](../../../examples/ball/schematic.py) for basic
builders and
[`examples/display-kernels/schematic.py`](../../../examples/display-kernels/schematic.py)
for expressions, schemas, and display kernels.

## Rules

- Work from the repository root in `nix develop`.
- Use `uv` for every Python command.
- Run `just install` before testing so the CLI and Python extension match.
- Preserve the source KDL until the user explicitly approves deletion.
- Keep `build()` deterministic and side-effect-light.
- Preserve asset paths, ordering, names, queries, colors, and layout structure
  during the initial conversion.
- Do not redesign and migrate in the same step.
- Do not replace generated builders with embedded KDL or `ui.from_kdl`.
- Prefer `--schematic`; `--kdl` is deprecated.

## Workflow

### 1. Inventory

Locate the root KDL, referenced KDL documents, relative assets, launch commands,
and saved layout behavior. Record which database or simulation target can
exercise representative live data.

Check whether the schematic contains display kernels or whether the migration is
expected to introduce them. Kernel validation requires more than emitted KDL.

### 2. Generate

Keep the Python file beside the KDL initially:

```sh
elodin schematic to-python path/to/schematic.kdl \
  --output path/to/schematic.py
```

Inspect the generated module. It must import `elodin.ui`, define
`build() -> ui.Schematic`, return `ui.schematic(...)`, and retain its executable
`emit_kdl()` block.

### 3. Validate the scaffold

Run the entry point:

```sh
uv run python path/to/schematic.py > /tmp/schematic.generated.kdl
```

Compare canonical models in Python:

```python
from pathlib import Path

import elodin.ui as ui

from schematic import build


def canonical(schematic: ui.Schematic) -> ui.Schematic:
    return ui.from_kdl(schematic.emit_kdl())


source = ui.from_kdl(Path("schematic.kdl").read_text())
assert canonical(build()) == canonical(source)
```

When equality fails, isolate top-level sections in this order:

1. Root options.
2. Layout and tabs.
3. Views and their properties.
4. Scene objects.
5. Styling and colors.

Compare parsed models, not only raw KDL text.

### 4. Validate runtime behavior

Preload the schematic:

```sh
elodin editor <target> --schematic path/to/schematic.py
```

Then exercise live updates:

```sh
elodin ui watch path/to/schematic.py --db 127.0.0.1:2240
```

Capture the same representative state with the KDL and Python versions. Compare:

- Tabs, splits, active view, and panel sizing.
- Graph series, ranges, labels, and colors.
- Viewport objects, transforms, visibility, and assets.
- Lines, trails, arrows, coordinates, and world geometry.
- Runtime queries, warnings, and build diagnostics.
- Saved layout overlay behavior.

If the entry point imports sibling modules, verify imports and touch the watched
entry point after changing a sibling if the watcher does not rebuild.

### 5. Refactor incrementally

Only after parity:

- Extract small helpers.
- Replace repeated builders with comprehensions or loops.
- Compose query strings with `ui.Expr`.
- Use `ui.Schema` when tensor shape and type matter.
- Add `@ui.kernel` only when a display transformation needs typed JAX math.

Re-run canonical and visual comparisons after each logical refactor. Keep
intentional redesigns separate and document their expected differences.

### 6. Validate kernels and overlays

Plain `emit_kdl()` parity does not validate display-kernel sidecars. Test kernels
through `elodin ui watch` or an explicit Python publish/write flow using the
actual component shapes and primitive types.

The editor's Save Layout action writes an overlay KDL; it does not edit Python.
Confirm that the watcher reapplies the active overlay. Copy desired default
layout changes into Python manually.

### 7. Cut over

Update launch commands, examples, and documentation to use the Python path.
Verify relative assets from the real launch environment. Retain the KDL as a
rollback until all consumers have switched and the user approves removal.

## Completion report

Report:

- Files converted and files intentionally retained.
- Canonical equality result and any intentional model differences.
- Runtime target and visual states tested.
- Asset, overlay, watcher, and kernel results.
- Refactors performed after parity.
- Commands run and remaining risks.
