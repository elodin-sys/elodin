# Schematic golden corpus (Phase 0)

Checked-in KDL sources used by `golden_corpus` tests (`cargo test -p impeller2-kdl --test golden_corpus`):

- `sources/examples/` — snapshot of this monorepo's `examples/*/*.kdl`
- `sources/fsw/` — vendored from `../fsw/assets/schematics` (sibling flight-software repo)

JSON goldens under `goldens/` are the **canonical** model after `parse → emit → parse`, with object keys sorted for stable diffs.

## Phase 0 decisions (locked)

- Authoring: `elodin.ui` (Python) emits KDL; KDL remains the wire/artifact format
- Expressions: typed Python frontend → `eql::Expr` (Tier B/C later)
- Editor: unchanged in Phases 0–2; layout overlay later

Refresh goldens after intentional serializer/parser changes:

```bash
BLESS_GOLDENS=1 cargo test -p impeller2-kdl --test golden_corpus
```
