# Vendored bbqueue

Elodin vendors [bbqueue](https://github.com/jamesmunns/bbqueue) instead of depending on
external git forks. This crate is a lock-free SPSC ring buffer used as the bridge between
Impeller2 TCP I/O threads and the Bevy editor main loop (`impeller2-bbq`, `impeller2-bevy`).

## Upstream

- **Crate:** bbqueue 0.7.0
- **Tag:** `v0.7.0` (`f212a93ff3729b7f62970ac440621b4091d269ca`)
- **License:** MIT OR Apache-2.0 (see `LICENSE-*` in this directory)

## Elodin patch

Upstream `ArcBBQueue::framed_producer()` / `framed_consumer()` hard-code a `u16` frame length
header, capping each frame at 65,535 bytes. Elodin-db sends single TCP responses up to 8 MiB
(Arrow IPC SQL results, large schema dumps), so we need `usize` headers like the old bbq2 fork.

**Patch file:** `bbqueue/src/queue.rs`

Added two methods on `ArcBBQueue`:

- `framed_producer_with_header::<H: LenHeader>()`
- `framed_consumer_with_header::<H: LenHeader>()`

Elodin uses `H = usize` everywhere:

```rust
queue.framed_producer_with_header::<usize>();
queue.framed_consumer_with_header::<usize>();
```

## History

1. **bbq2 fork** (`elodin-sys/bbq2`) — early Elodin fork with `framed_split()` and `usize`
   headers. Unmaintained; diverged from upstream.
2. **bbqueue migration (PR #610)** — move to actively maintained bbqueue 0.7 with
   `maitake-sync-0_3`, initially via `elodin-sys/bbqueue` git fork (+28 lines).
3. **Vendor in monorepo (`fix/bbq-migration`)** — copy bbqueue 0.7.0 here, apply the usize
   header patch locally, replace all `bbq2` / git `bbqueue` deps with `path` dependencies.

An upstream PR to expose generic frame headers on `ArcBBQueue` would let us drop the patch and
return to crates.io. Until then, sync from upstream by re-copying `v0.7.x` and re-applying the
patch block in `queue.rs`.

## Syncing from upstream

```bash
# From repo root — adjust tag as needed
git clone --depth 1 --branch v0.7.0 https://github.com/jamesmunns/bbqueue.git /tmp/bbqueue
rsync -a --delete /tmp/bbqueue/bbqueue/ vendor/bbqueue/bbqueue/
# Re-apply the framed_*_with_header block in vendor/bbqueue/bbqueue/src/queue.rs
```
