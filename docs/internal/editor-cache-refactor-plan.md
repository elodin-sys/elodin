# Editor Cache-First Playback Architecture

**Status**: Draft — for discussion  
**Author**: Generated from codebase investigation  
**Date**: 2026-02-25  

## Executive Summary

The Elodin Editor's playback timeline is tightly coupled to the Elodin-DB streaming API. Every user interaction (play, pause, scrub, jump-to-start/end) translates into `SetStreamState` messages sent to the database, which must then react by adjusting its internal `FixedRateStreamState`, advancing or rewinding the stream position, and sending back the correct data plus metadata like `StreamTimestamp` and `LastUpdated`. This round-trip coupling is brittle: changes to either the Editor UX or the DB streaming logic frequently break the other side.

This document proposes a **cache-first architecture** where the Editor maintains a local cache of all telemetry data, and all playback behavior operates entirely on top of that cache. The DB's role simplifies to *delivering data* — the Editor owns playback state, scrubbing, and replay.

---

## 1. Current Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Editor (Bevy + Egui)                                                    │
│                                                                         │
│  Timeline UI ──► SetStreamState ──────────────────────────┐             │
│       ▲                                                   │             │
│       │                                                   ▼             │
│  CurrentTimestamp ◄── StreamTimestamp ◄──── FixedRate ◄── DB            │
│  LastUpdated ◄─────── SubscribeLastUpdated ◄──────────── DB            │
│  EarliestTimestamp ◄── GetEarliestTimestamp ◄─────────── DB            │
│                                                                         │
│  Plot panels ◄── GetTimeSeries ◄── DB (historical chunks)              │
│  Video panels ◄── GetMsgs ◄── DB (historical frames)                  │
│  3D Viewport ◄── Table packets ◄── FixedRate stream ◄── DB            │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Coupling Points

| Editor action | Protocol message | DB-side handler |
|---|---|---|
| Play/Pause | `SetStreamState { playing }` | `FixedRateStreamState::set_playing()` |
| Scrub to timestamp | `SetStreamState { timestamp }` | `FixedRateStreamState::set_timestamp()` → sets `is_scrubbed` flag |
| Change speed | `SetStreamState { time_step }` | `FixedRateStreamState::set_time_step()` |
| Frame step | `SetStreamState::rewind()` | `FixedRateStreamState::set_timestamp()` → wakes tick driver |
| Jump to start/end | `SetStreamState::rewind()` | Same as scrub |
| Connect | `Stream { FixedRate }` | Creates `FixedRateStreamState`, spawns tick driver + stream handler |

### What the DB Must Track for the Editor

The `FixedRateStreamState` in `libs/db/src/lib.rs` maintains:

- `current_tick: AtomicI64` — the current playback timestamp
- `is_playing: AtomicBool` — play/pause state
- `is_scrubbed: AtomicBool` — flag to allow render during pause
- `time_step: AtomicU64` — playback speed (nanoseconds per tick)
- `tick_notify: WaitQueue` — wakes stream handler when tick advances
- Tick driver background task (wall-clock → tick advancement, scaled by speed)

The `--replay` flag adds more state:

- `replay: AtomicBool` — replay mode active
- `replay_end: AtomicI64` — original end timestamp
- `last_updated` is directly stored (not max'd) to simulate "data arriving"
- Playback stops when current timestamp ≥ `replay_end`

### Problems with Current Architecture

1. **Bidirectional coupling**: The Editor sends playback commands to the DB; the DB sends back position/timing metadata. A change on either side can break the contract.

2. **Latency-sensitive round-trip**: Scrubbing requires: Editor → TCP → DB state update → DB renders data at new position → TCP → Editor displays. This introduces visible latency on scrub.

3. **DB owns playback logic**: Play/pause, speed, tick advancement all live in the DB. The DB is a *data store*, not a *media player* — this is a misplaced responsibility.

4. **Replay is a DB concern**: `elodin-db --replay` manipulates `last_updated` to simulate data arrival. This is an Editor UX feature (playing back recorded data) implemented in the wrong layer.

5. **Edge cases compound**: Scrub-while-paused requires `is_scrubbed` flag. Replay requires `replay_end` bounds checking. Backward scrub in replay requires `last_updated` direct store instead of `update_max`. Each edge case adds fragile DB-side logic.

6. **3D viewport can't scrub smoothly**: The viewport data comes from the FixedRate stream, which only sends data at the DB-computed tick rate. Local scrubbing would be instant if data were cached.

---

## 2. Reference Implementations (Already in Codebase)

### Video Stream Panel — `VideoFrameCache`

`libs/elodin-editor/src/ui/video_stream.rs` already implements a cache-first pattern for video:

```
DB ──GetMsgs (paginated)──► raw_frames cache (BTreeMap<Timestamp, Vec<u8>>)
DB ──FixedRateMsgStream──►  raw_frames cache (live tail)

Playback:
  CurrentTimestamp → cache lookup → decoder → decoded_frames cache → display
```

**Key design choices:**
- Historical backfill via paginated `GetMsgs` (200 frames/page)
- Live-tail via `FixedRateMsgStream` subscription
- Playback reads from cache, *never* requests data per-frame from DB
- Keyframe index for efficient seeking
- LRU eviction for decoded frames, FIFO for raw frames
- Works offline once cache is warm

### Plot System — `LineTree` + Chunked Loading

`libs/elodin-editor/src/ui/plot/data.rs` implements a hybrid cache:

```
DB ──GetTimeSeries (chunked)──► LineTree (interval tree of Chunk<f32>)
DB ──FixedRateMsgStream──►      append to latest chunk (live tail)

Rendering:
  SelectedTimeRange → find chunks in range → render from cache
```

**Key design choices:**
- Chunked storage (3072 points per chunk) with interval-tree indexing
- Gap detection and automatic backfill
- Pagination: requests next chunk when previous is full
- GPU buffer allocation only for visible chunks
- Garbage collection outside visible range
- Overview mode (LTTB downsampling) for large time ranges

### What's Missing: Component State Cache for 3D Viewport

The 3D viewport currently receives entity state *only* from the FixedRate stream at the DB-computed timestamp. There is no local cache — if you scrub, you must wait for the DB to send data at the new position.

---

## 3. Proposed Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Editor                                                                  │
│                                                                         │
│  ┌─────────────┐    ┌──────────────────────────────────────────┐        │
│  │ DB Connection│───►│           Telemetry Cache                │        │
│  │ (data only)  │    │                                          │        │
│  │              │    │  Component data:  BTreeMap<Ts, Table>    │        │
│  │  - RealTime  │    │  Message logs:    BTreeMap<Ts, Bytes>   │        │
│  │    or Batch   │    │  Metadata:        ComponentRegistry     │        │
│  │  - Backfill  │    │                                          │        │
│  └──────────────┘    │  earliest_ts ◄─────────────────────────┐│        │
│                      │  latest_ts ◄───────────────────────────┤│        │
│                      └──────────────────────┬─────────────────┘│        │
│                                             │                   │        │
│  ┌──────────────────────────────────────────┤                   │        │
│  │          Playback Controller             │                   │        │
│  │                                          │                   │        │
│  │  CurrentTimestamp ◄── local timer        │                   │        │
│  │  Paused            (wall-clock based)    │                   │        │
│  │  PlaybackSpeed                           │                   │        │
│  │  TimeRange                               │                   │        │
│  └────────────┬─────────────────────────────┘                   │        │
│               │                                                  │        │
│  ┌────────────┼──────────────────────────────────────────────┐  │        │
│  │            ▼          Consumers                           │  │        │
│  │                                                           │  │        │
│  │  Timeline UI ──► reads CurrentTimestamp, TimeRange        │  │        │
│  │  3D Viewport ──► cache.get_at(CurrentTimestamp)           │  │        │
│  │  Plot panels ──► cache.get_range(SelectedTimeRange)       │  │        │
│  │  Video panels ──► cache.get_frame_at(CurrentTimestamp)    │  │        │
│  │  Inspector ──► cache.get_latest_at(CurrentTimestamp)      │  │        │
│  └───────────────────────────────────────────────────────────┘  │        │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core Principles

1. **DB delivers data, Editor owns playback.** The DB streams data to the Editor as efficiently as possible. The Editor decides what to display and when.

2. **Cache is the single source of truth for display.** All UI consumers (timeline, viewport, plots, video, inspector) read from the cache at `CurrentTimestamp` or `SelectedTimeRange`. No consumer directly interacts with the DB for position-dependent data.

3. **Local playback timer.** `CurrentTimestamp` advances from a local wall-clock timer in the Editor, scaled by `PlaybackSpeed`. No `SetStreamState` round-trip needed for play/pause/scrub.

4. **Replay is an Editor feature.** Opening a recorded database and playing it back is just "load data into cache, play from cache." No special DB mode needed.

5. **Incremental, backward-compatible migration.** Each phase delivers value independently and can be shipped/tested separately.

---

## 4. Detailed Design

### 4.1 `TelemetryCache` — Shared Data Cache

A new Bevy `Resource` that stores all component time-series data in a queryable structure.

```
TelemetryCache
├── components: HashMap<ComponentId, ComponentTimeSeries>
│   └── ComponentTimeSeries
│       ├── schema: Schema
│       ├── data: BTreeMap<Timestamp, ComponentValue>  // or chunked like LineTree
│       ├── earliest: Timestamp
│       └── latest: Timestamp
├── earliest_timestamp: Timestamp
├── latest_timestamp: Timestamp      // = max of all component latest
├── metadata: ComponentMetadataRegistry
└── schemas: ComponentSchemaRegistry
```

**Design considerations:**

- **Chunked vs. flat storage**: The plot system's `LineTree` (chunked interval tree) is proven and handles large datasets well. The component cache could use a similar chunked approach, or a simpler `BTreeMap` for entity-state data (which has fewer points than high-frequency sensor data).

- **Memory management**: Eviction policy based on distance from `CurrentTimestamp` and `SelectedTimeRange`. Keep a configurable window of data in memory; evict outside the window.

- **Thread safety**: Cache populated from async I/O thread, read from Bevy systems. Use `Arc<RwLock<>>` or Bevy's change detection. Alternatively, use a channel-based approach (existing `PacketRx`/`PacketTx` pattern).

- **Unification opportunity**: The plot system's `CollectedGraphData` and the video system's `VideoFrameCache` could share the underlying `TelemetryCache`, with view-specific projections on top.

### 4.2 `PlaybackController` — Local Playback State Machine

A Bevy system that owns `CurrentTimestamp` advancement, replacing the DB's tick driver.

```rust
// Bevy system running in Update
fn advance_playback(
    time: Res<Time>,
    mut current_ts: ResMut<CurrentTimestamp>,
    paused: Res<Paused>,
    speed: Res<PlaybackSpeed>,
    cache: Res<TelemetryCache>,
) {
    if paused.0 {
        return;
    }
    let delta_micros = (time.delta_secs_f64() * speed.0 * 1_000_000.0) as i64;
    let new_ts = Timestamp(current_ts.0.0 + delta_micros);
    // Clamp to cached data range
    let clamped = new_ts.clamp(cache.earliest_timestamp, cache.latest_timestamp);
    current_ts.0 = clamped;
}
```

**Replaces:**
- `FixedRateStreamState` tick driver in the DB
- `sync_paused()` system that sends `SetStreamState { playing }` to DB
- `SetStreamState::rewind()` for scrubbing (now just set `CurrentTimestamp` directly)
- Timeline controls' `PacketTx` usage for playback commands

**Timeline controls change from:**
```rust
// Current: send command to DB
event.send_msg(SetStreamState::rewind(stream_id, timestamp));
```
**to:**
```rust
// Proposed: set local resource directly
current_timestamp.0 = timestamp;
```

### 4.3 DB Streaming Simplification

With the Editor owning playback, the DB stream simplifies to:

#### New: Data Delivery Stream

Instead of the current `FixedRate` stream (which sends data at DB-computed timestamps), the DB provides a **data delivery stream** that sends new data as it arrives:

- **Live simulation**: `RealTimeBatched` behavior — sends component updates as data is written, batched per `last_updated` change. The Editor inserts into cache.
- **Recorded data**: On connection, Editor sends `GetTimeSeries` / bulk fetch requests to populate the cache. No special `--replay` mode needed.

#### Removable DB-side Concerns

| Current DB feature | Why it can be removed |
|---|---|
| `FixedRateStreamState` playback management | Editor owns playback locally |
| `is_scrubbed` flag | No scrub commands from Editor |
| `replay` mode + `replay_end` | Editor loads data into cache; playback is local |
| `last_updated` manipulation in replay | `last_updated` reflects actual data extent only |
| Tick driver (wall-clock → tick) | Editor has its own `advance_playback()` |
| `SetStreamState` play/pause handling | Not needed; no playback state in DB |

#### Retained DB Features

| Feature | Why it stays |
|---|---|
| `RealTime` / `RealTimeBatched` streaming | Data delivery for live simulations |
| `FollowStream` (DB-to-DB replication) | Unrelated to Editor playback |
| `GetTimeSeries` (historical queries) | Cache population |
| `GetMsgs` (message log queries) | Cache population |
| `SubscribeLastUpdated` | Editor needs to know when new data arrives |
| `GetEarliestTimestamp` | Initial cache range |
| `DumpMetadata` / `DumpSchema` | Schema information |

### 4.4 Connection Flow (After Refactor)

```
Editor connects:
  1. DumpMetadata
  2. DumpSchema
  3. GetEarliestTimestamp
  4. SubscribeLastUpdated
  5. Stream { RealTimeBatched }          // for live data delivery
  6. Bulk GetTimeSeries requests          // populate cache for available range

Editor receives:
  - Metadata + schema (one-time)
  - EarliestTimestamp (one-time)
  - LastUpdated (subscription, fires on new data)
  - RealTimeBatched data (ongoing, as simulation writes)
  - GetTimeSeries responses (paginated, for historical backfill)

Editor playback:
  - All local. No SetStreamState messages sent.
  - CurrentTimestamp advances from local timer.
  - All consumers read from TelemetryCache.
```

---

## 5. Migration Plan

### Phase 1: Local Playback Controller

**Goal**: Decouple play/pause/scrub from DB. Editor advances `CurrentTimestamp` locally.

**Changes:**
- Add `PlaybackController` system to the Editor
- Remove `sync_paused()` system (no more `SetStreamState { playing }` to DB)
- Timeline controls set `CurrentTimestamp` directly instead of sending `SetStreamState::rewind()`
- Keep the FixedRate stream as-is for now, but ignore `StreamTimestamp` for `CurrentTimestamp` advancement
- DB still sends data at its own rate; Editor displays whatever is available at `CurrentTimestamp`

**Deliverables:**
- Play/pause is instant (no round-trip)
- Scrubbing is instant (no round-trip)
- Frame stepping is instant
- 3D viewport still reads from FixedRate stream data (not yet cached)

**Risk**: 3D viewport may show stale data if `CurrentTimestamp` advances faster than the stream delivers. This is acceptable as a transitional state.

**Estimated scope**: ~200-400 lines changed in `libs/elodin-editor/`

### Phase 2: Component Data Cache

**Goal**: All consumers read from a local cache. 3D viewport no longer depends on stream timing.

**Changes:**
- Implement `TelemetryCache` resource
- Populate cache from incoming `Table` packets (existing `WorldSink` path)
- Populate cache with historical data via `GetTimeSeries` requests on connect
- 3D viewport reads entity state from cache at `CurrentTimestamp`
- Plot system optionally reads from shared cache instead of its own `LineTree`

**Deliverables:**
- Smooth scrubbing in 3D viewport (reads cached data instantly)
- Historical data available for scrubbing before stream has delivered it
- Unified data source for all panels

**Risk**: Memory usage — caching all component data for all entities for all timestamps could be large. Mitigated by windowed eviction and chunked storage.

**Estimated scope**: ~500-1000 lines new code for cache, ~200-400 lines refactoring consumers

### Phase 3: Move Replay to Editor

**Goal**: Remove `--replay` from `elodin-db`. Recorded data playback is an Editor feature.

**Changes:**
- Editor detects "recorded DB" (no live data arriving) from `LastUpdated` not changing
- Offers play-from-start UX (already natural with local playback controller)
- Remove `replay` mode from DB (`enable_replay_mode()`, `replay_end`, `last_updated` direct store)
- Remove `--replay` CLI flag

**Deliverables:**
- Simpler DB code
- Replay works with any recorded DB without special flags
- Auto-detect live vs. recorded mode in Editor

**Estimated scope**: ~100-200 lines removed from DB, ~50-100 lines added to Editor

### Phase 4: Simplify DB Streaming API

**Goal**: Remove playback state management from the DB entirely.

**Changes:**
- Remove `SetStreamState` message handling for play/pause/timestamp (keep for potential non-Editor clients if needed, or deprecate)
- Simplify `FixedRateStreamState` to only manage data delivery rate
- Or: replace `FixedRate` with `RealTimeBatched` for all Editor connections
- Remove tick driver complexity (`is_scrubbed`, speed scaling, etc.)
- Simplify `SubscribeLastUpdated` handler (no more replay-mode special cases)

**Deliverables:**
- Significantly simpler DB streaming code
- Fewer edge cases in stream management
- Clearer API contract: DB delivers data, clients render it

**Estimated scope**: ~300-500 lines removed from DB

### Phase 5 (Optional): Unified Cache

**Goal**: Plot, video, and viewport all share one `TelemetryCache`.

**Changes:**
- Migrate plot `LineTree` data storage to `TelemetryCache`
- Migrate video `VideoFrameCache` raw frames to `TelemetryCache`
- Keep view-specific projections (GPU buffers for plots, decoder for video)
- Single backfill/gap-detection system

**Deliverables:**
- Single source of truth for all data
- Shared backfill logic (less code duplication)
- Consistent eviction and memory management

**Estimated scope**: Large refactor (~1000-2000 lines). Should only be pursued if Phase 2 proves the architecture.

---

## 6. Risk Analysis

| Risk | Impact | Mitigation |
|---|---|---|
| Memory pressure from caching all data | High | Windowed eviction; chunked storage; configurable cache size |
| 3D viewport interpolation gaps | Medium | Cache includes interpolation support; show nearest-available marker |
| Breaking non-Editor clients | Medium | Keep `SetStreamState` protocol message; deprecate rather than remove |
| Large refactor surface area | High | Phased approach; each phase is independently shippable |
| Performance regression on cache reads | Low | BTreeMap/interval-tree lookups are O(log n); benchmark |
| Live simulation latency | Low | `RealTimeBatched` already delivers data at ~60Hz; cache insertion is O(log n) |

---

## 7. Success Metrics

1. **Scrub latency**: From ~50-100ms (TCP round-trip) to <1ms (local cache read)
2. **Code complexity**: `FixedRateStreamState` reduced from ~500 lines to ~100 lines
3. **Bug surface area**: Eliminate `is_scrubbed`, replay-mode `last_updated` manipulation, and `SetStreamState` edge cases
4. **Feature parity**: All existing timeline behaviors (play, pause, scrub, frame step, jump, speed change, time range selection) work identically from the user's perspective
5. **Replay just works**: Opening a recorded DB in the Editor plays back without `--replay` flag

---

## 8. Open Questions

1. **Cache granularity for 3D viewport**: Should the component cache store every sample at full fidelity, or downsample for older data (like plot overview mode)?

2. **Cache warming strategy**: On connect to a large recorded DB, should we backfill the entire dataset or use lazy loading (fetch on scrub)?

3. **Live simulation edge**: When a simulation is actively writing data, should the Editor auto-follow the latest timestamp (current behavior) or require explicit "follow live" mode?

4. **Multi-stream support**: The current architecture allows multiple `Stream` instances with different behaviors. Does the cache-first approach need to support this?

5. **Non-Editor clients**: If we remove `SetStreamState` handling, do any non-Editor clients (C/C++ clients, Lua REPL) depend on it?

6. **VTable stream**: The current `VTableStream` is used for individual component subscriptions. Should this be replaced by cache-driven reads, or kept for efficiency?

---

## Appendix A: File Map

### Editor Files to Modify

| File | Changes |
|---|---|
| `libs/elodin-editor/src/lib.rs` | Add `PlaybackController` system; remove `sync_paused()` |
| `libs/elodin-editor/src/ui/timeline/mod.rs` | Read from cache for time range |
| `libs/elodin-editor/src/ui/timeline/timeline_controls.rs` | Set `CurrentTimestamp` directly; remove `PacketTx` usage |
| `libs/elodin-editor/src/ui/timeline/timeline_slider.rs` | Set `CurrentTimestamp` directly; remove `SetStreamState::rewind()` |
| `libs/elodin-editor/src/ui/plot/data.rs` | (Phase 5) Read from shared cache |
| `libs/elodin-editor/src/ui/video_stream.rs` | (Phase 5) Read from shared cache |
| `libs/impeller2/bevy/src/lib.rs` | Populate `TelemetryCache`; change connection setup |

### DB Files to Simplify

| File | Changes |
|---|---|
| `libs/db/src/lib.rs` | Remove `FixedRateStreamState` playback logic, replay mode |
| `libs/db/src/main.rs` | Remove `--replay` flag |
| `libs/db/src/vtable_stream.rs` | Simplify to data-delivery only |

### New Files

| File | Purpose |
|---|---|
| `libs/elodin-editor/src/cache.rs` (or `src/telemetry_cache/`) | `TelemetryCache` implementation |
| `libs/elodin-editor/src/playback.rs` | `PlaybackController` system |

---

## Appendix B: Current Protocol Message Inventory

### Messages the Editor Sends to DB

| Message | Purpose | After Refactor |
|---|---|---|
| `Stream { FixedRate }` | Create playback stream | Replace with `Stream { RealTimeBatched }` |
| `SetStreamState { playing }` | Play/pause | **Remove** |
| `SetStreamState { timestamp }` | Scrub/rewind | **Remove** |
| `SetStreamState { time_step }` | Speed change | **Remove** |
| `GetEarliestTimestamp` | Initial time range | Keep |
| `SubscribeLastUpdated` | Track data extent | Keep |
| `DumpMetadata` | Schema info | Keep |
| `DumpSchema` | Schema info | Keep |
| `GetDbSettings` | DB config | Keep |
| `GetTimeSeries` | Historical data | Keep (primary cache population) |
| `GetMsgs` | Historical messages | Keep (video cache population) |
| `VTableStream` | Component subscription | Keep or replace with cache reads |

### Messages the DB Sends to Editor

| Message | Purpose | After Refactor |
|---|---|---|
| `StreamTimestamp` | Current playback position | **Remove** (local timer) |
| `Table` (from FixedRate) | Component data at current tick | Replace with `RealTimeBatched` tables |
| `LastUpdated` | Data extent | Keep |
| `EarliestTimestamp` | Data start | Keep |
| `DumpMetadataResp` | Metadata | Keep |
| `DumpSchemaResp` | Schemas | Keep |
| `OwnedTimeSeries` | Historical data response | Keep |
| `MsgBatch` | Historical messages response | Keep |
