# elodin-db Profiling Report (all scenarios)

---

# elodin-db Profiling Report

**Generated:** 21:29:35 UTC

## Configuration

| Parameter | Value |
|---|---|
| Scenario | customer (per-component) |
| Components | 400 |
| Frequency | 250 Hz |
| Duration | 10 s |
| Mode | per-component |
| Total ticks recorded | 8192 |
| Total `sink_table` calls | 776800 |

## Tick Latency Distribution

| Percentile | Latency |
|---|---|
| p50 | 1.3 µs |
| p75 | 1.9 µs |
| p90 | 2.5 µs |
| p95 | 2.9 µs |
| p99 | 4.1 µs |
| max | 12.9 µs |
| mean | 1.6 µs |

## Phase Breakdown (cumulative)

| Phase | Total Time | % of sink_table | Calls | Mean/call | Source |
|---|---|---|---|---|---|
| RwLock acquire | 54.3 ms | 3.8% | 776800 | 70 ns | `lib.rs` `db.with_state()` |
| VTable resolve | 1206.8 ms | 84.0% | 5844 | 206503 ns | `vtable.rs` `realize_fields()` |
| apply_value total | 1017.9 ms | 70.8% | 776800 | 1310 ns | `lib.rs` `DBSink::apply_value()` |
|   push_buf | 388.6 ms | 27.0% | 776800 | 500 ns | `time_series.rs` `push_buf()` |
|     data write | 125.9 ms | 8.8% | 776800 | 162 ns | `append_log.rs` `data.write()` |
|     index write | 36.6 ms | 2.5% | 776800 | 47 ns | `append_log.rs` `index.write()` |
|     timestamp check | 35.9 ms | 2.5% | 776800 | 46 ns | `time_series.rs` last_ts read |
|   wake_all (data) | 40.3 ms | 2.8% | 776800 | 52 ns | `time_series.rs` `data_waker` |
|   wake_all (last_upd) | 244.2 ms | 17.0% | 776800 | 314 ns | `lib.rs` `update_max()` |
|   wake_all (earliest) | 61.9 ms | 4.3% | 776800 | 80 ns | `lib.rs` `update_min()` |
|   HashMap lookup | 52.5 ms | 3.7% | 776800 | 68 ns | `lib.rs` `components.get()` |

## Identified Bottlenecks

### #1 — VTable realize chain (1206.8 ms, 84.0% of total)

5844 field resolutions (0/tick). Each field triggers ~7 recursive `realize()` calls. The VTable structure is static — results could be cached after first resolve.

### #2 — wake_all() cascade (346.3 ms, 24.1% of total)

2330400 total calls (3/tick). `update_min` rarely changes the value but calls `wake_all()` unconditionally. `update_max` is called 1× per tick with the same timestamp — only the first matters.

### #3 — push_buf Mutex churn (162.5 ms, 11.3% of total)

2 Mutex acquisitions per component per tick = 1553600 lock/unlock pairs. Uncontested in single-client batch mode but still ~20 ns each.

### #4 — HashMap lookups (52.5 ms, 3.7% of total)

776800 lookups. With 400 components the table fits in L2 cache, but pointer chasing to heap-allocated Components causes cache misses.

## Outlier Detection

| Metric | Value |
|---|---|
| Worst single `push_buf` | 74.9 µs |
| Worst single `apply_value` | 437.9 µs |
| Tick p99/p50 ratio | 3.1× |

> **Warning:** A single `push_buf` call took more than half the mean tick time. This is likely a **mmap page fault** on first write to a new 4 KB page.

## Optimization Recommendations

1. **Debounce `wake_all()` in `apply_value`** — Currently 24% of tick time. Call `update_max`/`update_min` once after the entire batch instead of per-component. Skip `wake_all()` when the atomic value didn't actually change.

2. **Cache VTable resolution** — Currently 84% of tick time. Pre-compile the field→component mapping on first `apply()` and reuse it on subsequent ticks, bypassing the recursive `realize()` chain.

3. **Batch mmap commits** — Instead of 2 separate `AppendLog::write()` calls (data + index) per component, accumulate all writes and commit once per tick.


---

# elodin-db Profiling Report

**Generated:** 21:29:45 UTC

## Configuration

| Parameter | Value |
|---|---|
| Scenario | customer (batch) |
| Components | 400 |
| Frequency | 250 Hz |
| Duration | 10 s |
| Mode | batch |
| Total ticks recorded | 8192 |
| Total `sink_table` calls | 9951 |

## Tick Latency Distribution

| Percentile | Latency |
|---|---|
| p50 | 178.8 µs |
| p75 | 202.8 µs |
| p90 | 227.6 µs |
| p95 | 244.7 µs |
| p99 | 296.5 µs |
| max | 1813.0 µs |
| mean | 180.1 µs |

## Phase Breakdown (cumulative)

| Phase | Total Time | % of sink_table | Calls | Mean/call | Source |
|---|---|---|---|---|---|
| RwLock acquire | 15.6 ms | 0.9% | 9951 | 1565 ns | `lib.rs` `db.with_state()` |
| VTable resolve | 1762.7 ms | 98.9% | 146 | 12073019 ns | `vtable.rs` `realize_fields()` |
| apply_value total | 1611.1 ms | 90.4% | 995100 | 1619 ns | `lib.rs` `DBSink::apply_value()` |
|   push_buf | 582.1 ms | 32.6% | 995100 | 585 ns | `time_series.rs` `push_buf()` |
|     data write | 155.3 ms | 8.7% | 995100 | 156 ns | `append_log.rs` `data.write()` |
|     index write | 58.6 ms | 3.3% | 995100 | 59 ns | `append_log.rs` `index.write()` |
|     timestamp check | 43.8 ms | 2.5% | 995100 | 44 ns | `time_series.rs` last_ts read |
|   wake_all (data) | 54.1 ms | 3.0% | 995100 | 54 ns | `time_series.rs` `data_waker` |
|   wake_all (last_upd) | 443.9 ms | 24.9% | 995100 | 446 ns | `lib.rs` `update_max()` |
|   wake_all (earliest) | 83.1 ms | 4.7% | 995100 | 84 ns | `lib.rs` `update_min()` |
|   HashMap lookup | 64.6 ms | 3.6% | 995100 | 65 ns | `lib.rs` `components.get()` |

## Identified Bottlenecks

### #1 — VTable realize chain (1762.7 ms, 98.9% of total)

146 field resolutions (0/tick). Each field triggers ~7 recursive `realize()` calls. The VTable structure is static — results could be cached after first resolve.

### #2 — wake_all() cascade (581.1 ms, 32.6% of total)

2985300 total calls (300/tick). `update_min` rarely changes the value but calls `wake_all()` unconditionally. `update_max` is called 100× per tick with the same timestamp — only the first matters.

### #3 — push_buf Mutex churn (214.0 ms, 12.0% of total)

2 Mutex acquisitions per component per tick = 1990200 lock/unlock pairs. Uncontested in single-client batch mode but still ~20 ns each.

### #4 — HashMap lookups (64.6 ms, 3.6% of total)

995100 lookups. With 400 components the table fits in L2 cache, but pointer chasing to heap-allocated Components causes cache misses.

## Outlier Detection

| Metric | Value |
|---|---|
| Worst single `push_buf` | 449.0 µs |
| Worst single `apply_value` | 450.8 µs |
| Tick p99/p50 ratio | 1.7× |

> **Warning:** A single `push_buf` call took more than half the mean tick time. This is likely a **mmap page fault** on first write to a new 4 KB page.

## Optimization Recommendations

1. **Debounce `wake_all()` in `apply_value`** — Currently 33% of tick time. Call `update_max`/`update_min` once after the entire batch instead of per-component. Skip `wake_all()` when the atomic value didn't actually change.

2. **Cache VTable resolution** — Currently 99% of tick time. Pre-compile the field→component mapping on first `apply()` and reuse it on subsequent ticks, bypassing the recursive `realize()` chain.

3. **Batch mmap commits** — Instead of 2 separate `AppendLog::write()` calls (data + index) per component, accumulate all writes and commit once per tick.


---

# elodin-db Profiling Report

**Generated:** 21:29:56 UTC

## Configuration

| Parameter | Value |
|---|---|
| Scenario | high-freq |
| Components | 50 |
| Frequency | 1000 Hz |
| Duration | 10 s |
| Mode | batch |
| Total ticks recorded | 8192 |
| Total `sink_table` calls | 39179 |

## Tick Latency Distribution

| Percentile | Latency |
|---|---|
| p50 | 15.4 µs |
| p75 | 18.8 µs |
| p90 | 22.4 µs |
| p95 | 25.5 µs |
| p99 | 37.0 µs |
| max | 295.7 µs |
| mean | 16.2 µs |

## Phase Breakdown (cumulative)

| Phase | Total Time | % of sink_table | Calls | Mean/call | Source |
|---|---|---|---|---|---|
| RwLock acquire | 3.6 ms | 0.5% | 39179 | 91 ns | `lib.rs` `db.with_state()` |
| VTable resolve | 646.3 ms | 97.3% | 940 | 687502 ns | `vtable.rs` `realize_fields()` |
| apply_value total | 565.5 ms | 85.1% | 489737 | 1155 ns | `lib.rs` `DBSink::apply_value()` |
|   push_buf | 254.6 ms | 38.3% | 489737 | 520 ns | `time_series.rs` `push_buf()` |
|     data write | 48.5 ms | 7.3% | 489737 | 99 ns | `append_log.rs` `data.write()` |
|     index write | 29.2 ms | 4.4% | 489737 | 60 ns | `append_log.rs` `index.write()` |
|     timestamp check | 16.5 ms | 2.5% | 489737 | 34 ns | `time_series.rs` last_ts read |
|   wake_all (data) | 24.2 ms | 3.6% | 489737 | 50 ns | `time_series.rs` `data_waker` |
|   wake_all (last_upd) | 43.0 ms | 6.5% | 489737 | 88 ns | `lib.rs` `update_max()` |
|   wake_all (earliest) | 42.5 ms | 6.4% | 489737 | 87 ns | `lib.rs` `update_min()` |
|   HashMap lookup | 28.7 ms | 4.3% | 489737 | 59 ns | `lib.rs` `components.get()` |

## Identified Bottlenecks

### #1 — VTable realize chain (646.3 ms, 97.3% of total)

940 field resolutions (0/tick). Each field triggers ~7 recursive `realize()` calls. The VTable structure is static — results could be cached after first resolve.

### #2 — wake_all() cascade (109.8 ms, 16.5% of total)

1469211 total calls (37/tick). `update_min` rarely changes the value but calls `wake_all()` unconditionally. `update_max` is called 12× per tick with the same timestamp — only the first matters.

### #3 — push_buf Mutex churn (77.7 ms, 11.7% of total)

2 Mutex acquisitions per component per tick = 979474 lock/unlock pairs. Uncontested in single-client batch mode but still ~20 ns each.

### #4 — HashMap lookups (28.7 ms, 4.3% of total)

489737 lookups. With 50 components the table fits in L2 cache, but pointer chasing to heap-allocated Components causes cache misses.

## Outlier Detection

| Metric | Value |
|---|---|
| Worst single `push_buf` | 66.4 µs |
| Worst single `apply_value` | 70.9 µs |
| Tick p99/p50 ratio | 2.4× |

> **Warning:** A single `push_buf` call took more than half the mean tick time. This is likely a **mmap page fault** on first write to a new 4 KB page.

## Optimization Recommendations

1. **Debounce `wake_all()` in `apply_value`** — Currently 17% of tick time. Call `update_max`/`update_min` once after the entire batch instead of per-component. Skip `wake_all()` when the atomic value didn't actually change.

2. **Cache VTable resolution** — Currently 97% of tick time. Pre-compile the field→component mapping on first `apply()` and reuse it on subsequent ticks, bypassing the recursive `realize()` chain.

3. **Batch mmap commits** — Instead of 2 separate `AppendLog::write()` calls (data + index) per component, accumulate all writes and commit once per tick.


---

# elodin-db Profiling Report

**Generated:** 21:30:06 UTC

## Configuration

| Parameter | Value |
|---|---|
| Scenario | high-fanout |
| Components | 1000 |
| Frequency | 100 Hz |
| Duration | 10 s |
| Mode | batch |
| Total ticks recorded | 3988 |
| Total `sink_table` calls | 3988 |

## Tick Latency Distribution

| Percentile | Latency |
|---|---|
| p50 | 461.3 µs |
| p75 | 551.9 µs |
| p90 | 634.2 µs |
| p95 | 681.7 µs |
| p99 | 793.3 µs |
| max | 82752.0 µs |
| mean | 515.3 µs |

## Phase Breakdown (cumulative)

| Phase | Total Time | % of sink_table | Calls | Mean/call | Source |
|---|---|---|---|---|---|
| RwLock acquire | 182.0 ms | 8.9% | 3988 | 45641 ns | `lib.rs` `db.with_state()` |
| VTable resolve | 1870.6 ms | 91.0% | 114 | 16408748 ns | `vtable.rs` `realize_fields()` |
| apply_value total | 1691.3 ms | 82.3% | 997000 | 1696 ns | `lib.rs` `DBSink::apply_value()` |
|   push_buf | 759.0 ms | 36.9% | 997000 | 761 ns | `time_series.rs` `push_buf()` |
|     data write | 232.4 ms | 11.3% | 997000 | 233 ns | `append_log.rs` `data.write()` |
|     index write | 60.1 ms | 2.9% | 997000 | 60 ns | `append_log.rs` `index.write()` |
|     timestamp check | 51.4 ms | 2.5% | 997000 | 52 ns | `time_series.rs` last_ts read |
|   wake_all (data) | 64.0 ms | 3.1% | 997000 | 64 ns | `time_series.rs` `data_waker` |
|   wake_all (last_upd) | 117.9 ms | 5.7% | 997000 | 118 ns | `lib.rs` `update_max()` |
|   wake_all (earliest) | 102.6 ms | 5.0% | 997000 | 103 ns | `lib.rs` `update_min()` |
|   HashMap lookup | 77.9 ms | 3.8% | 997000 | 78 ns | `lib.rs` `components.get()` |

## Identified Bottlenecks

### #1 — VTable realize chain (1870.6 ms, 91.0% of total)

114 field resolutions (0/tick). Each field triggers ~7 recursive `realize()` calls. The VTable structure is static — results could be cached after first resolve.

### #2 — push_buf Mutex churn (292.4 ms, 14.2% of total)

2 Mutex acquisitions per component per tick = 1994000 lock/unlock pairs. Uncontested in single-client batch mode but still ~20 ns each.

### #3 — wake_all() cascade (284.5 ms, 13.8% of total)

2991000 total calls (750/tick). `update_min` rarely changes the value but calls `wake_all()` unconditionally. `update_max` is called 250× per tick with the same timestamp — only the first matters.

### #4 — HashMap lookups (77.9 ms, 3.8% of total)

997000 lookups. With 1000 components the table fits in L2 cache, but pointer chasing to heap-allocated Components causes cache misses.

## Outlier Detection

| Metric | Value |
|---|---|
| Worst single `push_buf` | 399.0 µs |
| Worst single `apply_value` | 400.4 µs |
| Tick p99/p50 ratio | 1.7× |

> **Warning:** A single `push_buf` call took more than half the mean tick time. This is likely a **mmap page fault** on first write to a new 4 KB page.

## Optimization Recommendations

2. **Cache VTable resolution** — Currently 91% of tick time. Pre-compile the field→component mapping on first `apply()` and reuse it on subsequent ticks, bypassing the recursive `realize()` chain.

3. **Batch mmap commits** — Instead of 2 separate `AppendLog::write()` calls (data + index) per component, accumulate all writes and commit once per tick.


---

# elodin-db Profiling Report

**Generated:** 21:30:16 UTC

## Configuration

| Parameter | Value |
|---|---|
| Scenario | stress |
| Components | 400 |
| Frequency | 1000 Hz |
| Duration | 10 s |
| Mode | batch |
| Total ticks recorded | 8192 |
| Total `sink_table` calls | 39054 |

## Tick Latency Distribution

| Percentile | Latency |
|---|---|
| p50 | 185.0 µs |
| p75 | 203.0 µs |
| p90 | 217.5 µs |
| p95 | 228.5 µs |
| p99 | 260.8 µs |
| max | 1489.9 µs |
| mean | 185.5 µs |

## Phase Breakdown (cumulative)

| Phase | Total Time | % of sink_table | Calls | Mean/call | Source |
|---|---|---|---|---|---|
| RwLock acquire | 20.7 ms | 0.3% | 39054 | 530 ns | `lib.rs` `db.with_state()` |
| VTable resolve | 7126.4 ms | 99.5% | 748 | 9527300 ns | `vtable.rs` `realize_fields()` |
| apply_value total | 6449.1 ms | 90.0% | 3905400 | 1651 ns | `lib.rs` `DBSink::apply_value()` |
|   push_buf | 2267.4 ms | 31.7% | 3905400 | 581 ns | `time_series.rs` `push_buf()` |
|     data write | 418.8 ms | 5.8% | 3905400 | 107 ns | `append_log.rs` `data.write()` |
|     index write | 213.5 ms | 3.0% | 3905400 | 55 ns | `append_log.rs` `index.write()` |
|     timestamp check | 141.7 ms | 2.0% | 3905400 | 36 ns | `time_series.rs` last_ts read |
|   wake_all (data) | 180.3 ms | 2.5% | 3905400 | 46 ns | `time_series.rs` `data_waker` |
|   wake_all (last_upd) | 1531.7 ms | 21.4% | 3905400 | 392 ns | `lib.rs` `update_max()` |
|   wake_all (earliest) | 389.4 ms | 5.4% | 3905400 | 100 ns | `lib.rs` `update_min()` |
|   HashMap lookup | 215.2 ms | 3.0% | 3905400 | 55 ns | `lib.rs` `components.get()` |

## Identified Bottlenecks

### #1 — VTable realize chain (7126.4 ms, 99.5% of total)

748 field resolutions (0/tick). Each field triggers ~7 recursive `realize()` calls. The VTable structure is static — results could be cached after first resolve.

### #2 — wake_all() cascade (2101.4 ms, 29.3% of total)

11716200 total calls (300/tick). `update_min` rarely changes the value but calls `wake_all()` unconditionally. `update_max` is called 100× per tick with the same timestamp — only the first matters.

### #3 — push_buf Mutex churn (632.3 ms, 8.8% of total)

2 Mutex acquisitions per component per tick = 7810800 lock/unlock pairs. Uncontested in single-client batch mode but still ~20 ns each.

### #4 — HashMap lookups (215.2 ms, 3.0% of total)

3905400 lookups. With 400 components the table fits in L2 cache, but pointer chasing to heap-allocated Components causes cache misses.

## Outlier Detection

| Metric | Value |
|---|---|
| Worst single `push_buf` | 392.0 µs |
| Worst single `apply_value` | 393.6 µs |
| Tick p99/p50 ratio | 1.4× |

> **Warning:** A single `push_buf` call took more than half the mean tick time. This is likely a **mmap page fault** on first write to a new 4 KB page.

## Optimization Recommendations

1. **Debounce `wake_all()` in `apply_value`** — Currently 29% of tick time. Call `update_max`/`update_min` once after the entire batch instead of per-component. Skip `wake_all()` when the atomic value didn't actually change.

2. **Cache VTable resolution** — Currently 99% of tick time. Pre-compile the field→component mapping on first `apply()` and reuse it on subsequent ticks, bypassing the recursive `realize()` chain.


---

