# elodin-db Profiling Report (all scenarios)

---

# elodin-db Profiling Report

**Generated:** 21:04:47 UTC

## Configuration

| Parameter | Value |
|---|---|
| Scenario | customer (per-component) |
| Components | 400 |
| Frequency | 250 Hz |
| Duration | 10 s |
| Mode | per-component |
| Total ticks recorded | 8192 |
| Total `sink_table` calls | 194000 |

## Tick Latency Distribution

| Percentile | Latency |
|---|---|
| p50 | 1.7 µs |
| p75 | 2.5 µs |
| p90 | 3.1 µs |
| p95 | 3.5 µs |
| p99 | 5.0 µs |
| max | 211.8 µs |
| mean | 2.1 µs |

## Phase Breakdown (cumulative)

| Phase | Total Time | % of sink_table | Calls | Mean/call | Source |
|---|---|---|---|---|---|
| RwLock acquire | 208.3 ms | 39.8% | 194000 | 1074 ns | `lib.rs` `db.with_state()` |
| VTable resolve | 270.5 ms | 51.8% | 1156 | 234020 ns | `vtable.rs` `realize_fields()` |
| apply_value total | 242.2 ms | 46.3% | 194000 | 1248 ns | `lib.rs` `DBSink::apply_value()` |
|   push_buf | 106.9 ms | 20.5% | 194000 | 551 ns | `time_series.rs` `push_buf()` |
|     data write | 32.4 ms | 6.2% | 194000 | 167 ns | `append_log.rs` `data.write()` |
|     index write | 11.3 ms | 2.2% | 194000 | 58 ns | `append_log.rs` `index.write()` |
|     timestamp check | 7.5 ms | 1.4% | 194000 | 39 ns | `time_series.rs` last_ts read |
|   wake_all (data) | 9.7 ms | 1.9% | 194000 | 50 ns | `time_series.rs` `data_waker` |
|   wake_all (last_upd) | 25.2 ms | 4.8% | 194000 | 130 ns | `lib.rs` `update_max()` |
|   wake_all (earliest) | 9.7 ms | 1.9% | 194000 | 50 ns | `lib.rs` `update_min()` |
|   HashMap lookup | 10.4 ms | 2.0% | 194000 | 54 ns | `lib.rs` `components.get()` |

## Identified Bottlenecks

### #1 — VTable realize chain (270.5 ms, 51.8% of total)

1156 field resolutions (0/tick). Each field triggers ~7 recursive `realize()` calls. The VTable structure is static — results could be cached after first resolve.

### #2 — `wake_all` latency (data + metadata) (44.6 ms, 8.5% of total)

582000 profiled `wake_all` calls (~3/`sink_table`). Split: 22% of combined row time in `push_buf` `data_waker`, 78% in `update_max`/`update_min` rows (those rows include the atomic `fetch_*`, not only `wake_all`). Metadata waiters are woken only when `fetch_max`/`fetch_min` changes the stored value.

### #3 — push_buf Mutex churn (43.6 ms, 8.4% of total)

2 Mutex acquisitions per component per tick = 388000 lock/unlock pairs. Uncontested in single-client batch mode but still ~20 ns each.

### #4 — HashMap lookups (10.4 ms, 2.0% of total)

194000 lookups. With 400 components the table fits in L2 cache, but pointer chasing to heap-allocated Components causes cache misses.

## Outlier Detection

| Metric | Value |
|---|---|
| Worst single `push_buf` | 82.0 µs |
| Worst single `apply_value` | 210.9 µs |
| Tick p99/p50 ratio | 2.9× |

> **Warning:** A single `push_buf` call took more than half the mean tick time. This is likely a **mmap page fault** on first write to a new 4 KB page.

## Optimization Recommendations

1. **Cache VTable resolution** — Currently 52% of `sink_table` time. Pre-compile the field→component mapping on first `apply()` and reuse it on subsequent ticks, bypassing the recursive `realize()` chain.


---

# elodin-db Profiling Report

**Generated:** 21:04:57 UTC

## Configuration

| Parameter | Value |
|---|---|
| Scenario | customer (batch) |
| Components | 400 |
| Frequency | 250 Hz |
| Duration | 10 s |
| Mode | batch |
| Total ticks recorded | 788 |
| Total `sink_table` calls | 788 |

## Tick Latency Distribution

| Percentile | Latency |
|---|---|
| p50 | 246.9 µs |
| p75 | 306.0 µs |
| p90 | 370.6 µs |
| p95 | 414.3 µs |
| p99 | 865.9 µs |
| max | 158176.4 µs |
| mean | 661.8 µs |

## Phase Breakdown (cumulative)

| Phase | Total Time | % of sink_table | Calls | Mean/call | Source |
|---|---|---|---|---|---|
| RwLock acquire | 305.8 ms | 58.6% | 788 | 388069 ns | `lib.rs` `db.with_state()` |
| VTable resolve | 215.1 ms | 41.3% | 1 | 215132827 ns | `vtable.rs` `realize_fields()` |
| apply_value total | 199.8 ms | 38.3% | 78800 | 2536 ns | `lib.rs` `DBSink::apply_value()` |
|   push_buf | 98.2 ms | 18.8% | 78800 | 1246 ns | `time_series.rs` `push_buf()` |
|     data write | 34.7 ms | 6.7% | 78800 | 441 ns | `append_log.rs` `data.write()` |
|     index write | 6.7 ms | 1.3% | 78800 | 85 ns | `append_log.rs` `index.write()` |
|     timestamp check | 6.8 ms | 1.3% | 78800 | 86 ns | `time_series.rs` last_ts read |
|   wake_all (data) | 6.4 ms | 1.2% | 78800 | 81 ns | `time_series.rs` `data_waker` |
|   wake_all (last_upd) | 7.4 ms | 1.4% | 78800 | 94 ns | `lib.rs` `update_max()` |
|   wake_all (earliest) | 7.1 ms | 1.4% | 78800 | 90 ns | `lib.rs` `update_min()` |
|   HashMap lookup | 8.5 ms | 1.6% | 78800 | 107 ns | `lib.rs` `components.get()` |

## Identified Bottlenecks

### #1 — VTable realize chain (215.1 ms, 41.3% of total)

1 field resolutions (0/tick). Each field triggers ~7 recursive `realize()` calls. The VTable structure is static — results could be cached after first resolve.

### #2 — push_buf Mutex churn (41.5 ms, 8.0% of total)

2 Mutex acquisitions per component per tick = 157600 lock/unlock pairs. Uncontested in single-client batch mode but still ~20 ns each.

### #3 — `wake_all` latency (data + metadata) (20.9 ms, 4.0% of total)

236400 profiled `wake_all` calls (~300/`sink_table`). Split: 31% of combined row time in `push_buf` `data_waker`, 69% in `update_max`/`update_min` rows (those rows include the atomic `fetch_*`, not only `wake_all`). Metadata waiters are woken only when `fetch_max`/`fetch_min` changes the stored value.

### #4 — HashMap lookups (8.5 ms, 1.6% of total)

78800 lookups. With 400 components the table fits in L2 cache, but pointer chasing to heap-allocated Components causes cache misses.

## Outlier Detection

| Metric | Value |
|---|---|
| Worst single `push_buf` | 1486.0 µs |
| Worst single `apply_value` | 1676.2 µs |
| Tick p99/p50 ratio | 3.5× |

> **Warning:** A single `push_buf` call took more than half the mean tick time. This is likely a **mmap page fault** on first write to a new 4 KB page.

## Optimization Recommendations

1. **Cache VTable resolution** — Currently 41% of `sink_table` time. Pre-compile the field→component mapping on first `apply()` and reuse it on subsequent ticks, bypassing the recursive `realize()` chain.


---

# elodin-db Profiling Report

**Generated:** 21:05:08 UTC

## Configuration

| Parameter | Value |
|---|---|
| Scenario | high-freq |
| Components | 50 |
| Frequency | 1000 Hz |
| Duration | 10 s |
| Mode | batch |
| Total ticks recorded | 936 |
| Total `sink_table` calls | 936 |

## Tick Latency Distribution

| Percentile | Latency |
|---|---|
| p50 | 29.2 µs |
| p75 | 35.7 µs |
| p90 | 46.6 µs |
| p95 | 54.7 µs |
| p99 | 124.0 µs |
| max | 1428.0 µs |
| mean | 35.7 µs |

## Phase Breakdown (cumulative)

| Phase | Total Time | % of sink_table | Calls | Mean/call | Source |
|---|---|---|---|---|---|
| RwLock acquire | 0.2 ms | 0.7% | 936 | 263 ns | `lib.rs` `db.with_state()` |
| VTable resolve | 32.4 ms | 97.0% | 2 | 16181165 ns | `vtable.rs` `realize_fields()` |
| apply_value total | 29.7 ms | 89.0% | 11700 | 2539 ns | `lib.rs` `DBSink::apply_value()` |
|   push_buf | 15.2 ms | 45.6% | 11700 | 1300 ns | `time_series.rs` `push_buf()` |
|     data write | 6.8 ms | 20.4% | 11700 | 582 ns | `append_log.rs` `data.write()` |
|     index write | 1.3 ms | 3.8% | 11700 | 108 ns | `append_log.rs` `index.write()` |
|     timestamp check | 2.3 ms | 6.9% | 11700 | 196 ns | `time_series.rs` last_ts read |
|   wake_all (data) | 1.5 ms | 4.4% | 11700 | 125 ns | `time_series.rs` `data_waker` |
|   wake_all (last_upd) | 0.8 ms | 2.3% | 11700 | 66 ns | `lib.rs` `update_max()` |
|   wake_all (earliest) | 0.6 ms | 1.9% | 11700 | 55 ns | `lib.rs` `update_min()` |
|   HashMap lookup | 3.7 ms | 11.2% | 11700 | 319 ns | `lib.rs` `components.get()` |

## Identified Bottlenecks

### #1 — VTable realize chain (32.4 ms, 97.0% of total)

2 field resolutions (0/tick). Each field triggers ~7 recursive `realize()` calls. The VTable structure is static — results could be cached after first resolve.

### #2 — push_buf Mutex churn (8.1 ms, 24.2% of total)

2 Mutex acquisitions per component per tick = 23400 lock/unlock pairs. Uncontested in single-client batch mode but still ~20 ns each.

### #3 — HashMap lookups (3.7 ms, 11.2% of total)

11700 lookups. With 50 components the table fits in L2 cache, but pointer chasing to heap-allocated Components causes cache misses.

### #4 — `wake_all` latency (data + metadata) (2.9 ms, 8.6% of total)

35100 profiled `wake_all` calls (~37/`sink_table`). Split: 51% of combined row time in `push_buf` `data_waker`, 49% in `update_max`/`update_min` rows (those rows include the atomic `fetch_*`, not only `wake_all`). Metadata waiters are woken only when `fetch_max`/`fetch_min` changes the stored value.

## Outlier Detection

| Metric | Value |
|---|---|
| Worst single `push_buf` | 1312.2 µs |
| Worst single `apply_value` | 1393.2 µs |
| Tick p99/p50 ratio | 4.2× |

> **Warning:** A single `push_buf` call took more than half the mean tick time. This is likely a **mmap page fault** on first write to a new 4 KB page.

## Optimization Recommendations

1. **Cache VTable resolution** — Currently 97% of `sink_table` time. Pre-compile the field→component mapping on first `apply()` and reuse it on subsequent ticks, bypassing the recursive `realize()` chain.

2. **Batch mmap commits** — Instead of 2 separate `AppendLog::write()` calls (data + index) per component, accumulate all writes and commit once per tick.


---

# elodin-db Profiling Report

**Generated:** 21:05:19 UTC

## Configuration

| Parameter | Value |
|---|---|
| Scenario | high-fanout |
| Components | 1000 |
| Frequency | 100 Hz |
| Duration | 10 s |
| Mode | batch |
| Total ticks recorded | 884 |
| Total `sink_table` calls | 884 |

## Tick Latency Distribution

| Percentile | Latency |
|---|---|
| p50 | 668.1 µs |
| p75 | 878.4 µs |
| p90 | 1253.8 µs |
| p95 | 1948.8 µs |
| p99 | 5726.7 µs |
| max | 467638.2 µs |
| mean | 1854.2 µs |

## Phase Breakdown (cumulative)

| Phase | Total Time | % of sink_table | Calls | Mean/call | Source |
|---|---|---|---|---|---|
| RwLock acquire | 869.4 ms | 53.0% | 884 | 983514 ns | `lib.rs` `db.with_state()` |
| VTable resolve | 768.9 ms | 46.9% | 0 | 0 ns | `vtable.rs` `realize_fields()` |
| apply_value total | 722.9 ms | 44.1% | 221000 | 3271 ns | `lib.rs` `DBSink::apply_value()` |
|   push_buf | 371.6 ms | 22.7% | 221000 | 1682 ns | `time_series.rs` `push_buf()` |
|     data write | 130.5 ms | 8.0% | 221000 | 591 ns | `append_log.rs` `data.write()` |
|     index write | 27.4 ms | 1.7% | 221000 | 124 ns | `append_log.rs` `index.write()` |
|     timestamp check | 26.8 ms | 1.6% | 221000 | 121 ns | `time_series.rs` last_ts read |
|   wake_all (data) | 27.9 ms | 1.7% | 221000 | 126 ns | `time_series.rs` `data_waker` |
|   wake_all (last_upd) | 21.4 ms | 1.3% | 221000 | 97 ns | `lib.rs` `update_max()` |
|   wake_all (earliest) | 29.0 ms | 1.8% | 221000 | 131 ns | `lib.rs` `update_min()` |
|   HashMap lookup | 35.9 ms | 2.2% | 221000 | 163 ns | `lib.rs` `components.get()` |

## Identified Bottlenecks

### #1 — VTable realize chain (768.9 ms, 46.9% of total)

0 field resolutions (0/tick). Each field triggers ~7 recursive `realize()` calls. The VTable structure is static — results could be cached after first resolve.

### #2 — push_buf Mutex churn (157.9 ms, 9.6% of total)

2 Mutex acquisitions per component per tick = 442000 lock/unlock pairs. Uncontested in single-client batch mode but still ~20 ns each.

### #3 — `wake_all` latency (data + metadata) (78.3 ms, 4.8% of total)

663000 profiled `wake_all` calls (~750/`sink_table`). Split: 36% of combined row time in `push_buf` `data_waker`, 64% in `update_max`/`update_min` rows (those rows include the atomic `fetch_*`, not only `wake_all`). Metadata waiters are woken only when `fetch_max`/`fetch_min` changes the stored value.

### #4 — HashMap lookups (35.9 ms, 2.2% of total)

221000 lookups. With 1000 components the table fits in L2 cache, but pointer chasing to heap-allocated Components causes cache misses.

## Outlier Detection

| Metric | Value |
|---|---|
| Worst single `push_buf` | 12395.2 µs |
| Worst single `apply_value` | 12396.5 µs |
| Tick p99/p50 ratio | 8.6× |

> **Warning:** A single `push_buf` call took more than half the mean tick time. This is likely a **mmap page fault** on first write to a new 4 KB page.

## Optimization Recommendations

1. **Cache VTable resolution** — Currently 47% of `sink_table` time. Pre-compile the field→component mapping on first `apply()` and reuse it on subsequent ticks, bypassing the recursive `realize()` chain.


---

# elodin-db Profiling Report

**Generated:** 21:05:30 UTC

## Configuration

| Parameter | Value |
|---|---|
| Scenario | stress |
| Components | 400 |
| Frequency | 1000 Hz |
| Duration | 10 s |
| Mode | batch |
| Total ticks recorded | 508 |
| Total `sink_table` calls | 508 |

## Tick Latency Distribution

| Percentile | Latency |
|---|---|
| p50 | 249.1 µs |
| p75 | 314.5 µs |
| p90 | 473.4 µs |
| p95 | 869.8 µs |
| p99 | 6515.5 µs |
| max | 99865.4 µs |
| mean | 906.1 µs |

## Phase Breakdown (cumulative)

| Phase | Total Time | % of sink_table | Calls | Mean/call | Source |
|---|---|---|---|---|---|
| RwLock acquire | 248.2 ms | 53.9% | 508 | 488516 ns | `lib.rs` `db.with_state()` |
| VTable resolve | 211.5 ms | 45.9% | 0 | 0 ns | `vtable.rs` `realize_fields()` |
| apply_value total | 202.3 ms | 43.9% | 50800 | 3982 ns | `lib.rs` `DBSink::apply_value()` |
|   push_buf | 115.2 ms | 25.0% | 50800 | 2267 ns | `time_series.rs` `push_buf()` |
|     data write | 62.9 ms | 13.7% | 50800 | 1238 ns | `append_log.rs` `data.write()` |
|     index write | 9.5 ms | 2.1% | 50800 | 187 ns | `append_log.rs` `index.write()` |
|     timestamp check | 8.4 ms | 1.8% | 50800 | 165 ns | `time_series.rs` last_ts read |
|   wake_all (data) | 6.3 ms | 1.4% | 50800 | 124 ns | `time_series.rs` `data_waker` |
|   wake_all (last_upd) | 4.4 ms | 1.0% | 50800 | 86 ns | `lib.rs` `update_max()` |
|   wake_all (earliest) | 3.6 ms | 0.8% | 50800 | 72 ns | `lib.rs` `update_min()` |
|   HashMap lookup | 8.3 ms | 1.8% | 50800 | 164 ns | `lib.rs` `components.get()` |

## Identified Bottlenecks

### #1 — VTable realize chain (211.5 ms, 45.9% of total)

0 field resolutions (0/tick). Each field triggers ~7 recursive `realize()` calls. The VTable structure is static — results could be cached after first resolve.

### #2 — push_buf Mutex churn (72.4 ms, 15.7% of total)

2 Mutex acquisitions per component per tick = 101600 lock/unlock pairs. Uncontested in single-client batch mode but still ~20 ns each.

### #3 — `wake_all` latency (data + metadata) (14.3 ms, 3.1% of total)

152400 profiled `wake_all` calls (~300/`sink_table`). Split: 44% of combined row time in `push_buf` `data_waker`, 56% in `update_max`/`update_min` rows (those rows include the atomic `fetch_*`, not only `wake_all`). Metadata waiters are woken only when `fetch_max`/`fetch_min` changes the stored value.

### #4 — HashMap lookups (8.3 ms, 1.8% of total)

50800 lookups. With 400 components the table fits in L2 cache, but pointer chasing to heap-allocated Components causes cache misses.

## Outlier Detection

| Metric | Value |
|---|---|
| Worst single `push_buf` | 14371.4 µs |
| Worst single `apply_value` | 14373.2 µs |
| Tick p99/p50 ratio | 26.2× |

> **Warning:** A single `push_buf` call took more than half the mean tick time. This is likely a **mmap page fault** on first write to a new 4 KB page.

## Optimization Recommendations

1. **Cache VTable resolution** — Currently 46% of `sink_table` time. Pre-compile the field→component mapping on first `apply()` and reuse it on subsequent ticks, bypassing the recursive `realize()` chain.

2. **Batch mmap commits** — Instead of 2 separate `AppendLog::write()` calls (data + index) per component, accumulate all writes and commit once per tick.


---

