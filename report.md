# elodin-db Profiling Report (all scenarios)

---

# elodin-db Profiling Report

**Generated:** 21:24:05 UTC

## Configuration

| Parameter | Value |
|---|---|
| Scenario | customer (per-component) |
| Components | 400 |
| Frequency | 250 Hz |
| Duration | 10 s |
| Mode | per-component |
| Total ticks recorded | 8192 |
| Total `sink_table` calls | 203300 |

## Tick Latency Distribution

| Percentile | Latency |
|---|---|
| p50 | 1.3 µs |
| p75 | 1.7 µs |
| p90 | 2.2 µs |
| p95 | 2.7 µs |
| p99 | 4.2 µs |
| max | 87.0 µs |
| mean | 1.5 µs |

## Phase Breakdown (cumulative)

| Phase | Total Time | % of sink_table | Calls | Mean/call | Source |
|---|---|---|---|---|---|
| RwLock acquire | 226.2 ms | 46.3% | 203300 | 1113 ns | `lib.rs` `db.with_state()` |
| VTable resolve | 191.1 ms | 39.1% | 1004 | 190289 ns | `vtable.rs` `realize_fields()` |
| apply_value total | 161.0 ms | 33.0% | 203300 | 792 ns | `lib.rs` `DBSink::apply_value()` |
|   push_buf | 100.1 ms | 20.5% | 203300 | 492 ns | `time_series.rs` `push_buf()` |
|     data write | 26.7 ms | 5.5% | 203300 | 131 ns | `append_log.rs` `data.write()` |
|     index write | 11.7 ms | 2.4% | 203300 | 58 ns | `append_log.rs` `index.write()` |
|     timestamp check | 7.0 ms | 1.4% | 203300 | 35 ns | `time_series.rs` last_ts read |
|   wake_all (data) | 12.8 ms | 2.6% | 203300 | 63 ns | `time_series.rs` `data_waker` |
|   wake_all (last_upd) | 0.0 ms | 0.0% | 203300 | 0 ns | `lib.rs` `update_max()` |
|   wake_all (earliest) | 0.0 ms | 0.0% | 203300 | 0 ns | `lib.rs` `update_min()` |
|   HashMap lookup | 10.2 ms | 2.1% | 203300 | 50 ns | `lib.rs` `components.get()` |

## Identified Bottlenecks

### #1 — VTable realize chain (191.1 ms, 39.1% of total)

1004 field resolutions (0/tick). Each field triggers ~7 recursive `realize()` calls. The VTable structure is static — results could be cached after first resolve.

### #2 — push_buf Mutex churn (38.4 ms, 7.9% of total)

2 Mutex acquisitions per component per tick = 406600 lock/unlock pairs. Uncontested in single-client batch mode but still ~20 ns each.

### #3 — `wake_all` latency (data + metadata) (12.8 ms, 2.6% of total)

203300 profiled `wake_all` calls (~1/`sink_table`). Split: 100% of combined row time in `push_buf` `data_waker`, 0% in `update_max`/`update_min` rows (those rows include the atomic `fetch_*`, not only `wake_all`). Metadata waiters are woken only when `fetch_max`/`fetch_min` changes the stored value.

### #4 — HashMap lookups (10.2 ms, 2.1% of total)

203300 lookups. With 400 components the table fits in L2 cache, but pointer chasing to heap-allocated Components causes cache misses.

## Outlier Detection

| Metric | Value |
|---|---|
| Worst single `push_buf` | 119.0 µs |
| Worst single `apply_value` | 120.2 µs |
| Tick p99/p50 ratio | 3.3× |

> **Warning:** A single `push_buf` call took more than half the mean tick time. This is likely a **mmap page fault** on first write to a new 4 KB page.

## Optimization Recommendations

1. **Cache VTable resolution** — Currently 39% of `sink_table` time. Pre-compile the field→component mapping on first `apply()` and reuse it on subsequent ticks, bypassing the recursive `realize()` chain.


---

# elodin-db Profiling Report

**Generated:** 21:24:16 UTC

## Configuration

| Parameter | Value |
|---|---|
| Scenario | customer (batch) |
| Components | 400 |
| Frequency | 250 Hz |
| Duration | 10 s |
| Mode | batch |
| Total ticks recorded | 884 |
| Total `sink_table` calls | 884 |

## Tick Latency Distribution

| Percentile | Latency |
|---|---|
| p50 | 161.0 µs |
| p75 | 217.5 µs |
| p90 | 261.1 µs |
| p95 | 282.3 µs |
| p99 | 399.3 µs |
| max | 161268.3 µs |
| mean | 563.6 µs |

## Phase Breakdown (cumulative)

| Phase | Total Time | % of sink_table | Calls | Mean/call | Source |
|---|---|---|---|---|---|
| RwLock acquire | 331.4 ms | 66.5% | 884 | 374859 ns | `lib.rs` `db.with_state()` |
| VTable resolve | 165.6 ms | 33.2% | 0 | 0 ns | `vtable.rs` `realize_fields()` |
| apply_value total | 149.2 ms | 30.0% | 88400 | 1688 ns | `lib.rs` `DBSink::apply_value()` |
|   push_buf | 97.2 ms | 19.5% | 88400 | 1099 ns | `time_series.rs` `push_buf()` |
|     data write | 33.3 ms | 6.7% | 88400 | 376 ns | `append_log.rs` `data.write()` |
|     index write | 7.3 ms | 1.5% | 88400 | 83 ns | `append_log.rs` `index.write()` |
|     timestamp check | 6.6 ms | 1.3% | 88400 | 75 ns | `time_series.rs` last_ts read |
|   wake_all (data) | 6.9 ms | 1.4% | 88400 | 78 ns | `time_series.rs` `data_waker` |
|   wake_all (last_upd) | 0.0 ms | 0.0% | 88400 | 0 ns | `lib.rs` `update_max()` |
|   wake_all (earliest) | 0.0 ms | 0.0% | 88400 | 0 ns | `lib.rs` `update_min()` |
|   HashMap lookup | 8.1 ms | 1.6% | 88400 | 92 ns | `lib.rs` `components.get()` |

## Identified Bottlenecks

### #1 — VTable realize chain (165.6 ms, 33.2% of total)

0 field resolutions (0/tick). Each field triggers ~7 recursive `realize()` calls. The VTable structure is static — results could be cached after first resolve.

### #2 — push_buf Mutex churn (40.6 ms, 8.2% of total)

2 Mutex acquisitions per component per tick = 176800 lock/unlock pairs. Uncontested in single-client batch mode but still ~20 ns each.

### #3 — HashMap lookups (8.1 ms, 1.6% of total)

88400 lookups. With 400 components the table fits in L2 cache, but pointer chasing to heap-allocated Components causes cache misses.

### #4 — `wake_all` latency (data + metadata) (6.9 ms, 1.4% of total)

88400 profiled `wake_all` calls (~100/`sink_table`). Split: 100% of combined row time in `push_buf` `data_waker`, 0% in `update_max`/`update_min` rows (those rows include the atomic `fetch_*`, not only `wake_all`). Metadata waiters are woken only when `fetch_max`/`fetch_min` changes the stored value.

## Outlier Detection

| Metric | Value |
|---|---|
| Worst single `push_buf` | 3842.0 µs |
| Worst single `apply_value` | 3845.7 µs |
| Tick p99/p50 ratio | 2.5× |

> **Warning:** A single `push_buf` call took more than half the mean tick time. This is likely a **mmap page fault** on first write to a new 4 KB page.

## Optimization Recommendations

1. **Cache VTable resolution** — Currently 33% of `sink_table` time. Pre-compile the field→component mapping on first `apply()` and reuse it on subsequent ticks, bypassing the recursive `realize()` chain.


---

# elodin-db Profiling Report

**Generated:** 21:24:27 UTC

## Configuration

| Parameter | Value |
|---|---|
| Scenario | high-freq |
| Components | 50 |
| Frequency | 1000 Hz |
| Duration | 10 s |
| Mode | batch |
| Total ticks recorded | 820 |
| Total `sink_table` calls | 820 |

## Tick Latency Distribution

| Percentile | Latency |
|---|---|
| p50 | 19.2 µs |
| p75 | 28.7 µs |
| p90 | 33.8 µs |
| p95 | 39.5 µs |
| p99 | 139.9 µs |
| max | 4448.6 µs |
| mean | 30.1 µs |

## Phase Breakdown (cumulative)

| Phase | Total Time | % of sink_table | Calls | Mean/call | Source |
|---|---|---|---|---|---|
| RwLock acquire | 0.2 ms | 0.7% | 820 | 218 ns | `lib.rs` `db.with_state()` |
| VTable resolve | 23.7 ms | 96.0% | 0 | 0 ns | `vtable.rs` `realize_fields()` |
| apply_value total | 21.2 ms | 85.8% | 10250 | 2067 ns | `lib.rs` `DBSink::apply_value()` |
|   push_buf | 14.5 ms | 58.6% | 10250 | 1411 ns | `time_series.rs` `push_buf()` |
|     data write | 8.7 ms | 35.2% | 10250 | 847 ns | `append_log.rs` `data.write()` |
|     index write | 1.3 ms | 5.1% | 10250 | 122 ns | `append_log.rs` `index.write()` |
|     timestamp check | 0.8 ms | 3.3% | 10250 | 80 ns | `time_series.rs` last_ts read |
|   wake_all (data) | 1.0 ms | 4.0% | 10250 | 97 ns | `time_series.rs` `data_waker` |
|   wake_all (last_upd) | 0.0 ms | 0.0% | 10250 | 0 ns | `lib.rs` `update_max()` |
|   wake_all (earliest) | 0.0 ms | 0.0% | 10250 | 0 ns | `lib.rs` `update_min()` |
|   HashMap lookup | 1.1 ms | 4.4% | 10250 | 106 ns | `lib.rs` `components.get()` |

## Identified Bottlenecks

### #1 — VTable realize chain (23.7 ms, 96.0% of total)

0 field resolutions (0/tick). Each field triggers ~7 recursive `realize()` calls. The VTable structure is static — results could be cached after first resolve.

### #2 — push_buf Mutex churn (9.9 ms, 40.3% of total)

2 Mutex acquisitions per component per tick = 20500 lock/unlock pairs. Uncontested in single-client batch mode but still ~20 ns each.

### #3 — HashMap lookups (1.1 ms, 4.4% of total)

10250 lookups. With 50 components the table fits in L2 cache, but pointer chasing to heap-allocated Components causes cache misses.

### #4 — `wake_all` latency (data + metadata) (1.0 ms, 4.0% of total)

10250 profiled `wake_all` calls (~12/`sink_table`). Split: 100% of combined row time in `push_buf` `data_waker`, 0% in `update_max`/`update_min` rows (those rows include the atomic `fetch_*`, not only `wake_all`). Metadata waiters are woken only when `fetch_max`/`fetch_min` changes the stored value.

## Outlier Detection

| Metric | Value |
|---|---|
| Worst single `push_buf` | 4264.3 µs |
| Worst single `apply_value` | 4268.7 µs |
| Tick p99/p50 ratio | 7.3× |

> **Warning:** A single `push_buf` call took more than half the mean tick time. This is likely a **mmap page fault** on first write to a new 4 KB page.

## Optimization Recommendations

1. **Cache VTable resolution** — Currently 96% of `sink_table` time. Pre-compile the field→component mapping on first `apply()` and reuse it on subsequent ticks, bypassing the recursive `realize()` chain.

2. **Batch mmap commits** — Instead of 2 separate `AppendLog::write()` calls (data + index) per component, accumulate all writes and commit once per tick.


---

# elodin-db Profiling Report

**Generated:** 21:24:38 UTC

## Configuration

| Parameter | Value |
|---|---|
| Scenario | high-fanout |
| Components | 1000 |
| Frequency | 100 Hz |
| Duration | 10 s |
| Mode | batch |
| Total ticks recorded | 772 |
| Total `sink_table` calls | 772 |

## Tick Latency Distribution

| Percentile | Latency |
|---|---|
| p50 | 440.8 µs |
| p75 | 600.8 µs |
| p90 | 694.9 µs |
| p95 | 778.7 µs |
| p99 | 2838.0 µs |
| max | 431122.7 µs |
| mean | 2311.2 µs |

## Phase Breakdown (cumulative)

| Phase | Total Time | % of sink_table | Calls | Mean/call | Source |
|---|---|---|---|---|---|
| RwLock acquire | 1377.3 ms | 77.2% | 772 | 1784019 ns | `lib.rs` `db.with_state()` |
| VTable resolve | 406.2 ms | 22.8% | 0 | 0 ns | `vtable.rs` `realize_fields()` |
| apply_value total | 367.6 ms | 20.6% | 193000 | 1904 ns | `lib.rs` `DBSink::apply_value()` |
|   push_buf | 225.6 ms | 12.6% | 193000 | 1169 ns | `time_series.rs` `push_buf()` |
|     data write | 72.7 ms | 4.1% | 193000 | 376 ns | `append_log.rs` `data.write()` |
|     index write | 13.1 ms | 0.7% | 193000 | 68 ns | `append_log.rs` `index.write()` |
|     timestamp check | 17.3 ms | 1.0% | 193000 | 89 ns | `time_series.rs` last_ts read |
|   wake_all (data) | 15.8 ms | 0.9% | 193000 | 82 ns | `time_series.rs` `data_waker` |
|   wake_all (last_upd) | 0.0 ms | 0.0% | 193000 | 0 ns | `lib.rs` `update_max()` |
|   wake_all (earliest) | 0.0 ms | 0.0% | 193000 | 0 ns | `lib.rs` `update_min()` |
|   HashMap lookup | 23.8 ms | 1.3% | 193000 | 123 ns | `lib.rs` `components.get()` |

## Identified Bottlenecks

### #1 — VTable realize chain (406.2 ms, 22.8% of total)

0 field resolutions (0/tick). Each field triggers ~7 recursive `realize()` calls. The VTable structure is static — results could be cached after first resolve.

### #2 — push_buf Mutex churn (85.8 ms, 4.8% of total)

2 Mutex acquisitions per component per tick = 386000 lock/unlock pairs. Uncontested in single-client batch mode but still ~20 ns each.

### #3 — HashMap lookups (23.8 ms, 1.3% of total)

193000 lookups. With 1000 components the table fits in L2 cache, but pointer chasing to heap-allocated Components causes cache misses.

### #4 — `wake_all` latency (data + metadata) (15.8 ms, 0.9% of total)

193000 profiled `wake_all` calls (~250/`sink_table`). Split: 100% of combined row time in `push_buf` `data_waker`, 0% in `update_max`/`update_min` rows (those rows include the atomic `fetch_*`, not only `wake_all`). Metadata waiters are woken only when `fetch_max`/`fetch_min` changes the stored value.

## Outlier Detection

| Metric | Value |
|---|---|
| Worst single `push_buf` | 5195.4 µs |
| Worst single `apply_value` | 5225.8 µs |
| Tick p99/p50 ratio | 6.4× |

> **Warning:** A single `push_buf` call took more than half the mean tick time. This is likely a **mmap page fault** on first write to a new 4 KB page.

## Optimization Recommendations

1. **Cache VTable resolution** — Currently 23% of `sink_table` time. Pre-compile the field→component mapping on first `apply()` and reuse it on subsequent ticks, bypassing the recursive `realize()` chain.


---

# elodin-db Profiling Report

**Generated:** 21:24:48 UTC

## Configuration

| Parameter | Value |
|---|---|
| Scenario | stress |
| Components | 400 |
| Frequency | 1000 Hz |
| Duration | 10 s |
| Mode | batch |
| Total ticks recorded | 627 |
| Total `sink_table` calls | 627 |

## Tick Latency Distribution

| Percentile | Latency |
|---|---|
| p50 | 178.4 µs |
| p75 | 230.0 µs |
| p90 | 287.2 µs |
| p95 | 349.0 µs |
| p99 | 3319.7 µs |
| max | 117532.8 µs |
| mean | 728.6 µs |

## Phase Breakdown (cumulative)

| Phase | Total Time | % of sink_table | Calls | Mean/call | Source |
|---|---|---|---|---|---|
| RwLock acquire | 310.1 ms | 67.9% | 627 | 494526 ns | `lib.rs` `db.with_state()` |
| VTable resolve | 145.6 ms | 31.9% | 0 | 0 ns | `vtable.rs` `realize_fields()` |
| apply_value total | 133.1 ms | 29.1% | 62700 | 2123 ns | `lib.rs` `DBSink::apply_value()` |
|   push_buf | 78.9 ms | 17.3% | 62700 | 1258 ns | `time_series.rs` `push_buf()` |
|     data write | 31.3 ms | 6.9% | 62700 | 499 ns | `append_log.rs` `data.write()` |
|     index write | 4.6 ms | 1.0% | 62700 | 74 ns | `append_log.rs` `index.write()` |
|     timestamp check | 5.6 ms | 1.2% | 62700 | 90 ns | `time_series.rs` last_ts read |
|   wake_all (data) | 10.6 ms | 2.3% | 62700 | 169 ns | `time_series.rs` `data_waker` |
|   wake_all (last_upd) | 0.0 ms | 0.0% | 62700 | 0 ns | `lib.rs` `update_max()` |
|   wake_all (earliest) | 0.0 ms | 0.0% | 62700 | 0 ns | `lib.rs` `update_min()` |
|   HashMap lookup | 8.6 ms | 1.9% | 62700 | 137 ns | `lib.rs` `components.get()` |

## Identified Bottlenecks

### #1 — VTable realize chain (145.6 ms, 31.9% of total)

0 field resolutions (0/tick). Each field triggers ~7 recursive `realize()` calls. The VTable structure is static — results could be cached after first resolve.

### #2 — push_buf Mutex churn (36.0 ms, 7.9% of total)

2 Mutex acquisitions per component per tick = 125400 lock/unlock pairs. Uncontested in single-client batch mode but still ~20 ns each.

### #3 — `wake_all` latency (data + metadata) (10.6 ms, 2.3% of total)

62700 profiled `wake_all` calls (~100/`sink_table`). Split: 100% of combined row time in `push_buf` `data_waker`, 0% in `update_max`/`update_min` rows (those rows include the atomic `fetch_*`, not only `wake_all`). Metadata waiters are woken only when `fetch_max`/`fetch_min` changes the stored value.

### #4 — HashMap lookups (8.6 ms, 1.9% of total)

62700 lookups. With 400 components the table fits in L2 cache, but pointer chasing to heap-allocated Components causes cache misses.

## Outlier Detection

| Metric | Value |
|---|---|
| Worst single `push_buf` | 3277.8 µs |
| Worst single `apply_value` | 5623.9 µs |
| Tick p99/p50 ratio | 18.6× |

> **Warning:** A single `push_buf` call took more than half the mean tick time. This is likely a **mmap page fault** on first write to a new 4 KB page.

## Optimization Recommendations

1. **Cache VTable resolution** — Currently 32% of `sink_table` time. Pre-compile the field→component mapping on first `apply()` and reuse it on subsequent ticks, bypassing the recursive `realize()` chain.


---

