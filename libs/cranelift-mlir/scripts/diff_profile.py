#!/usr/bin/env python3
"""Compare two cranelift-mlir profile JSON captures.

Usage:
    diff_profile.py <baseline.json> <new.json> [--top N]

Consumes the `profile.json` file cranelift-mlir writes into
`$ELODIN_CRANELIFT_DEBUG_DIR` at simulation exit. Prints per-function
deltas, overall tick timing (raw and probe-overhead corrected),
SIMD-utilization delta, cross-ABI marshal delta (calls / bytes / ns),
libm-vs-SIMD transcendental split, top hot-edge deltas, and per-op
wall time. Exits non-zero only on JSON parse failure.

Example:

    ELODIN_BACKEND=cranelift ELODIN_CRANELIFT_DEBUG_DIR=/tmp/base \\
        elodin-cli sim run customer_sim.py
    # ... make changes ...
    ELODIN_BACKEND=cranelift ELODIN_CRANELIFT_DEBUG_DIR=/tmp/new \\
        elodin-cli sim run customer_sim.py
    libs/cranelift-mlir/scripts/diff_profile.py \\
        /tmp/base/profile.json /tmp/new/profile.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def pct_delta(old: float, new: float) -> str:
    if old == 0 and new == 0:
        return "  0.0%"
    if old == 0:
        return "  +inf%"
    d = 100.0 * (new - old) / old
    sign = "+" if d >= 0 else ""
    return f"  {sign}{d:.1f}%"


def fmt_ns(ns: float) -> str:
    if ns >= 1e9:
        return f"{ns / 1e9:7.2f} s "
    if ns >= 1e6:
        return f"{ns / 1e6:7.2f} ms"
    if ns >= 1e3:
        return f"{ns / 1e3:7.2f} us"
    return f"{ns:7.0f} ns"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline", type=Path, help="Baseline JSON profile")
    ap.add_argument("new", type=Path, help="New JSON profile")
    ap.add_argument(
        "--top",
        type=int,
        default=15,
        help="Show top N hot functions (default: 15)",
    )
    args = ap.parse_args()

    try:
        base = json.loads(args.baseline.read_text())
        new = json.loads(args.new.read_text())
    except Exception as e:
        print(f"failed to parse JSON: {e}", file=sys.stderr)
        return 1

    print(f"comparing {args.baseline} → {args.new}\n")

    # Overall tick timing — raw and probe-overhead corrected.
    base_wall = base.get("main_wall_ns", 0)
    new_wall = new.get("main_wall_ns", 0)
    base_ticks = max(base.get("main_ticks", 0), 1)
    new_ticks = max(new.get("main_ticks", 0), 1)
    base_mean = base_wall / base_ticks
    new_mean = new_wall / new_ticks
    print("main tick timing (raw):")
    print(f"  base: {fmt_ns(base_mean)} / tick × {base_ticks} ticks = {fmt_ns(base_wall)}")
    print(f"  new:  {fmt_ns(new_mean)} / tick × {new_ticks} ticks = {fmt_ns(new_wall)}")
    print(f"  mean delta: {pct_delta(base_mean, new_mean)}")

    bc = base.get("main_wall_ns_corrected", 0)
    nc = new.get("main_wall_ns_corrected", 0)
    bc_mean = bc / base_ticks
    nc_mean = nc / new_ticks
    print("main tick timing (probe-overhead corrected):")
    print(f"  base: {fmt_ns(bc_mean)} / tick")
    print(f"  new:  {fmt_ns(nc_mean)} / tick")
    print(f"  mean delta: {pct_delta(bc_mean, nc_mean)}")

    b_po = base.get("profile_overhead", {})
    n_po = new.get("profile_overhead", {})
    print(
        "  probe overhead:        "
        f"base={b_po.get('measured_ns_per_probe', 0)} ns/pair → "
        f"new={n_po.get('measured_ns_per_probe', 0)} ns/pair"
    )
    print()

    # Per-function deltas (by cumulative time).
    def func_by_name(report):
        return {f["name"]: f for f in report.get("functions", [])}

    base_fns = func_by_name(base)
    new_fns = func_by_name(new)
    all_names = sorted(set(base_fns) | set(new_fns))

    deltas = []
    for name in all_names:
        b = base_fns.get(name, {})
        n = new_fns.get(name, {})
        b_ns = b.get("total_ns", 0)
        n_ns = n.get("total_ns", 0)
        deltas.append((name, b, n, b_ns, n_ns, n_ns - b_ns))

    deltas.sort(key=lambda t: abs(t[5]), reverse=True)

    print(f"top {args.top} functions by absolute time delta (new − base):")
    print(
        f"  {'name':<24} {'base':>12} {'new':>12} {'delta':>12}  {'pct':>8}"
        f"  {'p99 delta':>12}  {'excl delta':>12}"
    )
    for name, b, n, b_ns, n_ns, d_ns in deltas[: args.top]:
        pct = pct_delta(b_ns, n_ns)
        b_p99 = b.get("p99_ns", 0)
        n_p99 = n.get("p99_ns", 0)
        b_excl = b.get("exclusive_ns", 0)
        n_excl = n.get("exclusive_ns", 0)
        print(
            f"  {name:<24} {fmt_ns(b_ns):>12} {fmt_ns(n_ns):>12}"
            f" {fmt_ns(d_ns):>12}  {pct:>8}"
            f"  {fmt_ns(n_p99 - b_p99):>12}  {fmt_ns(n_excl - b_excl):>12}"
        )
    print()

    # Inline vs calls split: distinguishes "more inline work" from
    # "more time in callees" so tensor_rt SIMD changes and JIT-inliner
    # changes attribute separately.
    print("inline-vs-calls shift (top 10 by |delta_calls|):")
    call_deltas = []
    for name, b, n, _, _, _ in deltas:
        b_ic = b.get("time_in_calls_ns", 0)
        n_ic = n.get("time_in_calls_ns", 0)
        if b_ic == 0 and n_ic == 0:
            continue
        call_deltas.append((name, b_ic, n_ic, n_ic - b_ic))
    call_deltas.sort(key=lambda t: abs(t[3]), reverse=True)
    print(f"  {'name':<24} {'base calls':>12} {'new calls':>12} {'delta':>12}")
    for name, b_ic, n_ic, d in call_deltas[:10]:
        print(f"  {name:<24} {fmt_ns(b_ic):>12} {fmt_ns(n_ic):>12} {fmt_ns(d):>12}")
    print()

    # SIMD utilization.
    b_simd = base.get("simd", {})
    n_simd = new.get("simd", {})
    print("simd utilization:")
    print(
        f"  static:           "
        f"{b_simd.get('static_vector_pct', 0):5.1f}% → "
        f"{n_simd.get('static_vector_pct', 0):5.1f}%"
    )
    print(
        f"  runtime-weighted: "
        f"{b_simd.get('runtime_weighted_vector_pct', 0):5.1f}% → "
        f"{n_simd.get('runtime_weighted_vector_pct', 0):5.1f}%"
    )
    print()

    # Cross-ABI marshal: calls, bytes, and ns per direction.
    b_mar = base.get("marshal", {})
    n_mar = new.get("marshal", {})
    print("cross-ABI marshal (per-sim totals):")
    for direction in ("scalar_to_pointer", "pointer_to_scalar"):
        bc = b_mar.get(f"{direction}_calls", 0)
        nc = n_mar.get(f"{direction}_calls", 0)
        bb = b_mar.get(f"{direction}_bytes", 0)
        nb = n_mar.get(f"{direction}_bytes", 0)
        bn = b_mar.get(f"{direction}_total_ns", 0)
        nn = n_mar.get(f"{direction}_total_ns", 0)
        print(f"  {direction:<20} calls: {bc:>10} → {nc:>10} ({pct_delta(bc, nc)})")
        print(f"  {'':<20} bytes: {bb:>10} → {nb:>10} ({pct_delta(bb, nb)})")
        print(f"  {'':<20} time:  {fmt_ns(bn):>10} → {fmt_ns(nn):>10} ({pct_delta(bn, nn)})")
    print()

    # Transcendental split (libm scalar fallback vs wide-SIMD batch).
    b_x = base.get("transcendental", {})
    n_x = new.get("transcendental", {})
    print("transcendental calls:")
    print(
        f"  libm scalar:    "
        f"{b_x.get('libm_scalar_calls', 0):>10} → {n_x.get('libm_scalar_calls', 0):>10}"
    )
    print(
        f"  wide-SIMD:      "
        f"{b_x.get('wide_simd_calls', 0):>10} → {n_x.get('wide_simd_calls', 0):>10}"
    )
    print(
        f"  libm time:      "
        f"{fmt_ns(b_x.get('libm_total_ns', 0)):>10} → "
        f"{fmt_ns(n_x.get('libm_total_ns', 0)):>10}"
    )
    print(
        f"  wide-SIMD time: "
        f"{fmt_ns(b_x.get('wide_simd_total_ns', 0)):>10} → "
        f"{fmt_ns(n_x.get('wide_simd_total_ns', 0)):>10}"
    )
    print()

    # Hot-edge deltas.
    def edges_by_key(report):
        return {(e["parent_name"], e["callee_name"]): e for e in report.get("call_graph", [])}

    b_edges = edges_by_key(base)
    n_edges = edges_by_key(new)
    all_edge_keys = set(b_edges) | set(n_edges)
    edge_deltas = []
    for k in all_edge_keys:
        bn = b_edges.get(k, {}).get("total_ns", 0)
        nn = n_edges.get(k, {}).get("total_ns", 0)
        edge_deltas.append((k, bn, nn, nn - bn))
    edge_deltas.sort(key=lambda t: abs(t[3]), reverse=True)
    print("hot-edge deltas (top 10 by absolute time):")
    for (parent, callee), bn, nn, d in edge_deltas[:10]:
        print(
            f"  {parent:<16} → {callee:<16} {fmt_ns(bn):>12} → "
            f"{fmt_ns(nn):>12}  delta={fmt_ns(d):>12}  ({pct_delta(bn, nn)})"
        )
    print()

    # Per-op wall-time table. Present when `op_category_timing` is a
    # non-empty list on both sides; the diff highlights whether
    # elision or inliner changes moved the expected ops (fmul, load,
    # store, etc.).
    b_op = base.get("op_category_timing", []) or []
    n_op = new.get("op_category_timing", []) or []
    if b_op and n_op:

        def op_map(rows):
            return {r.get("name"): r for r in rows}

        bm = op_map(b_op)
        nm = op_map(n_op)
        all_names = sorted(set(bm) | set(nm))
        print("per-op wall time (sampled):")
        print(
            f"  {'op':<12} {'base ns':>12} {'base smp':>10}  {'new ns':>12} {'new smp':>10}  {'delta ns':>12} {'delta':>8}"
        )
        rows_with_delta = []
        for name in all_names:
            bn_ns = bm.get(name, {}).get("total_ns", 0)
            bn_sa = bm.get(name, {}).get("sample_count", 0)
            nn_ns = nm.get(name, {}).get("total_ns", 0)
            nn_sa = nm.get(name, {}).get("sample_count", 0)
            rows_with_delta.append((name, bn_ns, bn_sa, nn_ns, nn_sa, nn_ns - bn_ns))
        rows_with_delta.sort(key=lambda r: abs(r[5]), reverse=True)
        for name, bn_ns, bn_sa, nn_ns, nn_sa, delta_ns in rows_with_delta:
            print(
                f"  {name:<12} {fmt_ns(bn_ns):>12} {bn_sa:>10}  "
                f"{fmt_ns(nn_ns):>12} {nn_sa:>10}  "
                f"{fmt_ns(delta_ns):>12} {pct_delta(bn_ns, nn_ns):>8}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
