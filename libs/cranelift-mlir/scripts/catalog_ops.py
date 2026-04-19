#!/usr/bin/env python3
"""Catalog MLIR ops from a large StableHLO file without OOM.

Reads line-by-line and truncates each line before 'dense<' constants
to avoid regex-scanning 100KB+ constant literal strings.

Usage: python3 catalog_ops.py <path-to-stablehlo.mlir>
"""

import re
import sys
import collections

pat = re.compile(r"((?:stablehlo|chlo|func)\.[a-z_]+)")
counts = collections.Counter()

with open(sys.argv[1]) as f:
    for line in f:
        idx = line.find("dense<")
        if idx >= 0:
            line = line[:idx]
        for m in pat.finditer(line):
            counts[m.group(1)] += 1

for op, n in counts.most_common():
    print(f"{n:8d}  {op}")
