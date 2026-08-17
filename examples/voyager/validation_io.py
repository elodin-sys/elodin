"""Stdout contract between the Voyager example and the truth harness."""

from __future__ import annotations

import json
from typing import Any

METRIC_PREFIX = "VOYAGER_VALIDATION_METRIC"
SEGMENT_PREFIX = "VOYAGER_ACTIVE_SEGMENT"
PLANET_SEGMENT_PREFIX = "VOYAGER_PLANET_SEGMENT"
INITIAL_STATE_PREFIX = "VOYAGER_INITIAL_STATE"


def emit_prefixed_json(prefix: str, payload: dict[str, Any]) -> None:
    print(f"{prefix} {json.dumps(payload, sort_keys=True)}", flush=True)


def parse_prefixed_json(output: str, prefix: str) -> list[dict[str, Any]]:
    records = []
    for line in output.splitlines():
        if not line.startswith(prefix):
            continue
        payload = line[len(prefix) :].lstrip()
        records.append(json.loads(payload))
    return records


def load_checkpoints_json(raw: str) -> tuple[dict[str, Any], ...]:
    try:
        checkpoints = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"VOYAGER_CHECKPOINTS_JSON is not valid JSON: {exc}") from exc
    if not isinstance(checkpoints, list):
        raise TypeError("VOYAGER_CHECKPOINTS_JSON must be a JSON array")
    for checkpoint in checkpoints:
        if (
            not isinstance(checkpoint, dict)
            or "name" not in checkpoint
            or "elapsed_seconds" not in checkpoint
        ):
            raise ValueError("each checkpoint needs name and elapsed_seconds")
    return tuple(checkpoints)
