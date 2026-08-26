import json
import os
from pathlib import Path

import pytest

from scripts.boson_ref.compare_rgba import evaluate_metrics, rgba_metrics

REFERENCE = Path(__file__).with_name("data") / "boson_reference_metrics.json"


def test_boson_reference_metrics_define_acceptance_window():
    metrics = json.loads(REFERENCE.read_text(encoding="utf-8"))["metrics"]
    assert evaluate_metrics(metrics) == []


@pytest.mark.skipif(
    "ELODIN_LWIR_FRAME" not in os.environ,
    reason="set ELODIN_LWIR_FRAME to a raw 640x512 RGBA sensor frame",
)
def test_rendered_lwir_frame_matches_boson_reference():
    frame = Path(os.environ["ELODIN_LWIR_FRAME"]).read_bytes()
    metrics = rgba_metrics(frame, 640, 512)
    assert evaluate_metrics(metrics) == []
