#!/usr/bin/env python3
"""Extract and characterize post-AGC Boson NV12 recordings."""

from __future__ import annotations

import argparse
import json
import math
import struct
import zlib
from collections.abc import Iterable, Iterator
from pathlib import Path

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 512
DEFAULT_REFERENCE_FRAMES = (5, 750, 810, 840, 870, 909)


def png_chunk(kind: bytes, data: bytes) -> bytes:
    payload = kind + data
    return (
        struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
    )


def write_grayscale_png(path: Path, pixels: bytes, width: int, height: int) -> None:
    if len(pixels) != width * height:
        raise ValueError(f"expected {width * height} pixels, got {len(pixels)}")
    rows = b"".join(b"\x00" + pixels[row * width : (row + 1) * width] for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(rows, 9))
        + png_chunk(b"IEND", b"")
    )


def raw_chunks(data_dir: Path) -> list[Path]:
    chunks = sorted(data_dir.glob("flight_video_*.raw"))
    if not chunks:
        raise FileNotFoundError(f"no flight_video_*.raw files in {data_dir}")
    return chunks


def nv12_frame_size(width: int, height: int) -> int:
    return width * height * 3 // 2


def frames_per_chunk(path: Path, width: int, height: int) -> int:
    frame_size = nv12_frame_size(width, height)
    size = path.stat().st_size
    if size % frame_size != 0:
        raise ValueError(f"{path} size {size} is not a whole number of NV12 frames")
    return size // frame_size


def iter_y_frames(chunks: Iterable[Path], width: int, height: int) -> Iterator[tuple[int, bytes]]:
    frame_size = nv12_frame_size(width, height)
    y_size = width * height
    global_frame = 0
    for chunk in chunks:
        with chunk.open("rb") as stream:
            for _ in range(frames_per_chunk(chunk, width, height)):
                y_plane = stream.read(y_size)
                if len(y_plane) != y_size:
                    raise EOFError(f"short Y plane in {chunk}")
                stream.seek(frame_size - y_size, 1)
                yield global_frame, y_plane
                global_frame += 1


def selected_y_frames(
    chunks: Iterable[Path],
    frame_ids: set[int],
    width: int,
    height: int,
) -> dict[int, bytes]:
    selected: dict[int, bytes] = {}
    for frame_id, y_plane in iter_y_frames(chunks, width, height):
        if frame_id in frame_ids:
            selected[frame_id] = y_plane
        if len(selected) == len(frame_ids):
            break
    missing = frame_ids - selected.keys()
    if missing:
        raise IndexError(f"frame ids outside recording: {sorted(missing)}")
    return selected


def percentile(histogram: list[int], fraction: float) -> int:
    total = sum(histogram)
    threshold = max(1, math.ceil(total * fraction))
    seen = 0
    for value, count in enumerate(histogram):
        seen += count
        if seen >= threshold:
            return value
    return 255


def frame_metrics(pixels: bytes, width: int, height: int) -> dict[str, float | int]:
    histogram = [0] * 256
    for value in pixels:
        histogram[value] += 1

    gradient_sum = 0
    laplacian_sum_sq = 0
    edge_count = 0
    sample_count = 0
    for y in range(1, height - 1, 2):
        row = y * width
        for x in range(1, width - 1, 2):
            index = row + x
            center = pixels[index]
            dx = abs(pixels[index + 1] - pixels[index - 1])
            dy = abs(pixels[index + width] - pixels[index - width])
            gradient = dx + dy
            laplacian = (
                4 * center
                - pixels[index - 1]
                - pixels[index + 1]
                - pixels[index - width]
                - pixels[index + width]
            )
            gradient_sum += gradient
            laplacian_sum_sq += laplacian * laplacian
            edge_count += gradient >= 32
            sample_count += 1

    mean = sum(value * count for value, count in enumerate(histogram)) / len(pixels)
    return {
        "min_dn": next(i for i, count in enumerate(histogram) if count),
        "p01_dn": percentile(histogram, 0.01),
        "p50_dn": percentile(histogram, 0.50),
        "p99_dn": percentile(histogram, 0.99),
        "max_dn": next(i for i in range(255, -1, -1) if histogram[i]),
        "mean_dn": round(mean, 3),
        "mean_gradient_dn": round(gradient_sum / sample_count, 3),
        "edge_fraction": round(edge_count / sample_count, 6),
        "laplacian_rms_dn": round(math.sqrt(laplacian_sum_sq / sample_count), 3),
    }


def temporal_noise_dn(
    first: bytes, second: bytes, width: int, height: int
) -> dict[str, float | int]:
    differences: list[int] = []
    for y in range(1, height - 1, 3):
        row = y * width
        for x in range(1, width - 1, 3):
            index = row + x
            local_gradient = max(
                abs(first[index + 1] - first[index - 1]),
                abs(first[index + width] - first[index - width]),
            )
            if local_gradient <= 8:
                differences.append(first[index] - second[index])
    if not differences:
        raise ValueError("no low-gradient samples available for temporal noise")
    mean = sum(differences) / len(differences)
    variance = sum((value - mean) ** 2 for value in differences) / len(differences)
    frame_difference_sigma = math.sqrt(variance)
    return {
        "sample_count": len(differences),
        "frame_difference_sigma_dn": round(frame_difference_sigma, 3),
        "estimated_single_frame_sigma_dn": round(frame_difference_sigma / math.sqrt(2.0), 3),
    }


def timestamp_metrics(timestamp_dir: Path) -> dict[str, float | int] | None:
    timestamps: list[int] = []
    for path in sorted(timestamp_dir.glob("camera_source_source_timestamps_*.csv")):
        with path.open(encoding="utf-8") as stream:
            next(stream, None)
            for line in stream:
                _, timestamp = line.strip().split(",", 1)
                timestamps.append(int(timestamp))
    if len(timestamps) < 2:
        return None
    intervals_ms = [
        (later - earlier) / 1_000_000.0
        for earlier, later in zip(timestamps, timestamps[1:], strict=False)
        if later > earlier
    ]
    intervals_ms.sort()
    median = intervals_ms[len(intervals_ms) // 2]
    duration_s = (timestamps[-1] - timestamps[0]) / 1e9
    return {
        "frame_count": len(timestamps),
        "duration_s": round(duration_s, 6),
        "median_interval_ms": round(median, 4),
        "mean_interval_ms": round(sum(intervals_ms) / len(intervals_ms), 4),
        "observed_fps": round((len(timestamps) - 1) / duration_s, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="cvapp data directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ai-context/bdx/assets/reference"),
        help="destination for extracted PNGs and metrics",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=list(DEFAULT_REFERENCE_FRAMES),
        help="global frame ids to extract",
    )
    parser.add_argument(
        "--noise-pair",
        type=int,
        nargs=2,
        default=(5, 6),
        metavar=("FIRST", "SECOND"),
        help="static adjacent frames used for temporal-noise estimation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunks = raw_chunks(args.data_dir)
    requested = set(args.frames) | set(args.noise_pair)
    frames = selected_y_frames(chunks, requested, args.width, args.height)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame_reports: dict[str, dict[str, float | int]] = {}
    for frame_id in args.frames:
        path = args.output_dir / f"boson_frame_{frame_id:04d}.png"
        write_grayscale_png(path, frames[frame_id], args.width, args.height)
        frame_reports[str(frame_id)] = frame_metrics(frames[frame_id], args.width, args.height)

    temporal_noise = temporal_noise_dn(
        frames[args.noise_pair[0]],
        frames[args.noise_pair[1]],
        args.width,
        args.height,
    )
    report = {
        "source": {
            "format": "NV12 post-AGC white-hot",
            "width": args.width,
            "height": args.height,
            "raw_chunks": len(chunks),
            "raw_frames": sum(frames_per_chunk(path, args.width, args.height) for path in chunks),
        },
        "timestamps": timestamp_metrics(args.data_dir / "flight_video_timestamps"),
        "temporal_noise": temporal_noise,
        "frames": frame_reports,
        "boson640p_initial_fit": {
            "resolution": [640, 512],
            "fps": 60.0,
            "lens_hfov_deg": 18.0,
            "lens_vfov_deg": 14.443,
            "palette": "white_hot",
            "agc_low_percentile": 0.01,
            "agc_high_percentile": 0.99,
            "agc_smoothing": 0.90,
            "agc_target_median": 0.35,
            "dde_strength": 0.60,
            "mtf_blur_px": 0.65,
            "temporal_noise_sigma_dn": temporal_noise["estimated_single_frame_sigma_dn"],
            "column_fpn_sigma_dn": 0.25,
            "dead_pixel_ppm": 0.0,
        },
    }
    report_path = args.output_dir / "reference_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fit = report["boson640p_initial_fit"]
    markdown = f"""# Boson+ 640 reference fit

Source: post-AGC NV12 recording, {args.width}×{args.height}, white-hot.

| Parameter | Initial fit |
|---|---:|
| Lens HFOV | {fit["lens_hfov_deg"]}° |
| Lens VFOV | {fit["lens_vfov_deg"]}° |
| Frame rate | {fit["fps"]} Hz |
| AGC percentiles | {fit["agc_low_percentile"]:.2f}–{fit["agc_high_percentile"]:.2f} |
| AGC smoothing | {fit["agc_smoothing"]:.2f} |
| AGC target median | {fit["agc_target_median"]:.2f} |
| DDE strength | {fit["dde_strength"]:.2f} |
| MTF blur radius | {fit["mtf_blur_px"]:.2f} px |
| Temporal noise σ | {fit["temporal_noise_sigma_dn"]:.3f} DN |
| Column FPN σ | {fit["column_fpn_sigma_dn"]:.2f} DN |

`reference_metrics.json` contains per-frame histogram, edge-density, gradient,
and Laplacian measurements. Temporal noise is estimated only from low-gradient
pixels in frames {args.noise_pair[0]} and {args.noise_pair[1]} to suppress
apparent differences caused by platform motion.
"""
    markdown_path = args.output_dir / "REFERENCE.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    print(f"wrote {len(args.frames)} PNGs, {report_path}, and {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
