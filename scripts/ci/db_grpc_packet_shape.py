#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

PCAP_MAGIC = {
    b"\xa1\xb2\xc3\xd4",
    b"\xd4\xc3\xb2\xa1",
    b"\xa1\xb2\x3c\x4d",
    b"\x4d\x3c\xb2\xa1",
    b"\x0a\x0d\x0d\x0a",
}


def tcpdump_text(path: Path, port: int | None) -> str:
    tcpdump = shutil.which("tcpdump")
    if tcpdump is None:
        raise RuntimeError("tcpdump is required to read pcap files")
    command = [tcpdump, "-nn", "-tt", "-r", str(path)]
    if port is not None:
        command.extend(["tcp", "and", "port", str(port)])
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout


def read_input(path: str, port: int | None) -> str:
    if path == "-":
        return sys.stdin.read()
    input_path = Path(path)
    with input_path.open("rb") as stream:
        magic = stream.read(4)
    if magic in PCAP_MAGIC:
        return tcpdump_text(input_path, port)
    return input_path.read_text(errors="replace")


def payload_lengths(text: str, port: int | None) -> list[int]:
    lengths = []
    port_pattern = re.compile(rf"(?:[.:]){port}\b") if port is not None else None
    for line in text.splitlines():
        if port_pattern is not None and port_pattern.search(line) is None:
            continue
        matches = re.findall(r"\blength (\d+)\b", line)
        if matches:
            lengths.append(int(matches[-1]))
    return lengths


def percentile(sorted_values: list[int], percentile_value: int) -> int:
    if not sorted_values:
        return 0
    index = max(0, (len(sorted_values) * percentile_value + 99) // 100 - 1)
    return sorted_values[index]


def summarize(lengths: list[int], mss: int, tiny: int) -> dict:
    payloads = sorted(length for length in lengths if length > 0)
    classes = {
        "tiny": [length for length in payloads if length <= tiny],
        "middle": [length for length in payloads if tiny < length < mss],
        "full_mss_or_larger": [length for length in payloads if length >= mss],
    }
    payload_count = len(payloads)
    payload_bytes = sum(payloads)

    def class_summary(values: list[int]) -> dict:
        return {
            "packets": len(values),
            "packet_percent": 100.0 * len(values) / payload_count if payload_count else 0.0,
            "bytes": sum(values),
            "byte_percent": 100.0 * sum(values) / payload_bytes if payload_bytes else 0.0,
        }

    return {
        "mss_bytes": mss,
        "tiny_max_bytes": tiny,
        "tcp_packets": len(lengths),
        "ack_only_packets": len(lengths) - payload_count,
        "payload_packets": payload_count,
        "payload_bytes": payload_bytes,
        "payload_p50_bytes": percentile(payloads, 50),
        "payload_p95_bytes": percentile(payloads, 95),
        "payload_p99_bytes": percentile(payloads, 99),
        "exact_mss_packets": sum(length == mss for length in payloads),
        "larger_than_mss_packets": sum(length > mss for length in payloads),
        "distribution": {name: class_summary(values) for name, values in classes.items()},
    }


def print_human(report: dict) -> None:
    print(
        f"TCP packets: {report['tcp_packets']} "
        f"(payload {report['payload_packets']}, ack-only {report['ack_only_packets']})"
    )
    print(
        "Payload bytes: "
        f"{report['payload_bytes']} "
        f"(p50={report['payload_p50_bytes']}, "
        f"p95={report['payload_p95_bytes']}, p99={report['payload_p99_bytes']})"
    )
    for name, values in report["distribution"].items():
        print(
            f"{name}: {values['packets']} packets "
            f"({values['packet_percent']:.1f}%), "
            f"{values['bytes']} bytes ({values['byte_percent']:.1f}%)"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report TCP payload packet shapes from tcpdump text or pcap"
    )
    parser.add_argument("input", help="tcpdump text, pcap/pcapng, or - for stdin")
    parser.add_argument("--port", type=int)
    parser.add_argument("--mss", type=int, default=1460)
    parser.add_argument("--tiny-max", type=int, default=256)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.mss <= 0 or args.tiny_max < 0 or args.tiny_max >= args.mss:
        parser.error("require 0 <= tiny-max < mss")

    try:
        lengths = payload_lengths(read_input(args.input, args.port), args.port)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if not lengths:
        print("error: no tcpdump TCP length fields found", file=sys.stderr)
        return 1
    report = summarize(lengths, args.mss, args.tiny_max)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
