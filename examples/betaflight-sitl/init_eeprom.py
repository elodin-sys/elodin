#!/usr/bin/env python3

import socket
import subprocess
import tempfile
import time
from pathlib import Path

root = Path(__file__).resolve().parent
binary = root / "betaflight/obj/main/betaflight_SITL.elf"
CLI_HOST = "localhost"
CLI_PORT = 5761


def receive_until(cli, marker):
    response = b""
    while marker not in response:
        chunk = cli.recv(4096)
        if not chunk:
            raise ConnectionError("Betaflight closed the CLI connection")
        response += chunk
    return response.decode(errors="replace")


def _sitl_pids():
    listed = subprocess.run(
        ["pgrep", "-f", "betaflight_SITL.elf"],
        check=False,
        capture_output=True,
        text=True,
    )
    return [int(pid) for pid in listed.stdout.split() if pid.isdigit()]


def _cli_is_open():
    try:
        with socket.create_connection((CLI_HOST, CLI_PORT), timeout=0.1):
            return True
    except OSError:
        return False


def stop_sitl(proc=None):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
    for pid in _sitl_pids():
        subprocess.run(["kill", str(pid)], check=False)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if not _sitl_pids() and not _cli_is_open():
            time.sleep(0.2)
            return
        time.sleep(0.05)
    for pid in _sitl_pids():
        subprocess.run(["kill", "-9", str(pid)], check=False)
    time.sleep(0.2)


def start_sitl():
    if not binary.is_file():
        raise FileNotFoundError(f"SITL binary not found: {binary}")
    stop_sitl()
    # stdout stays discarded: first-boot FLASH_ProgramWord prints every word and
    # will deadlock if that stream is an unread pipe.
    stderr = tempfile.TemporaryFile()
    sitl = subprocess.Popen(
        [binary],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=stderr,
    )
    # First-boot eeprom programming can take several seconds before UART1 binds.
    cli = None
    for _ in range(150):
        if sitl.poll() is not None:
            break
        try:
            cli = socket.create_connection((CLI_HOST, CLI_PORT), timeout=0.2)
            break
        except OSError:
            time.sleep(0.1)
    if cli is None:
        stop_sitl(sitl)
        stderr.seek(0)
        detail = stderr.read().decode(errors="replace").strip() or "no stderr"
        stderr.close()
        raise TimeoutError(f"Betaflight CLI did not start: {detail}")
    stderr.close()
    cli.settimeout(10)
    cli.sendall(b"#")
    receive_until(cli, b"# ")
    return sitl, cli


def command(cli, text):
    cli.sendall(f"{text}\n".encode())
    return receive_until(cli, b"# ")


def initialize():
    sitl, cli = start_sitl()
    try:
        _configure(cli)
        sitl.wait(timeout=10)
        # Let the reboot release UDP/TCP before the verification process starts.
        stop_sitl(sitl)
    except Exception:
        stop_sitl(sitl)
        raise


def _configure(cli):
    with cli:
        for text in (
            # AUX1 arms above 1700; AUX2 selects ANGLE mode above 1700.
            "aux 0 0 0 1700 2100 0 0",
            "aux 1 1 1 1700 2100 0 0",
            "set gyro_hardware_lpf = NORMAL",
            "set pid_process_denom = 1",
            # RC smoothing's auto cutoff relies on a valid RX frame-rate
            # measurement, which requires frame intervals >= RX_INTERVAL_MIN_US
            # (800us). At lockstep rates above ~1.25kHz the interval is always
            # below that floor, so the filter gain is never initialized and the
            # throttle channel is smoothed to zero. The SITL link is clean and
            # step-wise, so smoothing is unnecessary - disable it.
            "set rc_smoothing = off",
        ):
            command(cli, text)
        cli.sendall(b"save\n")
        receive_until(cli, b"Rebooting")


def verify():
    sitl, cli = start_sitl()
    expected = (
        ("get gyro_hardware_lpf", "gyro_hardware_lpf = NORMAL"),
        ("get pid_process_denom", "pid_process_denom = 1"),
        ("get rc_smoothing", "rc_smoothing = OFF"),
        ("aux", "aux 0 0 0 1700 2100 0 0"),
        ("aux", "aux 1 1 1 1700 2100 0 0"),
    )

    missing = []
    try:
        with cli:
            for query, value in expected:
                if value not in command(cli, query):
                    missing.append(value)
            cli.sendall(b"exit\n")
            receive_until(cli, b"Rebooting")
        sitl.wait()
    except Exception:
        stop_sitl(sitl)
        raise

    if missing:
        raise RuntimeError("EEPROM verification failed: " + ", ".join(missing))


initialize()
verify()
print(f"Initialized and verified {root / 'eeprom.bin'}")
