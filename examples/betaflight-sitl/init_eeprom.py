#!/usr/bin/env python3

import socket
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parent
binary = root / "betaflight/obj/main/betaflight_SITL.elf"


def receive_until(cli, marker):
    response = b""
    while marker not in response:
        chunk = cli.recv(4096)
        if not chunk:
            raise ConnectionError("Betaflight closed the CLI connection")
        response += chunk
    return response.decode(errors="replace")


def start_sitl():
    sitl = subprocess.Popen(
        [binary],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(50):
        try:
            cli = socket.create_connection(("localhost", 5761), timeout=0.2)
            break
        except OSError:
            time.sleep(0.1)
    else:
        raise TimeoutError("Betaflight CLI did not start")
    cli.settimeout(10)
    cli.sendall(b"#")
    receive_until(cli, b"# ")
    return sitl, cli


def command(cli, text):
    cli.sendall(f"{text}\n".encode())
    return receive_until(cli, b"# ")


def initialize():
    sitl, cli = start_sitl()
    with cli:
        for text in (
            "aux 0 0 0 1700 2100 0 0",
            "set gyro_hardware_lpf = NORMAL",
            "set pid_process_denom = 1",
        ):
            command(cli, text)
        cli.sendall(b"save\n")
        receive_until(cli, b"Rebooting")
    sitl.wait()


def verify():
    sitl, cli = start_sitl()
    expected = (
        ("get gyro_hardware_lpf", "gyro_hardware_lpf = NORMAL"),
        ("get pid_process_denom", "pid_process_denom = 1"),
        ("aux", "aux 0 0 0 1700 2100 0 0"),
    )

    missing = []
    with cli:
        for query, value in expected:
            if value not in command(cli, query):
                missing.append(value)
        cli.sendall(b"exit\n")
        receive_until(cli, b"Rebooting")
    sitl.wait()

    if missing:
        raise RuntimeError("EEPROM verification failed: " + ", ".join(missing))


initialize()
verify()
print(f"Initialized and verified {root / 'eeprom.bin'}")
