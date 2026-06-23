#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import signal
import socket
import threading

STATE_PORT = int(os.environ.get("ELODIN_MC_PORT_STATE", "9003"))
COMMAND_PORT = int(os.environ.get("ELODIN_MC_PORT_COMMAND", "9002"))
STOP = threading.Event()


def stop(_signum: int, _frame: object) -> None:
    STOP.set()


def main() -> None:
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind(("127.0.0.1", STATE_PORT))
    recv_sock.settimeout(0.05)
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while not STOP.is_set():
        try:
            raw, _ = recv_sock.recvfrom(1024)
        except TimeoutError:
            continue
        state = json.loads(raw.decode())
        error = float(state["target"]) - float(state["position"])
        velocity = float(state["velocity"])
        command = max(min(error * 1.2 - velocity * 0.35, 20.0), -20.0)
        send_sock.sendto(
            json.dumps({"command": command}).encode(),
            ("127.0.0.1", COMMAND_PORT),
        )


if __name__ == "__main__":
    main()
