#!/usr/bin/env python3
"""Stands in for an asynchronous rendered sensor. For each capture, it reads what the simulation
committed for agent 0 and its object at the capture's source time, waits, and publishes a small
frame message stamped with that source time. A progress message then says it handled the capture,
even when it dropped the frame on purpose."""

import os
import sys
import time

import elodin.db as edb
import numpy as np

from protocol import FRAME_DTYPE

ADDRESS = sys.argv[1]
PERIOD_US = int(sys.argv[2])
DELAY_S = float(os.environ.get("ELODIN_MANY_AGENT_PRODUCER_DELAY_MS", "2")) / 1000.0
DROP_EVERY = int(os.environ.get("ELODIN_MANY_AGENT_PRODUCER_DROP_EVERY", "7"))
FRAMES = int(os.environ.get("ELODIN_MANY_AGENT_PRODUCER_FRAMES", "0"))


def connect() -> edb.Client:
    # The simulation serves the DB once it has compiled, after its recipes are ready.
    deadline = time.monotonic() + 60.0
    while True:
        client = edb.Client.connect(ADDRESS)
        try:
            client.components()
            return client
        except RuntimeError:
            client.close()
            if time.monotonic() > deadline:
                raise
            time.sleep(0.1)


def committed(client: edb.Client, name: str, source: int) -> np.ndarray:
    """The value of `name` at `source`, once the simulation's commit of it is in the DB."""
    while True:
        timestamps, values = client.time_series(name, source, source + 1)
        if len(timestamps):
            return values[0]
        time.sleep(0.0002)


def main() -> None:
    client = connect()
    frame = 1
    while FRAMES == 0 or frame <= FRAMES:
        source = frame * PERIOD_US
        agent = committed(client, "agent_000.position", source)
        observed = committed(client, "object_000.object_position", source)
        time.sleep(DELAY_S)
        if DROP_EVERY == 0 or frame % DROP_EVERY:
            payload = np.array([(frame, source, agent, observed)], dtype=FRAME_DTYPE)
            client.send_msg("camera.frame", payload.tobytes(), source)
        client.send_msg("camera.progress", np.int64(frame).tobytes(), source)
        frame += 1


if __name__ == "__main__":
    main()
