"""Compatibility adapter: the old `impeller_py.ImpellerClient` API implemented
on top of `elodin.db` (the first-class DB client in the elodin wheel).

`impeller_py` is deprecated; these scripts now prefer `elodin.db` and only
fall back to the legacy module when the wheel is too old to provide it.
"""

from __future__ import annotations


class ComponentData:  # matches impeller_py.ComponentData
    __slots__ = ("name", "timestamp", "values", "shape")

    def __init__(self, name, timestamp, values, shape):
        self.name = name
        self.timestamp = timestamp
        self.values = values
        self.shape = shape

    def __repr__(self):
        return (
            f"ComponentData(name='{self.name}', timestamp={self.timestamp}, "
            f"values={self.values}, shape={self.shape})"
        )


class ImpellerClient:
    """Drop-in replacement for `impeller_py.ImpellerClient` backed by `elodin.db`."""

    def __init__(self, addr: str):
        import elodin.db as edb

        self._client = edb.Client.connect(addr)
        self._tracked: set[str] = set()

    def connect(self) -> None:
        pass  # elodin.db connects lazily; writers/subscriptions auto-reconnect

    def disconnect(self) -> None:
        self._client.close()

    def track_component(self, name: str) -> None:
        self._tracked.add(name)
        self._client._c.track(name)

    def subscribe_realtime(self) -> None:
        # elodin.db starts the subscription on first latest() call; prime it.
        for name in self._tracked:
            self._client.latest(name)

    def get_latest(self, name: str):
        sample = self._client.latest(name)
        if sample is None:
            return None
        values = list(sample.values.reshape(-1))
        return ComponentData(sample.name, sample.timestamp_us, values, [len(values)])

    def send_component(self, name: str, values, timestamp_us: int) -> None:
        self._client.send(name, values, timestamp_us)

    def send_component_fast(self, name: str, values, timestamp_us: int) -> None:
        self._client.send(name, values, timestamp_us)

    def reset_sender(self) -> None:
        pass  # writers reconnect automatically

    def get_connection_state(self) -> str:
        return self._client.state()

    def is_connected(self) -> bool:
        return self._client.state() == "Connected"

    def discover_components(self):
        return self._client.components()

    def list_components(self):
        return sorted(self._tracked)
