"""Component schema sources for typed expression validation (Phase 2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .expr import ComponentHandle, ExprError


class Schema:
    """Lookup table of component names → shape / element metadata."""

    def __init__(
        self,
        components: Mapping[str, Mapping[str, Any]],
        *,
        strict: bool = True,
    ):
        self._components = {k: dict(v) for k, v in components.items()}
        self.strict = strict

    @classmethod
    def from_db(cls, addr: str, *, strict: bool = True) -> Schema:
        import elodin.db as edb

        with edb.Client.connect(addr) as client:
            infos = client.components()
        components: dict[str, dict[str, Any]] = {}
        for name, info in infos.items():
            element_names = list(getattr(info, "element_names", None) or [])
            shape = list(getattr(info, "shape", None) or [])
            components[name] = {
                "element_names": element_names,
                "shape": shape,
                "prim_type": getattr(info, "prim_type", None),
            }
        return cls(components, strict=strict)

    @classmethod
    def from_json(cls, source: str | Path | Mapping[str, Any], *, strict: bool = True) -> Schema:
        if isinstance(source, (str, Path)):
            data = json.loads(Path(source).read_text())
        else:
            data = dict(source)
        raw = data.get("components", data)
        components: dict[str, dict[str, Any]] = {}
        if isinstance(raw, list):
            for item in raw:
                name = item["name"]
                meta = item.get("metadata") or {}
                element_names = []
                if "element_names" in meta:
                    element_names = [
                        p.strip() for p in str(meta["element_names"]).split(",") if p.strip()
                    ]
                elif "element_names" in item:
                    element_names = list(item["element_names"])
                components[name] = {
                    "element_names": element_names,
                    "shape": list(item.get("shape") or []),
                    "prim_type": item.get("type") or item.get("prim_type"),
                }
        elif isinstance(raw, dict):
            for name, item in raw.items():
                if not isinstance(item, Mapping):
                    continue
                meta = item.get("metadata") or {}
                element_names = list(item.get("element_names") or [])
                if not element_names and "element_names" in meta:
                    element_names = [
                        p.strip() for p in str(meta["element_names"]).split(",") if p.strip()
                    ]
                components[name] = {
                    "element_names": element_names,
                    "shape": list(item.get("shape") or []),
                    "prim_type": item.get("type") or item.get("prim_type"),
                }
        else:
            raise TypeError("components JSON must be a list or object")
        return cls(components, strict=strict)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._components

    def __getitem__(self, name: str) -> ComponentHandle:
        if name not in self._components:
            if self.strict:
                raise ExprError(f"unknown component {name!r}")
            return ComponentHandle(name, strict=False)
        info = self._components[name]
        return ComponentHandle(
            name,
            element_names=info.get("element_names") or [],
            shape=info.get("shape") or None,
            strict=self.strict,
        )

    def get(self, name: str, default: ComponentHandle | None = None) -> ComponentHandle | None:
        if name in self._components:
            return self[name]
        return default

    def names(self) -> list[str]:
        return sorted(self._components)
