"""Monte Carlo campaign helpers for Elodin."""

from __future__ import annotations

from elodin.elodin import monte_carlo as _native

Param = _native.Param
Params = _native.Params
ParamsSpec = _native.ParamsSpec
params_spec = _native.params_spec
params = _native.params
result = _native.result
spec_json = _native.spec_json

__all__ = [
    "Param",
    "Params",
    "ParamsSpec",
    "params_spec",
    "params",
    "result",
    "spec_json",
]
