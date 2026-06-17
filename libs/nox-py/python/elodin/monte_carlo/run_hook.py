"""Run user-defined Monte Carlo lifecycle hooks."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_hook(path: Path):
    spec = importlib.util.spec_from_file_location("elodin_user_monte_carlo_hook", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load hook: {path}")
    module = importlib.util.module_from_spec(spec)
    # Make the hook's own directory importable so a hook can `import` sibling
    # helper modules (e.g. a shared `mc_metrics.py`) regardless of the cwd.
    hook_dir = str(path.resolve().parent)
    if hook_dir not in sys.path:
        sys.path.insert(0, hook_dir)
    spec.loader.exec_module(module)
    return module


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: python -m elodin.monte_carlo.run_hook HOOK.py post_run|post_campaign CTX.json"
        )
    hook_path = Path(sys.argv[1])
    hook_name = sys.argv[2]
    context_path = Path(sys.argv[3])
    payload = json.loads(context_path.read_text())
    module = _load_hook(hook_path)
    hook = getattr(module, hook_name, None)
    if hook is None:
        # The hook was explicitly configured, so a missing function is a
        # misconfiguration: fail loudly instead of silently skipping scoring
        # (a skipped post_run leaves scored_pass unset, which counts as passed).
        raise SystemExit(f"hook {hook_path} does not define a `{hook_name}` function")
    if not callable(hook):
        raise SystemExit(f"hook {hook_path} attribute `{hook_name}` is not callable")
    result = hook(_namespace(payload))
    if result is not None:
        output = context_path.with_name(f"{hook_name}_result.json")
        output.write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
