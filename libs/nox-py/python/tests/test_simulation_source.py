import importlib
import json
import sys

import elodin as el


def test_capture_simulation_source_records_imported_project_modules(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("import config\nimport sim\n", encoding="utf-8")
    (project / "config.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project / "sim.py").write_text("from config import VALUE\n", encoding="utf-8")
    db_path = tmp_path / "db"

    sys.path.insert(0, str(project))
    try:
        importlib.import_module("config")
        importlib.import_module("sim")
        el._capture_simulation_source(str(db_path), str(project / "main.py"))
    finally:
        sys.path.remove(str(project))
        sys.modules.pop("config", None)
        sys.modules.pop("sim", None)

    source_root = db_path / "simulation_source"
    manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["entrypoint"] == "main.py"
    assert [file["path"] for file in manifest["files"]] == [
        "config.py",
        "main.py",
        "sim.py",
    ]
    assert (source_root / "files" / "config.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert (source_root / "files" / "main.py").read_text(encoding="utf-8") == (
        "import config\nimport sim\n"
    )
    assert (source_root / "files" / "sim.py").read_text(encoding="utf-8") == (
        "from config import VALUE\n"
    )
