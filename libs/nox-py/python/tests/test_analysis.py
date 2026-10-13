"""elodin.analysis against the synthetic fixture (libs/db/src/fixture.rs)."""

import inspect
import json
import re
from pathlib import Path

import pyarrow as pa
import pytest
from elodin import analysis as el
from elodin.analysis import _worker
from elodin.elodin import db as native

SDK_REFERENCE = Path(__file__).parents[3] / "elodin-mcp" / "src" / "sdk_reference.json"


@pytest.fixture(scope="module")
def recordings(tmp_path_factory):
    root = tmp_path_factory.mktemp("recordings")
    for name in ("flight", "sim"):
        native.Recording.write_fixture(str(root / name))
    return root


@pytest.fixture
def flight(recordings, tmp_path):
    el.open_local(str(recordings / "flight"), out_dir=str(tmp_path))
    return tmp_path


def execute(recordings, out, code, parameters=None, names=("flight",)):
    sources = [{"name": n, "path": str(recordings / n)} for n in names]
    spec = {"code": code, "parameters": parameters or {}, "execution_id": "x-test"}
    return _worker.run(spec, el._LocalBackend(sources), Path(out))


def test_signals_and_tables(flight):
    df = el.signal("IMU.TEMP").to_polars()
    assert df.columns == ["time_us", "value"]
    assert df.height == 6001
    gps = el.table("gps").to_polars()
    assert gps.columns == ["time_us", "alt", "count"]
    assert gps.height == 900
    assert el.signal("gps.count").count() == 300
    assert el.table("gps").select("count").to_polars().columns == ["time_us", "count"]
    assert {t["name"] for t in el.tables() if t["rows"]} == {"baro", "gps", "imu"}
    assert [s["name"] for s in el.signals("accel")] == ["imu.accel"]


def test_vectors_and_windows(flight):
    accel = el.signal("imu.accel")
    assert accel.to_arrow().schema.field("value").type == pa.list_(pa.float64(), 3)
    assert accel.to_polars(flatten=True).columns == ["time_us", "value_x", "value_y", "value_z"]
    start = 1_000_000 + 1000 * 10_000
    window = accel.window(start_us=start, end_us=start + 20_000)
    # The repeated timestamp keeps both samples (today's --join behavior).
    assert window.to_polars()["time_us"].to_list() == [start, start, start + 10_000]
    assert window.window(end_us=start + 1).count() == 2


def test_budget_errors_state_the_size(flight):  # QRY-02
    with pytest.raises(el.ResultBudgetError, match="6,001 rows, over max_rows=10"):
        el.signal("imu.temp").to_polars(max_rows=10)


def test_unknown_names_suggest(flight, recordings, tmp_path):  # MCP-03
    with pytest.raises(el.UnknownSignal) as e:
        el.signal("imu.acel")
    assert e.value.did_you_mean[0] == {"name": "imu.accel", "runs": {"flight": "populated"}}
    env = execute(recordings, tmp_path / "x", "el.signal('imu.acel')")
    assert env["status"] == "error"
    assert env["error"]["type"] == "UnknownSignal"
    assert env["error"]["line"] == 1
    assert env["error"]["did_you_mean"][0]["name"] == "imu.accel"


def test_private_components_do_not_exist(flight):
    with pytest.raises(el.UnknownSignal):
        el.signal("secret.key")
    assert not el.signals("secret")


def test_ambiguous_run(recordings, tmp_path):
    el.open_local(str(recordings / "flight"), str(recordings / "sim"))
    with pytest.raises(el.AmbiguousRun):
        el.signal("imu.temp")
    assert el.signal("imu.temp", run="sim").count() == 6001


def test_result_preview_and_logs(recordings, tmp_path):  # MCP-02
    code = "print('x' * 100)\nresult = {'head': ctx.preview(el.signal('imu.temp').to_polars(), n=1000)}"
    env = execute(recordings, tmp_path, code)
    assert env["status"] == "completed", env["error"]
    assert len(env["result"]["head"]) == 100
    assert "xxx" not in json.dumps(env)
    assert [a["name"] for a in env["artifacts"]] == ["stdout.txt"]
    assert env["usage"]["rows_read"] == 6001

    big = execute(recordings, tmp_path / "big", "result = {'s': 'x' * 20000}")
    assert big["status"] == "error" and "16,384" in big["error"]["message"]
    none = execute(recordings, tmp_path / "none", "x = 1")
    assert none["error"]["type"] == "NoResult"


def test_reproduction(recordings, tmp_path):  # MCP-05
    code = (
        "df = el.signal(ctx.parameters['signal']).to_polars()\n"
        "result = {'max': df['value'].max(), 'table': el.artifacts.save_table(df, 'out')}"
    )
    a = execute(recordings, tmp_path / "a", code, {"signal": "baro.pressure"})
    b = execute(recordings, tmp_path / "b", code, {"signal": "baro.pressure"})
    assert a["status"] == "completed", a["error"]
    assert a["result"] == b["result"]
    assert [x["sha256"] for x in a["artifacts"]] == [x["sha256"] for x in b["artifacts"]]


def test_sdk_reference_examples(recordings, tmp_path):
    entries = json.loads(SDK_REFERENCE.read_text())
    for i, entry in enumerate(entries):
        env = execute(recordings, tmp_path / str(i), entry["example"], {"signal": "imu.temp"})
        assert env["status"] == "completed", (entry["name"], env["error"])
        target = entry["name"].replace("el.", "", 1)
        if target.startswith("Table."):
            fn = getattr(el.Table, target.split(".", 1)[1])
        elif target.startswith("artifacts."):
            fn = getattr(el.artifacts, target.split(".", 1)[1])
        elif hasattr(el, target):
            fn = getattr(el, target)
        else:
            continue
        sig = re.search(r"\((.*)\)", entry["signature"].split(" / ")[0]).group(1)
        params = [p for p in inspect.signature(fn).parameters if p != "self"]
        named = [p.split("=")[0].strip().lstrip("*") for p in sig.split(",") if p.strip()]
        assert named == params, (entry["name"], named, params)
