"""Capture visible Safari trials of the two isolated editor builds."""

import base64
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request


output = Path(".local/terrain-progressive/safari")
output.mkdir(parents=True, exist_ok=True)
driver = subprocess.Popen(
    ["/usr/bin/safaridriver", "--port", "18446"],
    stdout=(output / "driver.log").open("w"),
    stderr=subprocess.STDOUT,
)
session = None


def request(method, path, data=None):
    body = json.dumps(data).encode() if data is not None else None
    request = urllib.request.Request(
        "http://127.0.0.1:18446" + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)["value"]


def command(method, path, data=None):
    return request(method, "/session/" + session + path, data)


def execute(script):
    return command("POST", "/execute/sync", {"script": script, "args": []})


try:
    for attempt in range(50):
        try:
            request("GET", "/status")
            break
        except (OSError, urllib.error.URLError):
            time.sleep(0.1)
    value = request(
        "POST", "/session", {"capabilities": {"alwaysMatch": {"browserName": "safari"}}}
    )
    session = value["sessionId"]
    (output / "session.json").write_text(json.dumps(value, indent=2))
    command("POST", "/window/rect", {"width": 1512, "height": 1000})
    command("POST", "/url", {"url": "http://127.0.0.1:18080/"})
    print("Waiting for the Safari automation window to be visible", flush=True)
    deadline = time.monotonic() + 600
    while not execute("return document.visibilityState === 'visible'"):
        if time.monotonic() > deadline:
            raise RuntimeError("Safari stayed hidden; no timing result is valid")
        time.sleep(0.5)
    for name, port in [("baseline", 18080), ("progressive", 18081)]:
        trial = output / name
        trial.mkdir(exist_ok=True)
        command("POST", "/url", {"url": f"http://127.0.0.1:{port}/?offline=1"})
        samples = []
        for second in range(0, 36, 2):
            remaining = second - execute("return performance.now() / 1000")
            if remaining > 0:
                time.sleep(remaining)
            before = execute("return performance.now()")
            picture = base64.b64decode(command("GET", "/screenshot"))
            after = execute("return performance.now()")
            (trial / f"{second:02d}.png").write_bytes(picture)
            samples.append({"nominal_seconds": second, "before_ms": before, "after_ms": after})
        result = execute("""
return {experiment: window.terrainExperiment, visibility: document.visibilityState,
  boot: window.__eloBoot, logs: window.__eloLogs,
  viewport: {width: innerWidth, height: innerHeight, dpr: devicePixelRatio},
  resources: performance.getEntriesByType('resource').map(r => ({
    name: r.name.split('?')[0], start: r.startTime, end: r.responseEnd,
    duration: r.duration, bytes: r.encodedBodySize}))};
""")
        result["screenshots"] = samples
        (trial / "result.json").write_text(json.dumps(result, indent=2))
        print(f"Completed {name}", flush=True)
finally:
    if session:
        command("DELETE", "")
    driver.terminate()
    driver.wait(timeout=10)
