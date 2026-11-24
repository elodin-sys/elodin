"""Run a headless rendering smoke test. These are not Safari performance results."""

import argparse
import json
import os
from pathlib import Path
import platform

from playwright.sync_api import sync_playwright


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("name", choices=["baseline", "progressive"])
parser.add_argument("--chrome", default=os.environ.get("ELODIN_EDITOR_CHROME"))
args = parser.parse_args()
port = 18080 if args.name == "baseline" else 18081
output = Path(".local/terrain-progressive/chromium") / args.name
output.mkdir(parents=True, exist_ok=True)
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(
        headless=True,
        executable_path=args.chrome,
        args=["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"],
    )
    page = browser.new_page(viewport={"width": 1512, "height": 1000}, device_scale_factor=1)
    logs = []
    errors = []
    failed_requests = []
    page.on("console", lambda message: logs.append({"type": message.type, "text": message.text}))
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on(
        "response",
        lambda response: (
            failed_requests.append({"status": response.status, "url": response.url})
            if response.status >= 400
            else None
        ),
    )
    page.goto(f"http://127.0.0.1:{port}/?offline=1", wait_until="domcontentloaded")
    samples = []
    for second in range(0, 41, 2):
        remaining = second * 1000 - page.evaluate("performance.now()")
        if remaining > 0:
            page.wait_for_timeout(remaining)
        before = page.evaluate("performance.now()")
        page.screenshot(path=str(output / f"{second:02d}.png"))
        after = page.evaluate("performance.now()")
        samples.append({"nominal_seconds": second, "before_ms": before, "after_ms": after})
    result = page.evaluate("""() => ({
        experiment: window.terrainExperiment, visibility: document.visibilityState,
        boot: window.__eloBoot, viewport: {width: innerWidth, height: innerHeight, dpr: devicePixelRatio},
        webgpu: !!navigator.gpu,
        resources: performance.getEntriesByType('resource').map(r => ({
            name: r.name.split('?')[0], start: r.startTime, end: r.responseEnd,
            duration: r.duration, bytes: r.encodedBodySize}))
    })""")
    result.update(
        browser=browser.version,
        architecture=platform.machine(),
        screenshots=samples,
        logs=logs,
        page_errors=errors,
        failed_requests=failed_requests,
    )
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"trial": args.name, "browser": browser.version, "errors": errors}))
    assert result["experiment"]["ready"] is not None, "The editor did not finish starting"
    assert not errors, errors
    browser.close()
