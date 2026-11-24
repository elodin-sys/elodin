"""Serve an isolated browser build against a local DB copy. Run from the repo root."""

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("build", type=Path)
parser.add_argument("--port", type=int, default=18081)
parser.add_argument("--db", default="http://127.0.0.1:18441")
args = parser.parse_args()
build = args.build.resolve()
page = (
    (build / "index.html")
    .read_text()
    .replace('name="elodin-web-db" content=""', f'name="elodin-web-db" content="{args.db}"')
)
# Keep the existing Safari shader spelling workaround identical in both builds.
probe = r"""<script>
if (globalThis.GPUDevice) {
  const create = GPUDevice.prototype.createShaderModule;
  GPUDevice.prototype.createShaderModule = function(d) {
    return create.call(this, {...d, code: d.code.replace(
      /\b34028235000000000000000000000000000000f\b/g, '3.4028235e37f')});
  };
}
performance.setResourceTimingBufferSize(20000);
window.terrainExperiment = {ready: null, gaps: [], errors: []};
addEventListener('elodin-ready', () => terrainExperiment.ready = performance.now());
addEventListener('error', e => terrainExperiment.errors.push(e.message));
addEventListener('unhandledrejection', e => terrainExperiment.errors.push(String(e.reason)));
let previous = performance.now();
function frame(t) {
  if (t - previous > 100) terrainExperiment.gaps.push({start: previous, duration: t - previous});
  previous = t;
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script>"""
page = page.replace("<head>", "<head>" + probe).encode()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *positional, **keyword):
        super().__init__(*positional, directory=str(build), **keyword)

    def do_GET(self):
        if self.path.split("?", 1)[0] in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(page)
        else:
            super().do_GET()


print(f"Editor: http://127.0.0.1:{args.port}/?offline=1", flush=True)
ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
