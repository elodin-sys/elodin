# -*- python -*-
# vi: ft=python

def step(label, command, env = {}, plugins = []):
  return {
    "label": label,
    "command": command,
    "env": env,
    "plugins": plugins
  }

def nix_step(label, flake, command, emoji = ":nix:", env = {}, plugins = []):
  return step(label = f"{emoji} {label}", command = f"nix develop {flake} --command bash -euo pipefail -c \"{command}\"", env = env, plugins = plugins)

def rust_step(label, command, env = {}):
  return nix_step(
    label = label,
    flake = ".#rust",
    emoji = ":crab:",
    command = command,
    plugins = [rust_cache_plugin()],
    env = env,
  )


def group(name, steps = []):
  return {"group": name, "steps": steps}

def pipeline(steps = [], env = {}):
  return {"steps": steps, "env": env}

def rust_cache_plugin():
  return {
    "cache#v0.6.0": {
      "path": "./target",
      "restore": "all",
      "save": "all",
    }
  }

pipeline(steps = [
  group(name = ":crab: rust", steps = [
    rust_step(
      label = "clippy",
      command = "cargo clippy -- -Dwarnings",
    ),
    rust_step(
      label = "cargo test",
      command = "cargo test -- -Z unstable-options --format json --report-time | buildkite-test-collector",
      env = {
        "RUSTC_BOOTSTRAP": "1",
        "BUILDKITE_ANALYTICS_TOKEN": "R6hH2MNhtMdbfQWhDd9cvZfo"
      }
    ),
    nix_step(
      label = "cargo fmt",
      command = "cargo fmt --check",
      flake = ".#rust",
      emoji = ":crab:",
    )
  ])
],
env = {
    "SCCACHE_DIR": "/buildkite/builds/sscache",
    "RUSTC_WRAPPER": "sccache",
    "BUILDKITE_PLUGIN_FS_CACHE_FOLDER": "/buildkite/cache",
})
