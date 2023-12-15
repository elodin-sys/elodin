# -*- python -*-
# vi: ft=python

################################################################################
# plugins
################################################################################

def rust_cache_plugin():
  return {
    "cache#v0.6.0": {
      "path": "./target",
      "manifest": "./Cargo.lock",
      "restore": "file",
      "save": "file",
    }
  }

def gcp_identity_plugin():
  return {
    "gcp-workload-identity-federation#v1.0.0": {
      "audience": "//iam.googleapis.com/projects/802981626435/locations/global/workloadIdentityPools/buildkite-pipeline/providers/buildkite",
      "service-account": "buildkite-802981626435@elodin-dev.iam.gserviceaccount.com",
      # TODO: Not working as intended, needs more investigation
      # "lifetime": 2 * 60 * 60,
    }
  }

################################################################################
# steps
################################################################################

def step(label, command, env = {}, plugins = [], skip = False):
  return {
    "label": label,
    "command": command,
    "env": env,
    "plugins": plugins,
    "skip": skip,
  }

def build_image_step(image_name, service_path, image_tag = "latest", repository = "us-central1-docker.pkg.dev"):
  remote_image_name = f"{repository}/elodin-dev/{image_name}/x86_64"

  return step(
    label = f":docker: build {image_name}",
    command = [
      f"cd {service_path}",
      "nix flake update",
      "nix build .#packages.x86_64-linux.docker.image",
      f"cat result | docker load",
      f"docker tag {image_name}:latest {remote_image_name}:{image_tag}",
      f"gcloud --quiet auth configure-docker {repository}",
      f"docker push {remote_image_name}:{image_tag}",
    ],
    env = {},
    plugins = [ gcp_identity_plugin() ],
    skip = "Disabled until cluster is included in this flow",
  )

def nix_step(label, flake, command, emoji = ":nix:", env = {}, plugins = []):
  return step(
    label = f"{emoji} {label}",
    command = f"nix develop {flake} --command bash -euo pipefail -c \"{command}\"",
    env = env,
    plugins = plugins,
  )

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
    ),
  ]),
  group(name = ":docker: docker", steps = [
    build_image_step(image_name = "elo-dashboard", service_path = "services/dashboard"),
    build_image_step(image_name = "elo-atc", service_path = "services/atc"),
    build_image_step(image_name = "elo-sim-runner", service_path = "services/paracosm-web-runner")
  ])
],
env = {
  "SCCACHE_DIR": "/buildkite/builds/sscache",
  "RUSTC_WRAPPER": "sccache",
  "BUILDKITE_PLUGIN_FS_CACHE_FOLDER": "/buildkite/cache",
})
