import json

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
    }
  }

################################################################################
# generic blocks
################################################################################

def step(label, command, key = None, depends_on = None, env = {}, plugins = [], skip = False):
  return {
    "label": label,
    "command": command,
    "key": key,
    "depends_on": depends_on,
    "env": env,
    "plugins": plugins,
    "skip": skip,
  }

def group(name, key = None, depends_on = None, steps = []):
  return {
    "group": name,
    "key": key,
    "depends_on": depends_on,
    "steps": steps,
  }

def pipeline(steps = [], env = {}):
  return print(json.dumps({"steps": steps, "env": env}))

################################################################################
# steps
################################################################################

def build_image_step(image_name, service_path, image_tag = "latest", repository = "us-central1-docker.pkg.dev"):
  remote_image_name = f"{repository}/elodin-dev/{image_name}/x86_64"

  return group(
    name = f":docker: {image_name}",
    key = image_name,
    steps = [
      step(
        label = f":docker: build {image_name}",
        key = f"build-{image_name}",
        command = [
          f"cd {service_path}",
          "nix flake update",
          "nix build .#packages.x86_64-linux.docker.image",
          "cat result | docker load",
          f"docker tag {image_name}:latest {remote_image_name}:{image_tag}",
        ],
      ),
      # NOTE: Workload auth is only active for about 10min so we need to trigger plugin when image is ready to be pushed
      step(
        label = f":docker: push {image_name}",
        depends_on = f"build-{image_name}",
        command = [
          "gcloud --quiet auth login --cred-file=\$GOOGLE_APPLICATION_CREDENTIALS",
          f"gcloud --quiet auth configure-docker {repository}",
          f"docker push {remote_image_name}:{image_tag}",
        ],
        plugins = [ gcp_identity_plugin() ],
      ),
    ]
  )

def deploy_k8s_step(label, key = None, depends_on = None):
  gke_cluster_name = "elodin-dev-development-gke"
  gke_region = "us-central1"
  
  return step(
    label = label,
    key = key,
    depends_on = depends_on,
    command = [
      f"gcloud container clusters get-credentials {gke_cluster_name} --region {gke_region}",
      "just decrypt-secrets-force",
      "kubectl kustomize kubernetes/overlays/dev > out.yaml",
      "envsubst < out.yaml > out-with-envs.yaml",
      "kubectl apply -f out-with-envs.yaml",
    ],
    env = {},
    plugins = [ gcp_identity_plugin() ],
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
  build_image_step(
    image_name = "elo-dashboard",
    service_path = "services/dashboard",
    image_tag = "\$BUILDKITE_COMMIT",
  ),
  build_image_step(
    image_name = "elo-atc",
    service_path = "services/atc",
    image_tag = "\$BUILDKITE_COMMIT",
  ),
  build_image_step(
    image_name = "elo-sim-runner",
    service_path = "services/paracosm-web-runner",
    image_tag = "\$BUILDKITE_COMMIT",
  ),
  deploy_k8s_step(
    label = ":kubernetes: deploy dev cluster",
    key = "kubernetes-deploy",
    depends_on = [
      "elo-dashboard",
      "elo-atc",
      "elo-sim-runner",
    ],
  ),
],
env = {
  "SCCACHE_DIR": "/buildkite/builds/sscache",
  "RUSTC_WRAPPER": "sccache",
  "BUILDKITE_PLUGIN_FS_CACHE_FOLDER": "/buildkite/cache",
})
