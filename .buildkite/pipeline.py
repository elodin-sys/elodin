from buildkite import *
from steps import *
from plugins import *

def deploy_k8s_step(label, key = None, depends_on = None):
  gke_cluster_name = "elodin-dev-gke"
  gke_project_id = "elodin-dev"
  gke_region = "us-central1"
  
  return step(
    label = label,
    key = key,
    depends_on = depends_on,
    command = [
      f"gcloud container clusters get-credentials {gke_cluster_name} --region {gke_region} --project {gke_project_id}",
      "just decrypt-secrets-force",
      "kubectl kustomize kubernetes/overlays/dev > out.yaml",
      "envsubst < out.yaml > out-with-envs.yaml",
      "kubectl apply -f out-with-envs.yaml",
    ],
    plugins = [ gcp_identity_plugin() ]
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
  group(name = ":elixir: elixir", steps = [
    nix_step(
      label = "elixir build",
      flake = ".#elixir",
      emoji = ":elixir:",
      command = "cd services/dashboard && mix deps.get && mix deps.unlock --check-unused && mix compile --all-warnings --warning-as-errors && mix format --dry-run --check-formatted",
      plugins = [elixir_cache_plugin()],
    ),
    nix_step(
      label = "dialyzer",
      flake = ".#elixir",
      emoji = ":elixir:",
      command = "cd services/dashboard && mix deps.get && mix dialyzer --plt && mix dialyzer",
      plugins = [elixir_cache_plugin()],
    )
  ]),
  group(
    name = ":docker: docker",
    key = "build-images",
    steps = [
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
        service_path = "services/elodin-web-runner",
        image_tag = "\$BUILDKITE_COMMIT",
      ),
    ],
  ),
  deploy_k8s_step(
    label = ":kubernetes: deploy dev cluster",
    key = "kubernetes-deploy",
    depends_on = "build-images",
  ),
],
env = {
  "SCCACHE_DIR": "/buildkite/builds/sscache",
  "RUSTC_WRAPPER": "sccache",
  "BUILDKITE_PLUGIN_FS_CACHE_FOLDER": "/buildkite/cache",
})
