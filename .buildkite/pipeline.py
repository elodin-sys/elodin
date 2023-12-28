from buildkite import *
from steps import *
from plugins import *

GKE_CONFIG = {
  "cluster_name": "elodin-dev-gke",
  "project_id": "elodin-dev",
  "region": "us-central1",
}

def generate_cluster_name(branch):
  return f"\$(wordchain random -s {branch})"

def deploy_k8s_step(is_main = False):
  gke_cluster_name = GKE_CONFIG["cluster_name"]
  gke_project_id = GKE_CONFIG["project_id"]
  gke_region = GKE_CONFIG["region"]

  cluster_name = "app" if is_main else generate_cluster_name("\$BUILDKITE_BRANCH")
  overlay_name = "dev" if is_main else "dev-branch"

  command = " && ".join([
    f"export CLUSTER_NAME={cluster_name}",
    f"gcloud container clusters get-credentials {gke_cluster_name} --region {gke_region} --project {gke_project_id}",
    "just decrypt-secrets-force",
    f"kubectl kustomize kubernetes/overlays/{overlay_name} > out.yaml",
    "envsubst < out.yaml > out-with-envs.yaml",
    "kubectl apply -f out-with-envs.yaml",
    "buildkite-agent annotate \"Deployed at https://\$CLUSTER_NAME.elodin.dev\" --style \"success\"",
  ])
  
  return nix_step(
    label = f":kubernetes: deploy {overlay_name} cluster",
    flake = ".#ops",
    command = command,
    condition = f"build.branch {'==' if is_main else '!='} \"main\"",
    env = { "ELO_DECRYPT_SECRETS": "1" },
    plugins = [ gcp_identity_plugin() ]
  )

def cleanup_k8s_step():
  gke_cluster_name = GKE_CONFIG["cluster_name"]
  gke_project_id = GKE_CONFIG["project_id"]
  gke_region = GKE_CONFIG["region"]

  cluster_name = generate_cluster_name("\$PR_CLOSED_BRANCH")

  command = " && ".join([
    f"export CLUSTER_NAME={cluster_name}",
    f"gcloud container clusters get-credentials {gke_cluster_name} --region {gke_region} --project {gke_project_id}",
    "kubectl delete ns elodin-app-\$CLUSTER_NAME",
    "kubectl delete ns elodin-vms-\$CLUSTER_NAME",
  ])
  
  return nix_step(
    label = f":kubernetes: delete dev-branch cluster",
    flake = ".#ops",
    command = command,
    plugins = [ gcp_identity_plugin() ]
  )


pipeline(steps = [
  # native buildkite trigger
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
  group(name = ":docker: docker", key = "build-images", steps = [
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
  ]),
  group(name = ":kubernetes: kubernetes", depends_on = "build-images", steps = [
    deploy_k8s_step(is_main = True),
    deploy_k8s_step(),
  ]),
  # github actions trigger
  group(name = ":kubernetes: kubernetes", is_gha = True, steps = [
    cleanup_k8s_step(),
  ])
],
env = {
  "SCCACHE_DIR": "/buildkite/builds/sscache",
  "RUSTC_WRAPPER": "sccache",
  "BUILDKITE_PLUGIN_FS_CACHE_FOLDER": "/buildkite/cache",
})
