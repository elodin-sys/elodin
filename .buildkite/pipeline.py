from buildkite import *
from steps import *
from plugins import *

def deploy_k8s_step(label, key = None, depends_on = None):
  gke_cluster_name = "elodin-dev-development-gke"
  gke_region = "us-central1"
  
  return step(
    label = label,
    key = key,
    depends_on = depends_on,
    command = [
      # NOTE: Prepare environment
      "export GCP_WIF_TMPDIR=$(mktemp -d -t 'buildkiteXXXX')",
      "export GOOGLE_APPLICATION_CREDENTIALS=\$GCP_WIF_TMPDIR/credentials.json",
      "envsubst < .buildkite/credentials.template.json > \$GOOGLE_APPLICATION_CREDENTIALS",
      # NOTE: Login into GCP (active for 10min)
      "buildkite-agent oidc request-token --audience \"\$GCP_WIF_AUDIENCE\" > \$GCP_WIF_TMPDIR/token.json",
      "gcloud --quiet auth login --cred-file=\$GOOGLE_APPLICATION_CREDENTIALS",
      # NOTE: Deploy cluster
      f"gcloud container clusters get-credentials {gke_cluster_name} --region {gke_region}",
      "just decrypt-secrets-force",
      "kubectl kustomize kubernetes/overlays/dev > out.yaml",
      "envsubst < out.yaml > out-with-envs.yaml",
      "kubectl apply -f out-with-envs.yaml",
      # NOTE: Clean up
      "rm -rf \$GCP_WIF_TMPDIR",
    ],
    env = {
      "GCP_WIF_AUDIENCE": "//iam.googleapis.com/projects/802981626435/locations/global/workloadIdentityPools/buildkite-pipeline/providers/buildkite",
      "GCP_WIF_SERVICE_ACCOUNT": "buildkite-802981626435@elodin-dev.iam.gserviceaccount.com",
    },
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
        service_path = "services/paracosm-web-runner",
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
