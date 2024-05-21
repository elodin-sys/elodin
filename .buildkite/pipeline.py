import os

from buildkite import *
from steps import *
from plugins import *
from utils import *


GKE_CONFIG = {
  "cluster_name": "elodin-dev-gke",
  "project_id": "elodin-dev",
  "region": "us-central1",
}

GITHUB_ACTION_TRIGGER = os.getenv("TRIGGERED_FROM_GHA", "") == "1"
BRANCH_NAME = os.environ["PR_CLOSED_BRANCH"] if GITHUB_ACTION_TRIGGER else os.environ["BUILDKITE_BRANCH"]


def deploy_k8s_step(branch_name):
  gke_cluster_name = GKE_CONFIG["cluster_name"]
  gke_project_id = GKE_CONFIG["project_id"]
  gke_region = GKE_CONFIG["region"]

  is_main = branch_name == "main"

  cluster_name = "app" if is_main else codename(branch_name)
  overlay_name = "dev" if is_main else "dev-branch"
  docs_subdomain = "docs" if is_main else f"{cluster_name}-docs"

  annotation_message = f"Deployed at https://{cluster_name}.elodin.dev | https://{docs_subdomain}.elodin.dev"

  command = " && ".join([
    f"gcloud container clusters get-credentials {gke_cluster_name} --region {gke_region} --project {gke_project_id}",
    "just decrypt-secrets-force",
    f"kubectl kustomize kubernetes/overlays/{overlay_name} > out.yaml",
    f"export CLUSTER_NAME={cluster_name}",
    "envsubst < out.yaml > out-with-envs.yaml",
    "kubectl apply -f out-with-envs.yaml",
    f"buildkite-agent annotate \"{annotation_message}\" --style \"success\"",
  ])
  
  return nix_step(
    label = f":kubernetes: deploy {overlay_name} cluster",
    flake = ".#ops",
    command = command,
    env = { "ELO_DECRYPT_SECRETS": "1" },
    plugins = [ gcp_identity_plugin() ]
  )

def cleanup_k8s_step(branch_name):
  return step(
    label = f":kubernetes: delete dev-branch cluster",
    command = [
      f"./justfile clean-dev-branch {codename(branch_name)}"
    ],
    plugins = [ gcp_identity_plugin() ]
  )


test_steps = [
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
  group(name = ":python: python", steps = [
    nix_step(
      label = ":python: pytest",
      command = "pytest libs/nox-py",
      flake = ".#python",
    ),
    nix_step(
      label = ":python: lint",
      command = "ruff format --check libs/nox-py",
      flake = ".#python",
    ),
    # rust_step(
    #   label = "mypy", emoji = ":python:",
    #   command = "cd libs/nox-py && python3 -m venv .venv && .venv/bin/pip install mypy && .venv/bin/mypy . --check-untyped-defs",
    # ), # TODO(sphw): replace with pyright
  ]),
  group(name = ":elixir: elixir", steps = [
    nix_step(
      label = "elixir build",
      flake = ".#elixir",
      emoji = ":elixir:",
      command = "cd services/dashboard && mix deps.get && mix deps.unlock --check-unused && mix compile --all-warnings --warning-as-errors && mix format --dry-run --check-formatted",
    ),
    nix_step(
      label = "dialyzer",
      flake = ".#elixir",
      emoji = ":elixir:",
      command = "cd services/dashboard && mix deps.get && mix dialyzer --plt && mix dialyzer",
    )
  ]),
  group(name = ":node: node", steps = [
    nix_step(
      label = "mintlify broken-links",
      flake = ".#node",
      emoji = ":node:",
      command = "cd docs/public && npx mintlify broken-links",
    ),
  ]),
]

cluster_app_deploy_steps = [
  group(
    name = ":docker: docker",
    key = "build-images",
    steps = [
      build_image_step(
        image_name = "elo-dashboard",
        target = "dashboard-image",
        image_tag = "\$BUILDKITE_COMMIT",
      ),
      build_image_step(
        image_name = "elo-atc",
        target = "atc-image",
        image_tag = "\$BUILDKITE_COMMIT",
      ),
      build_image_step(
        image_name = "elo-sim-agent",
        target = "sim-agent-image",
        image_tag = "\$BUILDKITE_COMMIT",
      ),
      build_image_step(
        image_name = "elo-docs",
        target = "docs-image",
        image_tag = "\$BUILDKITE_COMMIT",
      ),
    ]
  ),
  group(
    name = ":kubernetes: kubernetes",
    key = "deploy-app",
    depends_on = "build-images",
    steps = [ deploy_k8s_step(BRANCH_NAME) ]
  )
]

cluster_app_destroy_steps = [
  group(
    name = ":kubernetes: kubernetes",
    depends_on = None if GITHUB_ACTION_TRIGGER else "deploy-app",
    steps = [ cleanup_k8s_step(BRANCH_NAME) ]
  )
]


pipeline_steps = []

if GITHUB_ACTION_TRIGGER:
  # PR is closed and app needs to be deleted
  pipeline_steps = cluster_app_destroy_steps
elif BRANCH_NAME.startswith("gh-readonly-queue"):
  # GitHub Queue branch - run everything and clean up right away
  pipeline_steps = [
    *test_steps,
    *cluster_app_deploy_steps,
    *cluster_app_destroy_steps,
  ]
else:
  # Run tests and deploy app
  pipeline_steps = [
    *test_steps,
    *cluster_app_deploy_steps,
  ]

pipeline(
  steps = pipeline_steps,
  env = {
    "SCCACHE_DIR": "/buildkite/builds/sscache",
    "RUSTC_WRAPPER": "sccache",
    "BUILDKITE_PLUGIN_FS_CACHE_FOLDER": "/buildkite/cache",
  }
)
