from buildkite import step, group
from plugins import *

def build_image_step(image_name, service_path, image_tag = "latest", repository = "us-central1-docker.pkg.dev"):
  remote_image_name = f"{repository}/elodin-dev/{image_name}/x86_64"

  return step(
    label = f":docker: {image_name}",
    command = [
      # NOTE: Prepare environment
      "export GCP_WIF_TMPDIR=$(mktemp -d -t 'buildkiteXXXX')",
      "export GOOGLE_APPLICATION_CREDENTIALS=\$GCP_WIF_TMPDIR/credentials.json",
      "envsubst < .buildkite/credentials.template.json > \$GOOGLE_APPLICATION_CREDENTIALS",
      # NOTE: Build image
      f"cd {service_path}",
      "nix flake update",
      "nix build .#packages.x86_64-linux.docker.image",
      "cat result | docker load",
      f"docker tag {image_name}:latest {remote_image_name}:{image_tag}",
      # NOTE: Login into GCP (active for 10min)
      "buildkite-agent oidc request-token --audience \"\$GCP_WIF_AUDIENCE\" > \$GCP_WIF_TMPDIR/token.json",
      "gcloud --quiet auth login --cred-file=\$GOOGLE_APPLICATION_CREDENTIALS",
      # NOTE: Push image
      f"gcloud --quiet auth configure-docker {repository}",
      f"docker push {remote_image_name}:{image_tag}",
      # NOTE: Clean up
      "rm -rf \$GCP_WIF_TMPDIR",
    ],
    env = {
      "GCP_WIF_AUDIENCE": "//iam.googleapis.com/projects/802981626435/locations/global/workloadIdentityPools/buildkite-pipeline/providers/buildkite",
      "GCP_WIF_SERVICE_ACCOUNT": "buildkite-802981626435@elodin-dev.iam.gserviceaccount.com",
    },
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
    env = env,
  )
