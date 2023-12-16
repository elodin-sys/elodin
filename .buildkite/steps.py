from buildkite import step, group
from plugins import *

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
