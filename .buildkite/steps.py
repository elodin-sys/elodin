from buildkite import step, group
from plugins import *

def build_image_step(image_name, service_path, image_tag = "latest", repository = "us-central1-docker.pkg.dev"):
  remote_image_path = f"{repository}/elodin-infra/{image_name}/x86_64:{image_tag}"

  return step(
    label = f":docker: {image_name}",
    command = [
      f"cd {service_path}",
      "nix flake update",
      f"gcloud --quiet auth configure-docker {repository}",      
      "export IMAGE_PATH=\$(nix build .#packages.x86_64-linux.docker.image --print-out-paths)",
      f"skopeo --insecure-policy copy docker-archive:\$IMAGE_PATH docker://{remote_image_path}",
    ],
    plugins = [ gcp_identity_plugin() ]
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
