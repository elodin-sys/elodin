from buildkite import step, group
from plugins import *

def build_image_step(image_name, service_path, image_tag = "latest", repository = "us-central1-docker.pkg.dev"):
  remote_image_name = f"{repository}/elodin-infra/{image_name}/x86_64"

  return step(
    label = f":docker: {image_name}",
    command = [
      # NOTE: Build image
      f"cd {service_path}",
      "nix flake update",
      "nix build .#packages.x86_64-linux.docker.image",
      "cat result | docker load",
      f"docker tag {image_name}:latest {remote_image_name}:{image_tag}",
      # NOTE: Login into GCP (active for 10min)
      gcp_identity()["cmds"]["regenerate_token"],
      gcp_identity()["cmds"]["login"],
      # NOTE: Push image
      f"gcloud --quiet auth configure-docker {repository}",
      f"docker push {remote_image_name}:{image_tag}",
    ],
    env = {
      "GCP_WIF_AUDIENCE": gcp_identity()["audience"],
    },
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
