from buildkite import step, group
from plugins import *

def build_image_step(image_name, target, image_tag = "latest", repository = "us-central1-docker.pkg.dev"):
  remote_image_path = f"{repository}/elodin-infra/{image_name}/x86_64:{image_tag}"

  pre_command = " && ".join([
    f"export IMAGE_PATH=\$(nix build .#packages.x86_64-linux.{target} --print-out-paths)",
  ])

  command = " && ".join([
    f"gcloud --quiet auth configure-docker {repository}",      
    f"skopeo --insecure-policy copy docker-archive:\$IMAGE_PATH docker://{remote_image_path}",
  ])

  return nix_step(
    label = f":docker: {image_name}",
    flake = ".#ops",
    pre_command = pre_command,
    command = command,
    plugins = [ gcp_identity_plugin() ]
  )

def nix_step(label, flake, command, emoji = ":nix:", pre_command = None, key = None, depends_on = None, env = {}, plugins = []):
  return step(
    label = f"{emoji} {label}",
    command = [
      pre_command,
      f"nix develop {flake} --command bash -euo pipefail -c '{command}'",
    ],
    key = key,
    depends_on = depends_on,
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
