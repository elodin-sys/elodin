from buildkite import step, group
from plugins import *

def build_image_step(image_name, service_path, image_tag = "latest", repository = "us-central1-docker.pkg.dev"):
  remote_image_path = f"{repository}/elodin-infra/{image_name}/x86_64:{image_tag}"

  pre_command = " && ".join([
    f"pushd {service_path}",
    "nix flake update",
    "export IMAGE_PATH=\$(nix build .#packages.x86_64-linux.docker.image --print-out-paths)",
    "popd",
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

def nix_step(label, flake, command, emoji = ":nix:", pre_command = None, key = None, depends_on = None, condition = None, env = {}, plugins = []):
  return step(
    label = f"{emoji} {label}",
    command = [
      pre_command,
      f"echo '{command}' > script.sh",
      "chmod a+x script.sh",
      f"nix develop {flake} --command bash -euo pipefail -c ./script.sh",
      "rm ./script.sh",
    ],
    key = key,
    depends_on = depends_on,
    condition = condition,
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
