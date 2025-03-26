from buildkite import step


def build_image_step(image_name, target, image_tag="latest", repository="elodin.azurecr.io"):
    remote_image_path = f"{repository}/elodin-infra/{image_name}/x86_64:{image_tag}"

    pre_command = " && ".join(
        [
            f"export IMAGE_PATH=\$(nix build .#packages.x86_64-linux.{target} --print-out-paths)",
        ]
    )

    command = " && ".join(
        [
            "az login --identity",
            "az acr login --name elodin --expose-token --output tsv --query accessToken | skopeo login elodin.azurecr.io --password-stdin --username 00000000-0000-0000-0000-000000000000",
            f"skopeo --insecure-policy copy docker-archive:\$IMAGE_PATH docker://{remote_image_path}",
        ]
    )

    return nix_step(
        label=f":docker: {image_name}",
        flake=".#ops",
        pre_command=pre_command,
        command=command,
        env={"REGISTRY_AUTH_FILE": "./auth.json"},
    )


def nix_step(
    label,
    flake,
    command,
    emoji=":nix:",
    pre_command=None,
    key=None,
    depends_on=None,
    env={},
    plugins=[],
    agents={},
):
    return step(
        label=f"{emoji} {label}",
        command=[
            pre_command,
            f"nix develop {flake} --command bash -euo pipefail -c '{command}'",
        ],
        key=key,
        depends_on=depends_on,
        env=env,
        plugins=plugins,
        agents=agents,
    )


def rust_step(label, command, env={}, emoji=":crab:"):
    return nix_step(
        label=label,
        flake=".#rust",
        emoji=emoji,
        command=command,
        env=env,
    )


def c_step(label, command, env={}, emoji=":c:"):
    return nix_step(
        label=label,
        flake=".#c",
        emoji=emoji,
        command=command,
        env=env,
    )
