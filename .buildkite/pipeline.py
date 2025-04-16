import os

from buildkite import group, pipeline, step
from steps import build_image_step, c_step, nix_step, rust_step
from utils import codename

AZ_CONFIG = {"cluster_name": "dev", "rg_name": "dev"}

GITHUB_ACTION_TRIGGER = os.getenv("TRIGGERED_FROM_GHA", "") == "1"
BRANCH_NAME = (
    os.environ["PR_CLOSED_BRANCH"] if GITHUB_ACTION_TRIGGER else os.environ["BUILDKITE_BRANCH"]
)


def deploy_k8s_step(branch_name):
    az_cluster_name = AZ_CONFIG["cluster_name"]
    az_rg_name = AZ_CONFIG["rg_name"]

    is_main = branch_name == "main"

    cluster_name = "app" if is_main else codename(branch_name)
    overlay_name = "dev" if is_main else "dev-branch"
    docs_subdomain = "docs" if is_main else f"{cluster_name}-docs"

    annotation_message = (
        f"Deployed at https://{cluster_name}.elodin.dev | https://{docs_subdomain}.elodin.dev"
    )

    command = " && ".join(
        [
            "az login --identity",
            f"az aks get-credentials --resource-group {az_rg_name} --name {az_cluster_name} --overwrite-existing",
            f"kubectl kustomize kubernetes/overlays/{overlay_name} > out.yaml",
            f"export CLUSTER_NAME={cluster_name}",
            "envsubst < out.yaml > out-with-envs.yaml",
            "kubectl apply -f out-with-envs.yaml",
            f'buildkite-agent annotate "{annotation_message}" --style "success"',
        ]
    )

    return nix_step(
        label=f":kubernetes: deploy {overlay_name} cluster",
        flake=".#ops",
        command=command,
        env={"ELO_DECRYPT_SECRETS": "1"},
    )


def cleanup_k8s_step(branch_name):
    return step(
        label=":kubernetes: delete dev-branch cluster",
        command=[f"./justfile clean-dev-branch {codename(branch_name)}"],
    )


test_steps = [
    group(
        name=":c: C",
        steps=[
            c_step(label="db-c-example", command="cd libs/db; cc examples/client.c -lm"),
            c_step(
                label="db-cpp-example",
                command="cd libs/db; c++ -std=c++23 examples/client.cpp",
            ),
        ],
    ),
    group(
        name=":crab: rust",
        steps=[
            rust_step(
                label="clippy",
                command="cargo clippy -- -Dwarnings && cd fsw/sensor-fw && cargo clippy -- -Dwarnings",
            ),
            rust_step(
                label="cargo test",
                command="cargo test --release -- -Z unstable-options --format json --report-time | buildkite-test-collector",
                env={
                    "RUSTC_BOOTSTRAP": "1",
                    "BUILDKITE_ANALYTICS_TOKEN": "R6hH2MNhtMdbfQWhDd9cvZfo",
                },
            ),
            rust_step(
                label="cargo fmt",
                command="cargo fmt --check && cargo fmt --check --manifest-path fsw/sensor-fw/Cargo.toml",
            ),
        ],
    ),
    group(
        name=":black_nib: writing",
        steps=[
            nix_step(
                label="typos",
                flake=".#writing",
                command="typos -c typos.toml",
            ),
        ],
    ),
    nix_step(
        emoji=":python:",
        label="nox-py",
        key="nox-py",
        # this step is just to verify that the package can be imported
        # nix does all the actual work of building nox-py and installing it in the environment
        command="python -c 'import elodin; print(elodin.__version__)'",
        flake=".#python",
    ),
    group(
        name=":python: python",
        depends_on=["nox-py"],
        steps=[
            nix_step(
                label=":python: pytest",
                command="pytest libs/nox-py",
                flake=".#python",
            ),
            nix_step(
                label=":python: lint",
                command="ruff format --check && ruff check",
                flake=".#python",
            ),
        ],
    ),
    group(
        name=":python: examples",
        depends_on=["nox-py"],
        steps=[
            nix_step(
                label=":python: ball",
                command="python3 examples/ball/main.py bench --ticks 100",
                flake=".#python",
            ),
            nix_step(
                label=":python: drone",
                command="python3 examples/drone/main.py bench --ticks 100",
                flake=".#python",
            ),
            nix_step(
                label=":python: rocket",
                command="python3 libs/nox-py/examples/rocket.py bench --ticks 100",
                flake=".#python",
            ),
            nix_step(
                label=":python: three-body",
                command="python3 libs/nox-py/examples/three-body.py bench --ticks 100",
                flake=".#python",
            ),
            nix_step(
                label=":python: cube-sat",
                command="python3 libs/nox-py/examples/cube-sat.py bench --ticks 10",
                flake=".#python",
            ),
        ],
    ),
    nix_step(label="alejandra", flake=".#nix-tools", command="alejandra -c ."),
    step(
        label=":nix: elodin-cli",
        key="elodin-cli",
        command="nix build .#elodin-cli",
    ),
    step(
        label=":nix: aleph-os",
        key="aleph-os",
        command="cd images/aleph; nix build --accept-flake-config .#nixosConfigurations.default.config.system.build.toplevel",
        agents={"queue": "nixos-arm"},
    ),
]

cluster_app_deploy_steps = [
    group(
        name=":docker: docker",
        key="build-images",
        steps=[
            build_image_step(
                image_name="elo-docs",
                target="docs-image",
                image_tag="\$BUILDKITE_COMMIT",
            ),
        ],
    ),
    group(
        name=":kubernetes: kubernetes",
        key="deploy-app",
        depends_on="build-images",
        steps=[deploy_k8s_step(BRANCH_NAME)],
    ),
]

cluster_app_destroy_steps = [
    group(
        name=":kubernetes: kubernetes",
        depends_on=None if GITHUB_ACTION_TRIGGER else "deploy-app",
        steps=[cleanup_k8s_step(BRANCH_NAME)],
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
    steps=pipeline_steps,
    env={
        "BUILDKITE_PLUGIN_FS_CACHE_FOLDER": "/run/buildkite/cache",
    },
)
