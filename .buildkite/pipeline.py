import os

from buildkite import group, pipeline, step
from steps import nix_step

AZ_CONFIG = {"cluster_name": "dev", "rg_name": "dev"}

GITHUB_ACTION_TRIGGER = os.getenv("TRIGGERED_FROM_GHA", "") == "1"
BRANCH_NAME = (
    os.environ["PR_CLOSED_BRANCH"] if GITHUB_ACTION_TRIGGER else os.environ["BUILDKITE_BRANCH"]
)

test_steps = [
    group(
        name=":c: C",
        steps=[
            nix_step(
                emoji=":c:",
                label="db-c-example",
                command="cd libs/db; cc examples/client.c -lm",
            ),
            nix_step(
                emoji=":c:",
                label="db-cpp-example",
                command="cd libs/db; c++ -std=c++23 examples/client.cpp",
            ),
        ],
    ),
    group(
        name=":crab: rust",
        steps=[
            nix_step(
                emoji=":crab:",
                label="clippy",
                command="cargo clippy -- -Dwarnings && cd fsw/sensor-fw && cargo clippy -- -Dwarnings",
            ),
            nix_step(
                emoji=":crab:",
                label="cargo test",
                command="cargo test --release -- -Z unstable-options --format json --report-time | buildkite-test-collector",
                env={
                    "RUSTC_BOOTSTRAP": "1",
                    "BUILDKITE_ANALYTICS_TOKEN": "R6hH2MNhtMdbfQWhDd9cvZfo",
                },
            ),
            nix_step(
                emoji=":crab:",
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
    ),
    group(
        name=":python: python",
        depends_on=["nox-py"],
        steps=[
            nix_step(
                label=":python: pytest",
                command="pytest libs/nox-py",
            ),
            nix_step(
                label=":python: lint",
                command="ruff format --check && ruff check",
            ),
        ],
    ),
    group(
        name=":python: examples",
        depends_on=["nox-py"],
        steps=[
            nix_step(
                label=":python: ball",
                command="python3 examples/ball/main.py bench --ticks 100 --profile",
            ),
            nix_step(
                label=":python: drone",
                command="python3 examples/drone/main.py bench --ticks 100 --profile",
            ),
            nix_step(
                label=":python: rocket",
                command="python3 examples/rocket/main.py bench --ticks 100 --profile",
            ),
            nix_step(
                label=":python: three-body",
                command="python3 examples/three-body/main.py bench --ticks 100 --profile",
            ),
            nix_step(
                label=":python: cube-sat",
                command="python3 examples/cube-sat/main.py bench --ticks 100 --profile",
            ),
        ],
    ),
    nix_step(label="alejandra", command="alejandra -c ."),
    step(
        label=":nix: elodin-cli",
        key="elodin-cli",
        command="nix build .#elodin-cli",
    ),
    group(
        name=":nix: aleph-os",
        steps=[
            step(
                label=":nix: toplevel",
                key="toplevel",
                command=["cd aleph", "nix build --accept-flake-config .#toplevel"],
                agents={"queue": "nixos-arm-aws"},
            ),
            step(
                label=":nix: sdimage",
                key="sdimage",
                command=[
                    "cd aleph",
                    "nix build --accept-flake-config .#sdimage",
                ],
                agents={"queue": "nixos-arm-aws"},
            ),
            step(
                label=":nix: flash-uefi",
                key="flash-uefi",
                command=[
                    "cd aleph",
                    "nix build --accept-flake-config .#flash-uefi",
                ],
            ),
        ],
    ),
]


pipeline_steps = [
    *test_steps,
]

pipeline(
    steps=pipeline_steps,
    env={
        "BUILDKITE_PLUGIN_FS_CACHE_FOLDER": "/run/buildkite/cache",
    },
    agents={
        "queue": "nixos-x86-aws",
    },
)
