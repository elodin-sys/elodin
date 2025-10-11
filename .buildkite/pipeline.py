import os

from buildkite import group, pipeline, step
from steps import c_step, nix_step, rust_step

AZ_CONFIG = {"cluster_name": "dev", "rg_name": "dev"}

GITHUB_ACTION_TRIGGER = os.getenv("TRIGGERED_FROM_GHA", "") == "1"
BRANCH_NAME = (
    os.environ["PR_CLOSED_BRANCH"] if GITHUB_ACTION_TRIGGER else os.environ["BUILDKITE_BRANCH"]
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
                flake=".#elodin",
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
        flake=".#elodin",
    ),
    group(
        name=":python: python",
        depends_on=["nox-py"],
        steps=[
            nix_step(
                label=":python: pytest",
                command="pytest libs/nox-py",
                flake=".#elodin",
            ),
            nix_step(
                label=":python: lint",
                command="ruff format --check && ruff check",
                flake=".#elodin",
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
                flake=".#elodin",
            ),
            nix_step(
                label=":python: drone",
                command="python3 examples/drone/main.py bench --ticks 100",
                flake=".#elodin",
            ),
            nix_step(
                label=":python: rocket",
                command="python3 libs/nox-py/examples/rocket.py bench --ticks 100",
                flake=".#elodin",
            ),
            nix_step(
                label=":python: three-body",
                command="python3 libs/nox-py/examples/three-body.py bench --ticks 100",
                flake=".#elodin",
            ),
            nix_step(
                label=":python: cube-sat",
                command="python3 libs/nox-py/examples/cube-sat.py bench --ticks 10",
                flake=".#elodin",
            ),
        ],
    ),
    nix_step(label="alejandra", flake=".#elodin", command="alejandra -c ."),
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
                command=["cd images/aleph", "nix build --accept-flake-config .#toplevel"],
                agents={"queue": "nixos-arm"},
            ),
            step(
                label=":nix: sdimage",
                key="sdimage",
                command=[
                    "cd images/aleph",
                    "nix build --accept-flake-config .#sdimage",
                ],
                agents={"queue": "nixos-arm"},
            ),
            step(
                label=":nix: flash-uefi",
                key="flash-uefi",
                command=[
                    "cd images/aleph",
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
)
