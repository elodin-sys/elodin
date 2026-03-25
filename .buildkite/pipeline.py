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
                command="cd libs/db; clang examples/client.c -lm",
            ),
            nix_step(
                emoji=":c:",
                label="db-cpp-example",
                command="cd libs/db; clang++ -std=c++23 examples/client.cpp",
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
                command="cargo test --release --workspace --exclude elodin-db --exclude elodin-db-tests --exclude muxide -- -Z unstable-options --format json --report-time | buildkite-test-collector && cargo test --release -p elodin-db --lib -- --test-threads=1 -Z unstable-options --format json --report-time | buildkite-test-collector && cargo test --release -p elodin-db-tests -- --test-threads=1 -Z unstable-options --format json --report-time | buildkite-test-collector",
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
        flake=".#run",
        # this step is just to verify that the package can be imported
        command="python -c 'import elodin; print(elodin.__version__)'",
    ),
    group(
        name=":python: python",
        depends_on=["nox-py"],
        steps=[
            nix_step(
                label=":python: pytest",
                flake=".#run",
                command="pytest libs/nox-py -o 'pythonpath='",
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
                flake=".#run",
                command="bash ./scripts/ci/regress.sh ball examples/ball/main.py",
            ),
            nix_step(
                label=":python: drone",
                flake=".#run",
                command="bash ./scripts/ci/regress.sh drone examples/drone/main.py",
            ),
            nix_step(
                label=":python: rocket",
                flake=".#run",
                command="bash ./scripts/ci/regress.sh rocket examples/rocket/main.py",
            ),
            nix_step(
                label=":python: three-body",
                flake=".#run",
                command="bash ./scripts/ci/regress.sh three-body examples/three-body/main.py",
            ),
            nix_step(
                label=":python: cube-sat",
                flake=".#run",
                command="bash ./scripts/ci/regress.sh cube-sat examples/cube-sat/main.py",
            ),
            nix_step(
                label=":python: sensor-camera",
                flake=".#tracy",
                command="./scripts/ci/sensor_camera_perf.sh",
                env={"ELODIN_SENSOR_CAMERA_CAPTURE_TRACY": "1"},
            ),
        ],
    ),
    group(
        name=":racehorse: performance",
        steps=[
            nix_step(
                emoji=":racehorse:",
                label="perf-elodin-db",
                pre_command="nix develop --command bash -c 'cargo build --release -p elodin-db --bin elodin-db-bench --features tracy'",
                flake=".#tracy",
                command="bash ./scripts/ci/db_perf.sh",
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
