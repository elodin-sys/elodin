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
                label="db-cpp-batched",
                command="cd libs/db; clang++ -std=c++23 examples/client-batched.cpp",
            ),
            nix_step(
                emoji=":c:",
                label="db-cpp-grpc-batched",
                command="scripts/ci/db_grpc_cpp_smoke.sh",
            ),
            nix_step(
                emoji=":c:",
                label="db-cpp-per-component",
                command="cd libs/db; clang++ -std=c++23 examples/client-per-component.cpp",
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
                command="cargo test --release --workspace --exclude elodin-db --exclude elodin-db-tests -- -Z unstable-options --format json --report-time | buildkite-test-collector && cargo test --release -p elodin-db --lib -- --test-threads=1 -Z unstable-options --format json --report-time | buildkite-test-collector && cargo test --release -p elodin-db-tests -- --test-threads=1 -Z unstable-options --format json --report-time | buildkite-test-collector",
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
            nix_step(
                emoji=":crab:",
                label="elodin-db grpc",
                command="export CARGO_BUILD_JOBS=1 RUST_TEST_THREADS=1 RAYON_NUM_THREADS=1; cargo check -p elodin-db && cargo test -p elodin-db --features grpc --lib grpc:: -- --test-threads=1 && cargo test -p elodin-db --features grpc --bin elodin-db grpc_ -- --test-threads=1 && cargo test -p elodin-db --features grpc grpc_address_follows_main_port -- --test-threads=1",
            ),
        ],
    ),
    group(
        name=":black_nib: writing",
        steps=[
            nix_step(
                label="windows-safe paths",
                command="python3 scripts/ci/test_windows_paths.py",
            ),
            nix_step(
                label="typos",
                command="typos -c typos.toml",
            ),
            nix_step(
                label="buf lint",
                command="buf lint",
            ),
            nix_step(
                label="buf breaking",
                # Scoped to the db module; skips until the baseline branch has it.
                command="git fetch --force origin main:buf-breaking-baseline && if git cat-file -e buf-breaking-baseline:libs/db/proto 2>/dev/null; then buf breaking --against '.git#branch=buf-breaking-baseline,subdir=libs/db/proto'; else echo 'no protos in baseline; skipping'; fi",
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
                label=":python: regress.sh examples",
                flake=".#run",
                command="bash ./scripts/ci/regress.sh --all",
            ),
            nix_step(
                label=":python: frames",
                flake=".#run",
                command="python3 examples/frames/main.py",
            ),
            nix_step(
                label=":python: elodin-db gRPC full API",
                flake=".#run",
                pre_command="nix develop --command bash -c 'cargo build --release -p rc-jet-controller'",
                command="scripts/ci/db_grpc_full_api_demo.sh",
                env={"ELODIN_RC_JET_CONTROLLER_BIN": "target/release/rc-jet-controller"},
            ),
            nix_step(
                label=":python: sensor-camera",
                flake=".#tracy",
                command="./scripts/ci/sensor_camera_perf.sh",
                env={"ELODIN_SENSOR_CAMERA_CAPTURE_TRACY": "1"},
            ),
            # Monte Carlo steps must not run concurrently: each `monte-carlo run`
            # reaps stray elodin/elodin-db processes at startup, so two campaigns
            # sharing an agent would reap each other's orchestrator (SIGTERM /
            # exit 143). Serialize them with depends_on.
            nix_step(
                emoji=":rocket:",
                label="monte-carlo apollo",
                key="monte-carlo-apollo",
                command="just install && ./scripts/test-apollo-monte-carlo.sh",
            ),
            nix_step(
                emoji=":sparkles:",
                label="monte-carlo quickstart",
                depends_on=["monte-carlo-apollo"],
                command="just install && ./scripts/test-quickstart.sh",
            ),
        ],
    ),
    group(
        name=":racehorse: performance",
        steps=[
            nix_step(
                emoji=":racehorse:",
                label="perf-elodin-db",
                pre_command="nix develop --command bash -c 'cargo build --release -p elodin-db --bin elodin-db-bench --features grpc,tracy'",
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
    step(
        label=":nix: elodin-db-protos",
        command="nix build .#elodin-db-protos",
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
