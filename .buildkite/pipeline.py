import os

from buildkite import group, pipeline, step
from steps import c_step, nix_step, rust_step

AZ_CONFIG = {"cluster_name": "dev", "rg_name": "dev"}

GITHUB_ACTION_TRIGGER = os.getenv("TRIGGERED_FROM_GHA", "") == "1"
BRANCH_NAME = (
    os.environ["PR_CLOSED_BRANCH"] if GITHUB_ACTION_TRIGGER else os.environ["BUILDKITE_BRANCH"]
)

# Check if this is a maintenance/cleanup branch
IS_MAINTENANCE_BRANCH = (
    BRANCH_NAME.startswith("maintenance/") or 
    BRANCH_NAME.startswith("ops/cleanup") or
    BRANCH_NAME.startswith("ops/gc")
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
                command="python3 examples/ball/main.py bench --ticks 100",
            ),
            nix_step(
                label=":python: drone",
                command="python3 examples/drone/main.py bench --ticks 100",
            ),
            nix_step(
                label=":python: rocket",
                command="python3 libs/nox-py/examples/rocket.py bench --ticks 100",
            ),
            nix_step(
                label=":python: three-body",
                command="python3 libs/nox-py/examples/three-body.py bench --ticks 100",
            ),
            nix_step(
                label=":python: cube-sat",
                command="python3 libs/nox-py/examples/cube-sat.py bench --ticks 10",
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

# Maintenance steps for cleanup branches
maintenance_steps = [
    step(
        label=":information_source: System Info",
        command=[
            "echo '=== Disk Usage Before Cleanup ==='",
            "df -h",
            "echo ''",
            "echo '=== Nix Store Size ==='",
            "du -sh /nix/store 2>/dev/null || echo 'Cannot calculate nix store size'",
            "echo ''",
            "echo '=== Build Directory Size ==='", 
            "du -sh /var/lib/buildkite-agent-azure-* 2>/dev/null || echo 'Cannot find builds directory'",
        ]
    ),
    
    step(
        label=":broom: Clean Buildkite Builds",
        command=[
            "echo 'Cleaning old Buildkite builds...'",
            "# Find and clean builds older than 1 day",
            "find /var/lib/buildkite-agent-azure-*/builds -maxdepth 3 -type d -name 'elodin' -mtime +1 2>/dev/null | while read -r dir; do echo \"Removing: $dir\"; rm -rf \"$dir\"; done || true",
            "# Clean any leftover target directories",
            "find /var/lib/buildkite-agent-azure-*/builds -type d -name 'target' -mtime +0 2>/dev/null | while read -r dir; do echo \"Removing target: $dir\"; rm -rf \"$dir\"; done || true",
            "echo 'Buildkite cleanup complete'",
        ]
    ),
    
    step(
        label=":recycle: Nix Garbage Collection",
        command=[
            "echo 'Running Nix garbage collection...'",
            "# Show current Nix profiles",
            "nix-env --list-generations || true",
            "# Delete old generations (keep last 2)",
            "nix-collect-garbage --delete-older-than 2d || true",
            "# Run aggressive garbage collection",
            "nix-collect-garbage -d || true",
            "# Optimize the Nix store",
            "nix-store --optimise || true",
            "echo 'Nix GC complete'",
        ]
    ),
    
    step(
        label=":wastebasket: Clean Build Artifacts",
        command=[
            "echo 'Cleaning build artifacts...'",
            "# Remove result symlinks",
            "find /var/lib/buildkite-agent-azure-* -type l -name 'result*' 2>/dev/null | while read -r link; do echo \"Removing: $link\"; rm \"$link\"; done || true",
            "# Clean Nix build logs older than 1 day",
            "find /nix/var/log/nix/drvs -type f -mtime +1 -delete 2>/dev/null || true",
            "# Clean tmp",
            "rm -rf /tmp/nix-build-* 2>/dev/null || true",
            "rm -rf /tmp/tmp.* 2>/dev/null || true",
            "echo 'Build artifacts cleanup complete'",
        ]
    ),
    
    step(
        label=":whale: Clean Docker",
        command=[
            "echo 'Cleaning Docker resources...'",
            "# Only run if docker is available",
            "if command -v docker &> /dev/null; then",
            "  docker system prune -af --volumes 2>/dev/null || echo 'Docker prune failed'",
            "  echo 'Docker cleanup complete'",
            "else",
            "  echo 'Docker not found, skipping'",
            "fi",
        ]
    ),
    
    step(
        label=":package: Clean Package Caches",
        command=[
            "echo 'Cleaning package caches...'",
            "# Clean cargo cache if it exists",
            "if [ -d ~/.cargo ]; then",
            "  rm -rf ~/.cargo/registry/cache ~/.cargo/registry/index 2>/dev/null || true",
            "  echo 'Cargo cache cleaned'",
            "fi",
            "# Clean Python caches",
            "rm -rf ~/.cache/pip ~/.cache/uv 2>/dev/null || true",
            "# Clean npm cache if exists",
            "npm cache clean --force 2>/dev/null || true",
            "echo 'Package cache cleanup complete'",
        ]
    ),
    
    step(
        label=":chart_with_upwards_trend: Final Disk Usage",
        command=[
            "echo '=== Disk Usage After Cleanup ==='",
            "df -h",
            "echo ''",
            "echo '=== Nix Store Size After ==='",
            "du -sh /nix/store 2>/dev/null || echo 'Cannot calculate nix store size'",
            "echo ''",
            "echo '✅ Maintenance complete! Check df -h output above for freed space'",
        ]
    ),
]

# Choose which steps to run based on branch
if IS_MAINTENANCE_BRANCH:
    pipeline_steps = maintenance_steps
    env_vars = {
        "BUILDKITE_CLEAN_CHECKOUT": "true",  # Don't keep the repo around
    }
else:
    pipeline_steps = test_steps
    env_vars = {
        "BUILDKITE_PLUGIN_FS_CACHE_FOLDER": "/run/buildkite/cache",
    }

pipeline(
    steps=pipeline_steps,
    env=env_vars,
)
