#!/usr/bin/env python3
"""
Maintenance pipeline for Buildkite runners
Trigger by pushing to branches matching: maintenance/* or ops/cleanup/*
"""

import os
from buildkite import pipeline, step

def maintenance_step(label, command, emoji=":wrench:"):
    """Create a maintenance step that runs with elevated permissions"""
    return step(
        label=f"{emoji} {label}",
        command=command,
        agents={},  # Runs on default queue
    )

# Define maintenance steps
maintenance_steps = [
    maintenance_step(
        label="System Info",
        emoji=":information_source:",
        command=[
            "echo '=== Disk Usage Before Cleanup ==='",
            "df -h",
            "echo ''",
            "echo '=== Nix Store Size ==='",
            "du -sh /nix/store 2>/dev/null || echo 'Cannot calculate nix store size'",
            "echo ''",
            "echo '=== Build Directory Size ==='", 
            "du -sh /var/lib/buildkite-agent-azure-e/builds/ 2>/dev/null || echo 'Cannot calculate builds size'",
        ]
    ),
    
    maintenance_step(
        label="Clean Buildkite Builds",
        emoji=":broom:",
        command=[
            "echo 'Cleaning old Buildkite builds...'",
            "# Clean builds older than 1 day",
            "find /var/lib/buildkite-agent-azure-e/builds/ -maxdepth 3 -type d -name 'elodin' -mtime +1 -exec rm -rf {} + 2>/dev/null || true",
            "# Clean any leftover target directories",
            "find /var/lib/buildkite-agent-azure-e/builds/ -type d -name 'target' -mtime +0 -exec rm -rf {} + 2>/dev/null || true",
            "echo 'Buildkite cleanup complete'",
        ]
    ),
    
    maintenance_step(
        label="Nix Garbage Collection",
        emoji=":recycle:",
        command=[
            "echo 'Running Nix garbage collection...'",
            "# Delete old generations",
            "nix-collect-garbage --delete-old || true",
            "# Run aggressive garbage collection",
            "nix-collect-garbage -d || true",
            "# Optimize the Nix store",
            "nix-store --optimise || true",
            "echo 'Nix GC complete'",
        ]
    ),
    
    maintenance_step(
        label="Clean Nix Build Artifacts",
        emoji=":wastebasket:",
        command=[
            "echo 'Cleaning Nix build artifacts...'",
            "# Remove result symlinks",
            "find /var/lib/buildkite-agent-azure-e -type l -name 'result*' -exec rm {} + 2>/dev/null || true",
            "# Clean Nix build logs",
            "rm -rf /nix/var/log/nix/drvs/* 2>/dev/null || true",
            "# Clean tmp if it exists",
            "rm -rf /tmp/nix-build-* 2>/dev/null || true",
            "echo 'Build artifacts cleanup complete'",
        ]
    ),
    
    maintenance_step(
        label="Clean Docker (if applicable)",
        emoji=":whale:",
        command=[
            "echo 'Cleaning Docker resources...'",
            "# Only run if docker is available",
            "if command -v docker &> /dev/null; then",
            "  docker system prune -af --volumes 2>/dev/null || true",
            "  echo 'Docker cleanup complete'",
            "else",
            "  echo 'Docker not found, skipping'",
            "fi",
        ]
    ),
    
    maintenance_step(
        label="Clean Package Caches",
        emoji=":package:",
        command=[
            "echo 'Cleaning package caches...'",
            "# Clean cargo cache if it exists",
            "rm -rf ~/.cargo/registry/cache 2>/dev/null || true",
            "rm -rf ~/.cargo/registry/index 2>/dev/null || true",
            "# Clean Python caches",
            "rm -rf ~/.cache/pip 2>/dev/null || true",
            "rm -rf ~/.cache/uv 2>/dev/null || true",
            "echo 'Package cache cleanup complete'",
        ]
    ),
    
    maintenance_step(
        label="Final Disk Usage",
        emoji=":chart_with_upwards_trend:",
        command=[
            "echo '=== Disk Usage After Cleanup ==='",
            "df -h",
            "echo ''",
            "echo '=== Nix Store Size After ==='",
            "du -sh /nix/store 2>/dev/null || echo 'Cannot calculate nix store size'",
            "echo ''",
            "echo '=== Space Freed ==='",
            "echo 'Check the before/after df -h output above to see space freed'",
        ]
    ),
]

# Only run on maintenance branches
branch_name = os.environ.get("BUILDKITE_BRANCH", "")
is_maintenance_branch = (
    branch_name.startswith("maintenance/") or 
    branch_name.startswith("ops/cleanup") or
    branch_name.startswith("ops/gc")
)

if is_maintenance_branch:
    pipeline(
        steps=maintenance_steps,
        env={
            "BUILDKITE_CLEAN_CHECKOUT": "true",  # Don't keep the repo around
        }
    )
else:
    # Empty pipeline for non-maintenance branches
    # This script can be safely called but won't do anything
    pipeline(steps=[], env={})
