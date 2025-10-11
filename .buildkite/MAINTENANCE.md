# Buildkite Runner Maintenance

## Overview

This repository includes automated maintenance pipelines to clean up disk space on Buildkite runners when they run low on storage.

## How to Trigger Maintenance

To trigger a maintenance/cleanup job on the Buildkite runners, simply push a branch with one of these naming patterns:

- `maintenance/*` (e.g., `maintenance/cleanup`, `maintenance/gc-2024`)
- `ops/cleanup/*` (e.g., `ops/cleanup/urgent`)
- `ops/gc` (exact match)

### Example Commands

```bash
# Create and push a maintenance branch
git checkout -b maintenance/cleanup-$(date +%Y%m%d)
git commit --allow-empty -m "Trigger runner maintenance"
git push origin maintenance/cleanup-$(date +%Y%m%d)

# After the pipeline completes, you can delete the branch
git push origin --delete maintenance/cleanup-$(date +%Y%m%d)
```

### Quick One-Liner

```bash
# Trigger maintenance with a single command
git push origin HEAD:maintenance/gc-$(date +%s) && sleep 5 && git push origin --delete maintenance/gc-$(date +%s)
```

## What Gets Cleaned

The maintenance pipeline performs the following cleanup tasks:

1. **System Info** - Shows disk usage before cleanup
2. **Buildkite Builds** - Removes old build directories and target folders
3. **Nix Garbage Collection** - Runs aggressive Nix GC and store optimization
4. **Build Artifacts** - Cleans result symlinks and old build logs
5. **Docker** - Prunes Docker images, containers, and volumes (if Docker is installed)
6. **Package Caches** - Clears Cargo, pip, uv, and npm caches
7. **Final Report** - Shows disk usage after cleanup

## Expected Results

Typical cleanup can free:
- 10-50 GB from old Nix store entries
- 5-20 GB from old Buildkite builds and Rust target directories
- 1-5 GB from various caches

## Important Notes

- The maintenance pipeline will NOT run regular tests
- It only runs on branches matching the maintenance patterns
- The cleanup is aggressive but safe - it won't delete currently needed dependencies
- The Nix GC keeps the last 2 days of generations to avoid breaking active builds

## Monitoring

After triggering maintenance, monitor the Buildkite pipeline to see:
- How much space was freed
- Which cleanup steps were most effective
- Any errors that might need attention

## Regular Maintenance Schedule

Consider setting up a scheduled trigger (via cron or Buildkite scheduled builds) to run maintenance weekly:

```bash
# Example: Weekly cleanup every Sunday at 2 AM
# Add to Buildkite scheduled builds or external scheduler
0 2 * * 0 git push origin HEAD:maintenance/weekly-gc
```
