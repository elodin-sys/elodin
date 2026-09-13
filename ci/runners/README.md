# Self-hosted GitHub Actions runners

Elodin release builds need more RAM than free GitHub-hosted runners provide. Heavy jobs now target machines you own. Tiny orchestration jobs (`plan`, `build-global-artifacts`, `host`, `announce`, `pypi-publish`, `release-dry-run`) stay on free `ubuntu-24.04`.

| Label | Machine | Workload |
|---|---|---|
| `ci-linux-x64` | x86_64 Linux box, Docker, Ubuntu 24.04 image | `x86_64-unknown-linux-{gnu,musl}` dist builds, `linux-wheel`, `cargo-deb` |
| `ci-macos-arm64` | Mac Studio, native runner | `aarch64-apple-darwin` dist build, `macos-wheel` |
| `ci-linux-arm64` | Mac Studio, Apple Container, same Ubuntu 24.04 image | `aarch64-unknown-linux-musl` (`elodin-db`) |
| `ci-windows-x64` | Windows 11 x64 box, native runner service | `x86_64-pc-windows-msvc` dist build + MSI |

Do not merge these workflow changes until the corresponding runner is online. A `push` to `main` will sit in queue forever if the label is missing. Emergency fallback: **Actions → Release → Run workflow → runners = `hosted-large`**.

PR builds still only run `dist plan` plus `Windows Check` on free hosted runners.

## Security (do this first)

The repository is public. A fork PR that edits `runs-on` can execute on these machines unless GitHub blocks it.

1. Repo **Settings → Actions → General → Fork pull request workflows from outside collaborators** → **Require approval for all outside collaborators**.
2. Confirm **Settings → Actions → Runners** offers **New self-hosted runner**. If the org blocks repo-level runners, register at org level in a group restricted to `elodin-sys/elodin` (`gh auth refresh -s admin:org`).
3. Never route `pull_request` jobs to these labels. Every job in `release.yml` that can land on a `ci-*` label is hard-gated with `github.event_name != 'pull_request'`; `plan` evaluates the PR's own `dist-workspace.toml`, so `pr_run_mode` alone is not a safe gate. Do not attach the runners to other public repos.

Registration tokens live one hour. Mint from any machine with `gh` logged in as a repo admin:

```bash
RUNNER_TOKEN=$(ci/runners/bin/mint-token.sh)
```

Do not commit tokens, paste them into issues, or put them in image build args.

## Phase 1 — x86_64 Linux box

Target: ≥ 8 cores, ≥ 32 GB RAM, ≥ 200 GB free. Any distro is fine; jobs run inside Ubuntu 24.04.

1. Inventory:

   ```bash
   cat /etc/os-release
   nproc
   free -g
   df -h /
   docker --version
   ```

2. Install Docker Engine, add your user to the `docker` group, enable `docker.service` at boot. Log out and back in so the group applies.

3. Clone this repo (or copy `ci/runners/linux/`) and build the image:

   ```bash
   ci/runners/linux/build-image.sh
   ci/runners/linux/create-runner.sh ci-linux-x64
   ```

   `create-runner.sh` makes volume `ci-linux-x64-work` and a container with `--restart unless-stopped`.

4. Mint a token (Studio or any admin laptop) and register:

   ```bash
   RUNNER_TOKEN=... ci/runners/linux/register.sh ci-linux-x64
   ```

5. Confirm **Settings → Actions → Runners** shows `ci-linux-x64` online.

6. From the Studio, dispatch **CI runner smoke tests** with `linux_x64`. Then dispatch **Release** (dry-run, `runners=self-hosted`) and compare the x86_64 artifacts, `--print=linkage` output, wheel manylinux tag, and `.deb` against [run 34528998758](https://github.com/elodin-sys/elodin/actions/runs/34528998758).

7. Reboot the box. The container should return on its own. Confirm the runner is online, then merge Phase 1.

## Phase 2 — Studio native macOS

1. As `stdio`, from the repo:

   ```bash
   ci/runners/mac/provision-host.sh
   sudo fdesetup add -usertoadd ci
   ```

   `provision-host.sh` creates the `ci` user, installs `git-lfs`, `protobuf`, `ffmpeg@8`, accepts the Xcode license, installs Apple Container 1.4.1, and runs `seed-python.sh`.

   `seed-python.sh` pre-populates `/Users/runner/hostedtoolcache` with Python `PYTHON_SERIES` (from `versions.env`). `actions/setup-python` hardcodes that path on macOS and its installer needs `sudo`; seeding once as an administrator means jobs on `ci` hit the cache and never install. On an already-provisioned host run it alone: `ci/runners/mac/seed-python.sh`. `pip install` in jobs falls back to the `ci` user site (`~/Library/Python/3.13`), which is expected.

2. Fast User Switch to `ci`. Keep that session logged in (screen lock is fine; logout is not). System Settings → Lock Screen: do not log out the `ci` session on idle.

3. As `ci`:

   ```bash
   RUNNER_TOKEN=$(/path/to/elodin/ci/runners/bin/mint-token.sh)
   /path/to/elodin/ci/runners/mac/provision-runner.sh
   ```

4. Dispatch smoke with `macos_arm64`, then a Release dry-run. Compare Mach-O architecture, `--print=linkage`, and the macOS wheel tag.

## Phase 3 — Studio Linux ARM64 (Apple Container)

As `ci`, after Phase 2:

1. First-time kernel install (interactive):

   ```bash
   container system start
   container run --rm --arch arm64 ubuntu:24.04 uname -m   # aarch64
   ```

2. Build and register, from a checkout the `ci` user can read:

   ```bash
   ci/runners/linux/build-image.sh
   ci/runners/linux/create-runner.sh ci-linux-arm64
   RUNNER_TOKEN=... ci/runners/linux/register.sh ci-linux-arm64
   ```

   The container gets 6 vCPUs and 16 GB.

3. Install the login helper (paths assume the `ci` home is `/Users/ci`):

   ```bash
   mkdir -p "$HOME/ci/runners/bin" "$HOME/ci/logs" "$HOME/Library/LaunchAgents"
   cp ci/runners/bin/start-linux-runner.sh "$HOME/ci/runners/bin/"
   chmod 0755 "$HOME/ci/runners/bin/start-linux-runner.sh"
   cp ci/runners/launchd/systems.elodin.ci.linux-start.plist "$HOME/Library/LaunchAgents/"
   plutil -lint "$HOME/Library/LaunchAgents/systems.elodin.ci.linux-start.plist"
   launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/systems.elodin.ci.linux-start.plist"
   ```

4. Dispatch smoke with `linux_arm64`, then a Release dry-run. Inspect the `elodin-db` ELF architecture.

Planned maintenance: `touch "$HOME/ci/MAINTENANCE"` stops the helper from restarting the Linux container. It does not stop a running job or the macOS runner.

## Phase 4 — Windows x64 box (native)

Target: Windows 11 x64, ≥ 8 cores, ≥ 32 GB RAM, ≥ 200 GB free, always on (plugged in, no sleep). Apply Windows Update first. Your normal account stays the interactive user; the runner runs as a hidden `ci-build` service account. You can stay logged in and work while jobs run.

1. Clone this repo (or copy `ci/runners/windows/`) onto the box. In PowerShell:

   ```powershell
   Set-ExecutionPolicy -Scope Process Bypass
   ```

2. In an elevated PowerShell on the Windows box, start with only the service-account password. Registration tokens last one hour; mint after the VS install so it is still valid:

   ```powershell
   $env:CI_BUILD_PASSWORD = '...'
   .\ci\runners\windows\provision.ps1
   ```

   Expect a reboot-free run of ~20–40 min (Visual Studio Build Tools dominates). At registration it prompts for `RUNNER_TOKEN`; mint that on the Studio (or any admin laptop) and paste it:

   ```bash
   RUNNER_TOKEN=$(ci/runners/bin/mint-token.sh)
   ```

   Then, in an elevated PowerShell on the Windows box:

   ```powershell
   $env:RUNNER_TOKEN = '...'
   $env:CI_BUILD_PASSWORD = '...'
   .\ci\runners\windows\provision.ps1
   ```

   Expect a reboot-free run of ~20–40 min (Visual Studio Build Tools dominates). The script is re-runnable. It creates `ci-build` if missing, hides that account from the sign-in screen, and lets `config.cmd --runasservice` grant **Log on as a service**. It ends by running `seed-python.ps1`, which pre-populates the runner tool cache (`C:\actions-runner\_work\_tool`) with Python `PYTHON_SERIES`. `actions/setup-python`'s Windows installer uses `InstallAllUsers=1` and needs admin, which `ci-build` does not have; the seeded cache means jobs skip the install. On an already-provisioned box run it alone from an elevated PowerShell: `.\ci\runners\windows\seed-python.ps1`.

3. Confirm the service is running and the runner is online:

   ```powershell
   Get-Service actions.runner.*
   ```

   Then check **Settings → Actions → Runners**.

4. Dispatch smoke with `windows_x64`. Then a Release dry-run. Install the produced MSI to confirm.

5. Builds use the box's cores (typically ~8). Expect noticeable load while you work. To cap parallelism, add `CARGO_BUILD_JOBS=N` to `C:\actions-runner\.env`.

## Final verification

1. Reboot the Studio, unlock FileVault as `ci`. Both Studio runners (`ci-macos-arm64`, `ci-linux-arm64`) should return online. Reboot the x86 box; `ci-linux-x64` should return. Reboot the Windows box; `ci-windows-x64` returns as a service without login.
2. Dispatch Release with `runners=hosted-large` once to prove the paid fallback still works.
3. Watch the next `push` to `main`. GitHub Actions minutes for this repo should then be $0.

## Resource budget

- x86 box: no container memory cap. `CARGO_BUILD_JOBS` is set to `nproc`.
- Studio (64 GB): Linux container 6 vCPU / 16 GB, macOS `CARGO_BUILD_JOBS=10`. A `main` push starts those lanes together with the Windows box.
- Windows box: no memory cap. `CARGO_BUILD_JOBS` defaults to all cores. Optionally set it in `C:\actions-runner\.env`.
- Persistent `_work/.../target` caches make warm builds much faster than hosted. Prune when free disk drops below ~100 GB.

## Day-to-day

- **Online check:** GitHub Settings → Actions → Runners.
- **Linux x64 restart:** `docker restart ci-linux-x64` (only when idle).
- **Linux ARM64 restart:** `container stop --time 60 ci-linux-arm64 && container start ci-linux-arm64`.
- **Replace a Linux image:** drain work, remove the runner in GitHub, `docker rm` / `container rm` the old container (keep or drop the work volume), rebuild, recreate, register with a new token. Do not copy `.runner` / `.credentials*` between containers.
- **Windows restart:** `Restart-Service actions.runner.*` when idle. Never clone or image the disk with a live `.runner` / `.credentials` identity.
- **Bumping `python-version` in `release.yml`:** update `PYTHON_SERIES` in `versions.env` and `$PythonSeries` in `seed-python.ps1`, then re-run `ci/runners/mac/seed-python.sh` (Studio, as administrator) and `seed-python.ps1` (Windows, elevated) before merging. The Linux images use setup-python's relocatable builds and need nothing.
- **Persistent-workspace rules for `release.yml`:** never enable `Swatinem/rust-cache` (or `setup-rust-toolchain`'s default `cache: true`) on `ci-*` lanes; its post step deletes `~/.cargo/bin` and prunes `target/`. After any `lfs: true` checkout on a self-hosted lane, run `git lfs pull`; `actions/checkout` only smudges files that changed since the previous checkout in that workspace.
- **Suspected compromise:** remove the runner in GitHub, revoke tokens, rebuild the environment, and treat artifacts from that host as untrusted. Deleting `_work` is not recovery.

`windows-check.yml`, `deploy-docs.yml`, and `flakehub-publish-tagged.yml` stay on free hosted runners.
