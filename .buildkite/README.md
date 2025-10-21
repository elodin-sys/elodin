# Buildkite CI Setup

This directory contains the configuration for Elodin's Buildkite CI infrastructure, designed for reproducible, ephemeral builds on AWS EC2. The CI system uses self-hosted agents with overlayroot (tmpfs root filesystem) to ensure a clean state on every reboot, preventing build artifacts and state from persisting between jobs. Agents automatically configure themselves based on instance architecture, joining either the `nixos-x86-aws` queue (for most CI tasks) or `nixos-arm-aws` queue (for ARM-specific builds like Aleph-OS). All build environments use Nix for reproducible dependencies, and the entire agent bootstrap process is fully automated via cloud-init—requiring only SSH deploy keys and a Buildkite agent token to be configured once in the user-data script.

## Quick Start

Here's the complete setup in order:

```bash
# 1. Generate SSH deploy key
ssh-keygen -t ed25519 -C "buildkite-ci@elodin.systems" -f ~/.ssh/buildkite_deploy_key

# 2. Base64 encode the private key (single line, no wrapping)
base64 -w 0 ~/.ssh/buildkite_deploy_key
# Copy the output and paste it into user-data at the SSH_KEY_BASE64 variable

# 3. Add your Buildkite agent token to user-data
# Get from: Buildkite Dashboard → Organization Settings → Agents → Reveal Agent Token
# Replace BUILDKITE_AGENT_TOKEN="bkt_your_actual_token_here" in user-data

# 4. Add public key to GitHub
cat ~/.ssh/buildkite_deploy_key.pub
# Copy output and add as Deploy Key in GitHub repo settings

# 5. Launch EC2 instances with the updated user-data script
```

**Note:** If you're on macOS, use `base64 -i ~/.ssh/buildkite_deploy_key` (no `-w` flag).

## Prerequisites

### 1. Generate and Encode SSH Deploy Key

Your Buildkite agents need access to your Git repository. Generate a deploy key and encode it:

```bash
# Generate a new SSH key (recommended: use a dedicated deploy key)
ssh-keygen -t ed25519 -C "buildkite-ci@elodin.systems" -f ~/.ssh/buildkite_deploy_key

# Base64 encode the PRIVATE key (no line wrapping)
# Linux:
base64 -w 0 ~/.ssh/buildkite_deploy_key

# macOS:
base64 -i ~/.ssh/buildkite_deploy_key

# Copy the entire base64 output and paste it into user-data at:
# SSH_KEY_BASE64="paste-here"
```

**Why base64?** The private key has multiple lines which causes YAML parsing issues in cloud-init. Base64 encoding makes it a single line.

### 2. Add Public Key to GitHub

```bash
# Display the PUBLIC key
cat ~/.ssh/buildkite_deploy_key.pub
# Output looks like: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAA...
```

Add to GitHub:
1. Go to your repository → Settings → Deploy keys
2. Click "Add deploy key"
3. Paste the public key content
4. Name it "Buildkite CI"
5. **Do NOT** check "Allow write access" (read-only is safer)

### 3. Add Buildkite Agent Token

Get your agent token from Buildkite:
- Go to: Buildkite Dashboard → Organization Settings → Agents → Reveal Agent Token
- Copy the token (starts with `bkt_`)
- In `user-data`, replace: `BUILDKITE_AGENT_TOKEN="bkt_your_actual_token_here"`

## How It Works

The user-data script uses a two-stage boot process:

### First Boot
1. Cloud-init installs packages
2. Writes configuration files (including systemd service)
3. Configures GRUB and initramfs for overlayroot
4. **Enables** `buildkite-bootstrap.service` (but doesn't run it yet)
5. **Reboots** to activate overlayroot

### Second Boot (Automatic)
1. System boots with overlayroot active (root filesystem is now tmpfs)
2. `buildkite-bootstrap.service` runs automatically
3. Installs Nix and Buildkite agent
4. Creates flag file `/var/lib/buildkite-bootstrap-complete` so it only runs once
5. Buildkite agent starts and joins the queue

**Why two boots?** Overlayroot needs to be active before installing Nix/Buildkite, otherwise those installations would be on the persistent disk instead of tmpfs.

## Architecture

The user-data script automatically detects the instance architecture using `uname -m` and configures the appropriate queue:

- **x86-64 agents**: Queue `nixos-x86-aws` (default for most steps)
  - Detected architectures: `x86_64`, `amd64`
- **ARM64 agents**: Queue `nixos-arm-aws` (for Aleph-OS builds: toplevel, sdimage)
  - Detected architectures: `aarch64`, `arm64`

This means you can use the same user-data script for both x86 and ARM instances!

## Troubleshooting

### Check bootstrap service status

```bash
# Check if the bootstrap service ran successfully
sudo systemctl status buildkite-bootstrap.service

# View bootstrap logs
sudo journalctl -u buildkite-bootstrap.service

# Check if bootstrap completed
ls -la /var/lib/buildkite-bootstrap-complete
# If this file exists, bootstrap ran successfully

# Check if overlayroot is active
mount | grep overlayroot
# Should show: overlayroot on / type overlay
```

### View agent logs

```bash
sudo journalctl -u buildkite-agent -f
```

### Manually trigger bootstrap (if needed)

```bash
# If bootstrap didn't run or you need to re-run it
sudo rm -f /var/lib/buildkite-bootstrap-complete
sudo systemctl start buildkite-bootstrap.service
```

### Verify architecture detection and queue assignment

```bash
# Check what architecture was detected
uname -m

# Check the configured agent tags
grep tags /etc/buildkite-agent/buildkite-agent.cfg
# Should show: tags="nix=true,os=nixos,arch=x86_64,queue=nixos-x86-aws"
#          or: tags="nix=true,os=nixos,arch=arm64,queue=nixos-arm-aws"
```

