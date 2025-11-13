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

# 3. Generate Nix cache signing keys
nix-store --generate-binary-cache-key elodin-cache-1 cache-priv-key.pem cache-pub-key.pem
base64 -w 0 cache-priv-key.pem
# Copy the output and paste it into user-data at the CACHE_SIGNING_KEY_BASE64 variable

# 4. Add your Buildkite agent token to user-data
# Get from: Buildkite Dashboard → Organization Settings → Agents → Reveal Agent Token
# Replace BUILDKITE_AGENT_TOKEN="bkt_your_actual_token_here" in user-data

# 5. Add public key to GitHub
cat ~/.ssh/buildkite_deploy_key.pub
# Copy output and add as Deploy Key in GitHub repo settings

# 6. Launch EC2 instances with the updated user-data script
```

**Note:** If you're on macOS, use `base64 -i` instead of `base64 -w 0` for base64 encoding.

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

### 3. Configure S3 Binary Cache (Already Set Up)

The CI uses an S3 bucket (`elodin-nix-cache`) for caching Nix build artifacts:

**S3 Bucket Setup** (one-time, already completed):
```bash
# Create the S3 bucket
aws s3 mb s3://elodin-nix-cache --region us-west-2

# Set bucket policy for public read access (allowing developers and CI to download)
# See user-data script for the full configuration
```

**Generate Signing Keys** (needed for each agent setup):
```bash
# Generate new signing keys
nix-store --generate-binary-cache-key elodin-cache-1 cache-priv-key.pem cache-pub-key.pem

# Base64 encode the PRIVATE key (no line wrapping)
# Linux:
base64 -w 0 cache-priv-key.pem

# macOS:
base64 -i cache-priv-key.pem

# Copy the output and paste it into user-data at:
# CACHE_SIGNING_KEY_BASE64="paste-here"
```

**Important:** Keep `cache-priv-key.pem` secure and do NOT commit it to the repository. Only the public key should be shared (already in `aleph/flake.nix`).

### 4. Add Buildkite Agent Token

Get your agent token from Buildkite:
- Go to: Buildkite Dashboard → Organization Settings → Agents → Reveal Agent Token
- Copy the token (starts with `bkt_`)
- In `user-data`, replace: `BUILDKITE_AGENT_TOKEN="bkt_your_actual_token_here"`

### 5. IAM Role for EC2 Instances

Ensure your EC2 instances have an IAM role with permissions to read/write from the S3 cache:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::elodin-nix-cache",
        "arn:aws:s3:::elodin-nix-cache/*"
      ]
    }
  ]
}
```

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
3. Installs standard Nix using Determinate Systems installer (without auth features)
4. Configures S3 binary cache for downloads and uploads
5. Sets up post-build-hook to automatically upload builds to S3
6. Installs and configures Buildkite agent
7. Creates flag file `/var/lib/buildkite-bootstrap-complete` so it only runs once
8. Buildkite agent starts and joins the queue

**Why two boots?** Overlayroot needs to be active before installing Nix/Buildkite, otherwise those installations would be on the persistent disk instead of tmpfs.

**Note on Nix installation:** We use the Determinate Systems installer but without the `--determinate` flag. This gives us standard Nix without authentication requirements, which is ideal for ephemeral CI environments.

### Cache Upload Process

After each successful build:
1. Nix invokes the post-build-hook (`/etc/nix/upload-to-cache.sh`)
2. The script signs the build outputs with the private signing key
3. Build artifacts are uploaded to `s3://elodin-nix-cache`
4. Other agents and developers can now fetch these cached builds

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

### Verify S3 cache configuration

```bash
# Check if the upload script exists and is executable
ls -la /etc/nix/upload-to-cache.sh

# Check if the signing key is in place
ls -la /etc/nix/cache-priv-key.pem

# Verify Nix configuration includes S3 cache
cat /etc/nix/nix.conf | grep -A 3 "S3 Binary Cache"

# Test S3 access (requires AWS credentials via IAM role)
aws s3 ls s3://elodin-nix-cache/

# Manually test uploading to cache
nix copy --to 's3://elodin-nix-cache?region=us-west-2&secret-key=/etc/nix/cache-priv-key.pem' /nix/store/some-path
```

### Check cache upload logs

```bash
# Watch Nix daemon logs to see cache uploads
sudo journalctl -u nix-daemon.service -f

# Look for lines like:
# "Uploading paths to S3 cache: /nix/store/..."
```
