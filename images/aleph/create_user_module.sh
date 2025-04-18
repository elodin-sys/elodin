#!/usr/bin/env bash
set -eu

# Directory for user module
USER_MODULE_DIR="$HOME/.config/aleph/user-module"
USER_MODULE_FILE="$USER_MODULE_DIR/default.nix"

# Set up logging functions consistent with deploy.sh
log_info() { echo -e "\033[1;36m[INFO]\033[0m  $*"; }
log_warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
log_error() { echo -e "\033[1;31m[ERROR]\033[0m $*"; }

# Determine the preferred editor
get_editor() {
  if [ -n "${EDITOR:-}" ]; then
    echo "$EDITOR"
  elif [ -n "${VISUAL:-}" ]; then
    echo "$VISUAL"
  elif command -v nvim >/dev/null 2>&1; then
    echo "nvim"
  elif command -v vim >/dev/null 2>&1; then
    echo "vim"
  elif command -v nano >/dev/null 2>&1; then
    echo "nano"
  elif command -v vi >/dev/null 2>&1; then
    echo "vi"
  else
    echo "cat" # Fallback to just viewing the file
  fi
}

EDITOR_CMD=$(get_editor)

# Check if module already exists
if [ -f "$USER_MODULE_FILE" ]; then
  log_error "User module already exists at $USER_MODULE_FILE"
  log_info "To edit your existing module, run: $EDITOR_CMD $USER_MODULE_FILE"
  exit 1
fi

# Get user's SSH public key
SSH_KEY_FILE="$HOME/.ssh/id_ed25519.pub"
if [ -f "$SSH_KEY_FILE" ]; then
  SSH_KEY=$(cat "$SSH_KEY_FILE")
  log_info "Found SSH key at $SSH_KEY_FILE"
else
  log_warn "No SSH key found at $SSH_KEY_FILE"
  SSH_KEY_FILE="$HOME/.ssh/id_rsa.pub"
  if [ -f "$SSH_KEY_FILE" ]; then
    SSH_KEY=$(cat "$SSH_KEY_FILE")
    log_info "Found SSH key at $SSH_KEY_FILE"
  else
    log_error "No SSH key found. Please create one with: ssh-keygen -t ed25519"
    exit 1
  fi
fi

# Create directory
log_info "Creating user module directory at $USER_MODULE_DIR"
mkdir -p "$USER_MODULE_DIR"

# Create the module file
cat > "$USER_MODULE_FILE" << EOF
{ ... }: {
  users.users.${USER} = {
    isNormalUser = true;
    extraGroups = ["wheel" "video" "dialout"];
    openssh.authorizedKeys.keys = [
      "${SSH_KEY}"
    ];
  };

  # Add any other customizations below
}
EOF

log_info "Created user module at $USER_MODULE_FILE"
log_info "To deploy with this configuration: ./deploy.sh"
log_info "To edit this configuration: $EDITOR_CMD $USER_MODULE_FILE"

# Prompt user to edit if they want
read -p "Would you like to edit this file now? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  $EDITOR_CMD "$USER_MODULE_FILE"
fi

log_info "You can now run ./deploy.sh to deploy with your user configuration (it should automatically detect your user module)"

# Show how to use with override-input
log_info "If needed, you can manually specify it with: nix build --override-input user-module path:$USER_MODULE_DIR .#nixosConfigurations.default.config.system.build.toplevel"
