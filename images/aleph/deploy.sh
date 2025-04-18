#!/usr/bin/env bash
set -eu

# Default values
default_user="${USER}"
default_host="fde1:2240:a1ef::1"
target=".#nixosConfigurations.default.config.system.build.toplevel"
no_aleph_builder=false

log_info() { echo -e "\033[1;36m[INFO]\033[0m  $*" >&2; }
log_warn() { echo -e "\033[1;33m[WARN]\033[0m  $*" >&2; }
log_error() { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

show_usage() {
  echo "Usage: $0 [options]"
  echo
  echo "Options:"
  echo "  -h, --host HOST       Specify the hostname or IP address (default: $default_host)"
  echo "  -u, --user USER       Specify the SSH username (default: $default_user)"
  echo "  --no-aleph-builder    Don't use Aleph as a remote builder (use local machine or"
  echo "                         configured remote builders instead)"
  echo "  --help                Show this help message"
  echo
  echo "Example:"
  echo "  $0 -h fde1:2240:a1ef::1 -u myuser"
  exit 1
}

# Parse command line arguments
user="$default_user"
host="$default_host"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--host)
      host="$2"
      shift 2
      ;;
    -u|--user)
      user="$2"
      shift 2
      ;;
    --no-aleph-builder)
      no_aleph_builder=true
      shift
      ;;
    --help)
      show_usage
      ;;
    *)
      log_error "Unknown option: $1"
      show_usage
      ;;
  esac
done

log_info "Using host: $host, user: $user"
if [ "$no_aleph_builder" = true ]; then
  log_info "Not using Aleph as a remote builder"
fi

ssh_execute() {
  local ssh_user="$1"
  local ssh_host="$2"
  local ssh_command="$3"
  local silent="${4:-false}"
  local timeout="${5:-5}"
  local output

  local ssh_opts="-o BatchMode=yes -o ConnectTimeout=$timeout -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PreferredAuthentications=publickey -o PasswordAuthentication=no"

  if [ "$silent" = "true" ]; then
    output=$(ssh -q $ssh_opts "$ssh_user@$ssh_host" "$ssh_command" 2>/dev/null) || return $?
  else
    output=$(ssh $ssh_opts "$ssh_user@$ssh_host" "$ssh_command") || return $?
  fi

  echo "$output"
  return 0
}

get_user_pubkey() {
  for key_file in ~/.ssh/id_ed25519.pub ~/.ssh/id_rsa.pub; do
    if [ -f "$key_file" ]; then
      cat "$key_file"
      return 0
    fi
  done

  log_error "No SSH public key found. Please create one with: ssh-keygen -t ed25519"
  exit 1
}

create_remote_user() {
  local new_user="$1"
  local ssh_host="$2"

  # Get the public key
  local pub_key=$(get_user_pubkey)

  log_info "Creating user $new_user on $ssh_host..."

  # Create user and add to appropriate groups
  ssh "root@$ssh_host" "
      if ! id $new_user &>/dev/null; then
        useradd -m -G wheel,video,dialout $new_user
        mkdir -p /home/$new_user/.ssh
        echo '$pub_key' > /home/$new_user/.ssh/authorized_keys
        chmod 700 /home/$new_user/.ssh
        chmod 600 /home/$new_user/.ssh/authorized_keys
        user_group=\$(id -gn $new_user)
        chown -R $new_user:\$user_group /home/$new_user/.ssh
        echo 'User $new_user created successfully'
      else
        echo 'User $new_user already exists, adding SSH key'
        mkdir -p /home/$new_user/.ssh
        echo '$pub_key' >> /home/$new_user/.ssh/authorized_keys
      fi
    "

  log_info "User $new_user set up successfully on $ssh_host"

  # Test connection with the new user
  if ! ssh_execute "$new_user" "$ssh_host" "echo SSH connection successful" true; then
    log_error "Failed to connect as user $new_user"
    exit 1
  fi
}

# Try connecting as the specified user
if ! ssh_execute "$user" "$host" "readlink -f /run/current-system" true; then
  log_warn "Failed to SSH to $host as $user"

  # Try root instead
  if ssh_execute "root" "$host" "readlink -f /run/current-system" true; then
    log_info "Connected as root. Checking if user $user exists..."

    # Check if the user exists on the remote system
    if ! ssh "root@$host" "id $user &>/dev/null"; then
      log_info "User $user does not exist on $host"
      read -p "Would you like to create the user $user on $host? [Y/n] " create_user
      create_user=${create_user:-Y}

      if [[ $create_user =~ ^[Yy]$ ]]; then
        create_remote_user "$user" "$host"
      else
        log_error "Cannot proceed without creating user $user"
        exit 1
      fi
    else
      log_info "User $user exists but SSH authentication failed"
      log_info "Setting up SSH key for $user"
      create_remote_user "$user" "$host"
    fi
  else
    log_info "Attempting to add SSH key to $host's root account (NOTE: The default password is 'root')"
    ssh-copy-id "root@$host" >/dev/null || true

    if ssh_execute "root" "$host" "echo SSH connection successful" true; then
      log_info "Connected as root. Checking if user $user exists..."

      # Now check and create user if needed
      if ! ssh "root@$host" "id $user &>/dev/null"; then
        log_info "User $user does not exist on $host"
        read -p "Would you like to create the user $user on $host? [Y/n] " create_user
        create_user=${create_user:-Y}

        if [[ $create_user =~ ^[Yy]$ ]]; then
          create_remote_user "$user" "$host"
        else
          log_error "Cannot proceed without creating user $user"
          exit 1
        fi
      else
        log_info "User $user exists but SSH authentication failed"
        log_info "Setting up SSH key for $user"
        create_remote_user "$user" "$host"
      fi
    else
      log_error "Failed to SSH to $host as root or $user"
      exit 1
    fi
  fi
else
  log_info "$user@$host SSH connectivity verified"
fi

if [ "$no_aleph_builder" = false ] && ! ( ([ "$(uname -m)" = "aarch64" ] && [ "$(uname)" = "Linux" ]) ||
  ([ -f /etc/nix/machines ] && grep -q 'aarch64-linux' /etc/nix/machines)); then
  log_warn "No aarch64-linux builder found, falling back to building on Aleph (slow)"
  build_cmd="nix build --accept-flake-config --eval-store auto --store ssh-ng://$user@$host $target --print-out-paths"
  log_info "Running: $build_cmd"
  out_path=$(eval "$build_cmd")
else
  build_cmd="nix build --accept-flake-config $target --print-out-paths"
  log_info "Running: $build_cmd"
  out_path=$(eval "$build_cmd")
  copy_cmd="nix copy --no-check-sigs --to ssh-ng://$user@$host $out_path"
  log_info "Running: $copy_cmd"
  eval "$copy_cmd"
fi

log_info "Activating $out_path on $user@$host"
ssh "$user@$host" "sudo nix-env -p /nix/var/nix/profiles/system --set ${out_path} \
  && sudo ${out_path}/bin/switch-to-configuration switch;"
log_info "Deployment completed successfully"
