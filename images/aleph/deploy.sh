#!/usr/bin/env bash
set -eu

user="${USER}"
host="aleph.local"
target=".#nixosConfigurations.default.config.system.build.toplevel"

USER_MODULE_DIR="$HOME/.config/aleph/user-module"

log_info() { echo -e "\033[1;36m[INFO]\033[0m  $*" >&2; }
log_warn() { echo -e "\033[1;33m[WARN]\033[0m  $*" >&2; }
log_error() { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

_ssh_execute() {
  local ssh_user="$1"
  local ssh_host="$2"
  local ssh_command="$3"
  local ssh_key="${4:-}"
  local timeout="${5:-5}"
  local ssh_opts="-o BatchMode=yes -o ConnectTimeout=$timeout -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PreferredAuthentications=publickey -o PasswordAuthentication=no"
  if [ -n "$ssh_key" ]; then
    ssh_opts="$ssh_opts -i $ssh_key"
  fi
  ssh -q $ssh_opts "$ssh_user@$ssh_host" "$ssh_command" 2>/dev/null
  return $?
}

attempt_ssh() {
  local test_user="$1"
  local test_host="$2"
  log_info "Attempting to SSH to $test_host as $test_user (nopass)"
  local active_store_path
  if ! active_store_path=$(_ssh_execute "$test_user" "$test_host" "readlink -f /run/current-system"); then
    return 1
  fi
  log_info "SSH connectivity test to $test_host was successful"
  log_info "Active store path: $active_store_path"
}

find_working_ssh_key() {
  local test_user="$1"
  local test_host="$2"
  local key_found=""
  for key_file in ~/.ssh/id_ed25519 ~/.ssh/id_rsa; do
    if [ -f "$key_file" ]; then
      if _ssh_execute "$test_user" "$test_host" "echo success > /dev/null" "$key_file"; then
        key_found=$(realpath "$key_file")
        echo "$key_found"
        return 0
      fi
    fi
  done
  log_error "No working SSH key found"
  exit 1
}

# Check for user module and prepare override if needed
if [ -d "$USER_MODULE_DIR" ] && [ -f "$USER_MODULE_DIR/default.nix" ]; then
  log_info "Found user module at $USER_MODULE_DIR"
  override_arg="--override-input user-module path:$USER_MODULE_DIR"
else
  log_info "No user module found at $USER_MODULE_DIR"
  log_info "Using default configuration"
  log_info "Run ./create_user_module.sh to create your user module"
  override_arg=""
fi

if ! attempt_ssh $user $host; then
  log_warn "Failed to SSH to $host as $user"
  user="root"
  if ! attempt_ssh $user $host; then
    log_info "Attempting to add SSH key to $host's root authorized keys (NOTE: The default password is 'root')"
    ssh-copy-id $user@$host >/dev/null
    if ! attempt_ssh $user $host; then
      log_error "Failed to SSH to $host"
      exit 1
    fi
  fi
fi

remote_arg=""
if ! ( ([ "$(uname -m)" = "aarch64" ] && [ "$(uname)" = "Linux" ]) ||
  ([ -f /etc/nix/machines ] && grep -q 'aarch64-linux' /etc/nix/machines)); then
  log_warn "No aarch64-linux builder found, falling back to building on Aleph (slow)"
  ssh_key=$(find_working_ssh_key $user $host)
  remote_arg="--builders 'ssh://$user@$host aarch64-linux $ssh_key' --option builders-use-substitutes false --max-jobs 0"
fi

build_cmd="nix build --accept-flake-config $override_arg $remote_arg $target -v --print-out-paths"
log_info "Running: $build_cmd"
log_file=$(mktemp)
log_info "Streaming build logs to: $log_file"
set +e
out_path=$(eval "$build_cmd" 2>"$log_file")
build_status=$?
set -e

if [ $build_status -ne 0 ]; then
  if grep -q "Host key verification failed" "$log_file"; then
    log_error "Build failed due to host key verification issues."
    log_info "To resolve this, please run SSH into Aleph as root and accept the host key:"
    log_info "    sudo ssh -o StrictHostKeyChecking=ask $user@$host exit"
    log_info "After that, run this script again."
  else
    log_error "Build failed"
  fi
  exit 1
fi

copy_cmd="nix copy --no-check-sigs --to ssh-ng://$user@$host $out_path"
log_info "Running: $copy_cmd"
eval "$copy_cmd"

log_info "Activating $out_path on $user@$host"
ssh "$user@$host" "sudo nix-env -p /nix/var/nix/profiles/system --set ${out_path} \
  && sudo ${out_path}/bin/switch-to-configuration switch;"
log_info "Deployment completed successfully"
