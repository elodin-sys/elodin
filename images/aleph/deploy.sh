#!/usr/bin/env bash
set -eu

user="${USER}"
host="aleph.local"
target=".#nixosConfigurations.default.config.system.build.toplevel"

log_info() { echo -e "\033[1;36m[INFO]\033[0m  $*"; }
log_warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
log_error() { echo -e "\033[1;31m[ERROR]\033[0m $*"; }

attempt_ssh() {
  log_info "Attempting to SSH to $2 as $1 (nopass)"
  active_store_path=$(ssh -q \
    -o BatchMode=yes \
    -o ConnectTimeout=5 \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o PreferredAuthentications=publickey \
    -o PasswordAuthentication=no \
    "$1@$2" "readlink -f /run/current-system")
  if [ $? -ne 0 ]; then
    return 1
  fi
  log_info "SSH connectivity test to $2 was successful"
  log_info "Active store path: $active_store_path"
}

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

if ! ( ([ "$(uname -m)" = "aarch64" ] && [ "$(uname)" = "Linux" ]) ||
  ([ -f /etc/nix/machines ] && grep -q 'aarch64-linux' /etc/nix/machines)); then
  log_warn "No aarch64-linux builder found, falling back to building on Aleph (slow)"
  build_cmd="nix build --accept-flake-config --eval-store auto --store ssh-ng://$user@$host $target --print-out-paths"
  log_info "Running: $build_cmd"
  out_path=$(${build_cmd})
else
  build_cmd="nix build --accept-flake-config $target --print-out-paths"
  log_info "Running: $build_cmd"
  out_path=$(${build_cmd})
  copy_cmd="nix copy --to ssh-ng://$user@$host $out_path"
  log_info "Running: $copy_cmd"
  $(${copy_cmd})
fi
log_info "Activating $out_path on $user@$host"
ssh "$user@$host" "sudo nix-env -p /nix/var/nix/profiles/system --set ${out_path} \
  && sudo ${out_path}/bin/switch-to-configuration switch;"
log_info "Deployment completed successfully"
