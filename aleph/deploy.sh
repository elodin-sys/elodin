#! /usr/bin/env nix-shell
#! nix-shell -i bash -p gum nix-output-monitor
set -eu

# Default values
default_target="${USER}@fde1:2240:a1ef::1"
default_config="default"
aleph_builder=false

log_info() { gum log --level info "$*"; }
log_warn() { gum log --level warn "$*"; }

check_system_aarch64_builder() {
  ([ "$(uname -m)" = "aarch64" ] && [ "$(uname)" = "Linux" ]) ||
    ([ -f /etc/nix/machines ] && grep -q 'aarch64-linux' /etc/nix/machines)
}

show_usage() {
  echo "Usage: $0 [options] [user@host]"
  echo
  echo "Options:"
  echo "  -o SSHOPTS            Set NIX_SSHOPTS to the provided SSH options string"
  echo "  -c, --config CONFIG   Specify the NixOS configuration (default: $default_config)"
  echo "                        Valid values include: \"default\", \"base\", \"c-blinky\", \"sensor-fw\" "
  echo "  --aleph-builder       Force build on Aleph/Jetson (slow!) "
  echo "  --help                Show this help message"
  echo
  echo "Examples:"
  echo "  $0"
  echo "  $0 root@aleph-99a2.local"
  echo "  $0 -o \"-i $HOME/.ssh/key_file -o StrictHostKeyChecking=no\" root@aleph-99a2.local"
  exit "${1:-1}"
}

# Parse command line arguments
deploy_target="$default_target"
config="$default_config"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o)
      export NIX_SSHOPTS="$2"
      shift 2
      ;;
    -c|--config)
      config="$2"
      shift 2
      ;;
    --aleph-builder)
      aleph_builder=true
      shift
      ;;
    --help)
      show_usage 0
      ;;
    -*)
      log_warn "Unknown option: $1"
      show_usage 1
      ;;
    *)
      deploy_target="$1"
      break
      ;;
  esac
done

# Construct the target path with the selected configuration
target=".#nixosConfigurations.$config.config.system.build.toplevel"
ssh_store="ssh-ng://$deploy_target"

log_info "Using target: $deploy_target, configuration: $config"
if [ "$aleph_builder" = true ]; then
  build_cmd="nom build --accept-flake-config --eval-store auto --store $ssh_store $target --print-out-paths"
  log_info "Using Aleph as a remote builder"
  log_info "Running: $build_cmd"
  out_path=$(eval "$build_cmd")
else
  if ! check_system_aarch64_builder; then
    log_warn "No aarch64-linux builder found on this machine or in /etc/nix/machines"
    log_warn "Re-run with --aleph-builder to build on your Jetson (slow)."
    exit 1
  fi

  build_cmd="nom build --accept-flake-config $target --print-out-paths"
  log_info "Running: $build_cmd"
  out_path=$(eval "$build_cmd")
  copy_cmd="nix copy --no-check-sigs --to $ssh_store $out_path"
  log_info "Running: $copy_cmd"
  eval "$copy_cmd"
fi

log_info "Activating $out_path on $deploy_target"
remote_cmd="sudo nix-env -p /nix/var/nix/profiles/system --set ${out_path} && sudo ${out_path}/bin/switch-to-configuration switch;"
eval "ssh ${NIX_SSHOPTS:-} \"$deploy_target\" \"$remote_cmd\""
log_info "Deployment completed successfully"
