#! /usr/bin/env nix-shell
#! nix-shell -i bash -p gum nix-output-monitor
set -eu

# Default values
default_user="${USER}"
default_host="fde1:2240:a1ef::1"
target=".#nixosConfigurations.default.config.system.build.toplevel"
no_aleph_builder=false

log_info() { gum log --level info "$*"; }
log_warn() { gum log --level warn "$*"; }
log_error() { gum log --level error "$*"; }

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

if [ "$no_aleph_builder" = false ] && ! ( ([ "$(uname -m)" = "aarch64" ] && [ "$(uname)" = "Linux" ]) ||
  ([ -f /etc/nix/machines ] && grep -q 'aarch64-linux' /etc/nix/machines)); then
  log_warn "No aarch64-linux builder found, falling back to building on Aleph (slow)"
  build_cmd="nom build --accept-flake-config --eval-store auto --store ssh-ng://$user@$host $target --print-out-paths"
  log_info "Running: $build_cmd"
  out_path=$(eval "$build_cmd")
else
  build_cmd="nom build --accept-flake-config $target --print-out-paths"
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
