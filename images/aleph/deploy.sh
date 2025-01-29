#!/usr/bin/env bash
aleph_ip="fde1:2240:a1ef::1"
set -euxo pipefail
export CONFIG="$(nix eval .#toplevel --raw)"
echo "deploying $CONFIG"
nix copy --no-check-sigs .#toplevel --to ssh-ng://$aleph_ip
ssh $aleph_ip "sudo nix-env -p /nix/var/nix/profiles/system --set ${CONFIG} && sudo ${CONFIG}/bin/switch-to-configuration switch;"
