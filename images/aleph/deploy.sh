#!/usr/bin/env bash
set -euxo pipefail
export CONFIG="$(nix eval .#toplevel --raw)"
echo "deploying $CONFIG"
nix copy --no-check-sigs .#toplevel --to ssh-ng://aleph.local
ssh aleph.local "sudo ${CONFIG}/bin/switch-to-configuration boot; sudo ${CONFIG}/bin/switch-to-configuration switch;"
