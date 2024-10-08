#!/usr/bin/env bash
set -euxo pipefail
export CONFIG="$(nix eval .#toplevel --raw)"
echo "deploying $CONFIG"
nix copy --no-check-sigs .#toplevel --to ssh-ng://10.224.0.1
ssh 10.224.0.1 "sudo ${CONFIG}/bin/switch-to-configuration boot; sudo ${CONFIG}/bin/switch-to-configuration switch;"
