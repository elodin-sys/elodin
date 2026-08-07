#!/usr/bin/env bash
# Fix up jetpack signedFirmware for Aleph NVMe initrd flash:
# - Remove OS image copies from the tree: the device caps RCM blobs between
#   383 MB (accepted) and 803 MB (rejected), so images cannot ride in the
#   initrd and are sideloaded over the USB ethernet gadget instead
# - Point flash.idx esp/APP rows at the host HTTP payload URLs
# - Fail if a boardspec flash.idx lacks NVMe esp or APP rows
set -euo pipefail

src="$1"
out="$2"
base_url="$3"

cp -a "$src"/. "$out/"
chmod -R u+w "$out"

for dir in "$out"/*/; do
  [ -d "$dir" ] || continue
  [ -f "$dir/flash.idx" ] || continue

  # Drop image copies tegraflash placed in SKU dirs; they must not enter the initrd
  rm -f "$dir/esp.img" "$dir/system.img"

  # flash.idx fields: num, loc, start, size, file, filesize, attrs, sha.
  # URLs are comma/space-free, safe for the IFS=", " parser in flash-from-device.
  python3 - "$dir/flash.idx" "$base_url" <<'PY'
import re
import sys

path, base_url = sys.argv[1], sys.argv[2]
lines = open(path).read().splitlines()
out_lines = []
fixed_esp = 0
fixed_app = 0
for line in lines:
    m_esp = re.match(r"^(\d+, 12:0:esp, \d+, \d+), [^,]*, [^,]*, (.*)$", line)
    m_app = re.match(r"^(\d+, 12:0:APP, \d+, \d+), [^,]*, [^,]*, (.*)$", line)
    if m_esp:
        line = f"{m_esp.group(1)}, {base_url}/esp.img.gz, 0, {m_esp.group(2)}"
        fixed_esp += 1
    elif m_app:
        line = f"{m_app.group(1)}, {base_url}/system.img.gz, 0, {m_app.group(2)}"
        fixed_app += 1
    out_lines.append(line)
if fixed_esp != 1:
    sys.exit(f"{path}: expected one NVMe esp row, found {fixed_esp}")
if fixed_app != 1:
    sys.exit(f"{path}: expected one NVMe APP row, found {fixed_app}")
open(path, "w").write("\n".join(out_lines) + "\n")
PY
done

echo "Fixed signed firmware: esp/APP rows -> $base_url sideload URLs, no embedded images"
