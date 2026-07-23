#!/usr/bin/env bash
# Fix up jetpack signedFirmware for Aleph NVMe initrd flash:
# - Ensure each boardspec dir has esp.img + system.img
# - Repair flash.idx APP lines that tegraflash left with an empty filename
# - Hardlink large images across SKU dirs (cpio stores one copy in the initrd)
set -euo pipefail

src="$1"
out="$2"
esp_img="$3"
system_img="$4"

system_size=$(stat -c %s "$system_img")

mkdir -p "$out/shared"
cp -a "$src"/. "$out/"
chmod -R u+w "$out"

cp -f "$esp_img" "$out/shared/esp.img"
cp -f "$system_img" "$out/shared/system.img"

for dir in "$out"/*/; do
  [ -d "$dir" ] || continue
  base=$(basename "$dir")
  [ "$base" = shared ] && continue
  [ -f "$dir/flash.idx" ] || continue

  rm -f "$dir/esp.img" "$dir/system.img"
  ln "$out/shared/esp.img" "$dir/esp.img"
  ln "$out/shared/system.img" "$dir/system.img"

  # flash.idx fields: num, loc, start, size, file, filesize, attrs, sha
  # Tegra leaves APP file empty under NO_ROOTFS even when system.img exists.
  python3 - "$dir/flash.idx" "$system_size" <<'PY'
import re, sys
path, system_size = sys.argv[1], sys.argv[2]
lines = open(path).read().splitlines()
out_lines = []
for line in lines:
    m = re.match(
        r"^(\d+, (?:12:\d+|9:0):APP, \d+, \d+), , , (.*)$",
        line,
    )
    if m:
        line = f"{m.group(1)}, system.img, {system_size}, {m.group(2)}"
    out_lines.append(line)
open(path, "w").write("\n".join(out_lines) + "\n")
PY
done

echo "Fixed signed firmware APP/system.img entries; deduped into $out/shared"
