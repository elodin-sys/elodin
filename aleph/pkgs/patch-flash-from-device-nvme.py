#!/usr/bin/env python3
"""Patch jetpack flash-from-device.sh to write Aleph's NVMe partitions.

esp/APP rows carry http URLs (host sideload server over the ECM gadget);
those are streamed with wget | gunzip | dd instead of read from the initrd.
"""

import sys
from pathlib import Path

# The Aleph partition template maps NVMe instance 0 to flash.idx device 12:0.
NVME_TEST = '[[ "$devnum" -eq 12 && "$instnum" -eq 0 ]]'


def main() -> None:
    path = Path(sys.argv[1])
    text = path.read_text()

    disk_size_needle = """  elif [[ "$devnum" -eq 1 && "$instnum" -eq 3 ]] || [[ "$devnum" -eq 6 && "$instnum" -eq 0 ]]; then
    echo "$(($(cat /sys/block/mmcblk0/size) * $(cat /sys/block/mmcblk0/queue/hw_sector_size)))"
  else
    echo ""
  fi
}"""
    disk_size_repl = f"""  elif [[ "$devnum" -eq 1 && "$instnum" -eq 3 ]] || [[ "$devnum" -eq 6 && "$instnum" -eq 0 ]]; then
    echo "$(($(cat /sys/block/mmcblk0/size) * $(cat /sys/block/mmcblk0/queue/hw_sector_size)))"
  elif {NVME_TEST}; then
    echo "$(($(cat /sys/block/nvme0n1/size) * $(cat /sys/block/nvme0n1/queue/hw_sector_size)))"
  else
    echo ""
  fi
}}"""
    if disk_size_needle not in text:
        raise SystemExit("disk_size() pattern not found")
    text = text.replace(disk_size_needle, disk_size_repl, 1)

    write_needle = """    elif [[ "$devnum" -eq 1 && "$instnum" -eq 3 ]] || [[ "$devnum" -eq 6 && "$instnum" -eq 0 ]]; then
      report_step "Writing $partfile (size=$partsize) to $partname on /dev/mmcblk0 (offset=$start_location)"
      file_size=$(stat -c "%s" "$partfile")
      if ! dd if="$partfile" of="/dev/mmcblk0" bs=4096 seek="$start_location" oflag=seek_bytes >/dev/null; then
        return 1
      fi
    fi
  done <flash.idx
}"""
    write_repl = f"""    elif [[ "$devnum" -eq 1 && "$instnum" -eq 3 ]] || [[ "$devnum" -eq 6 && "$instnum" -eq 0 ]]; then
      report_step "Writing $partfile (size=$partsize) to $partname on /dev/mmcblk0 (offset=$start_location)"
      file_size=$(stat -c "%s" "$partfile")
      if ! dd if="$partfile" of="/dev/mmcblk0" bs=4096 seek="$start_location" oflag=seek_bytes >/dev/null; then
        return 1
      fi
    elif {NVME_TEST}; then
      report_step "Writing $partfile to $partname on /dev/nvme0n1 (offset=$start_location)"
      if [[ "$partfile" == http* ]]; then
        fetch_ok=0
        for attempt in 1 2 3 4 5; do
          # dd seeks to a fixed offset, so retrying after a partial write is safe
          if wget -q -O - "$partfile" | gzip -dc | dd of="/dev/nvme0n1" bs=1M seek="$start_location" oflag=seek_bytes conv=fsync >/dev/null; then
            fetch_ok=1
            break
          fi
          echo "WARN: sideload attempt $attempt failed for $partfile; retrying..."
          sleep 2
        done
        if [[ "$fetch_ok" -ne 1 ]]; then
          echo "ERR: failed sideloading $partfile to /dev/nvme0n1" >&2
          return 1
        fi
      else
        if ! dd if="$partfile" of="/dev/nvme0n1" bs=1M seek="$start_location" oflag=seek_bytes conv=fsync >/dev/null; then
          echo "ERR: failed writing $partfile to /dev/nvme0n1" >&2
          return 1
        fi
      fi
    fi
  done <flash.idx
}}"""
    if write_needle not in text:
        raise SystemExit("write_partitions() pattern not found")
    text = text.replace(write_needle, write_repl, 1)

    main_needle = """steps=$(expr "$(wc -l <flash.idx)" + "1")

erase_bootdev
write_partitions

echo Finished flashing device
"""
    main_repl = """for i in $(seq 1 60); do
  if [[ -b /dev/nvme0n1 ]]; then
    echo "NVMe device ready: /dev/nvme0n1"
    break
  fi
  echo "Waiting for /dev/nvme0n1 ($i/60)..."
  sleep 1
done
if [[ ! -b /dev/nvme0n1 ]]; then
  echo "ERR: /dev/nvme0n1 did not appear" >&2
  exit 1
fi

first_url=""
while IFS=", " read -r _pn _loc _a _b partfile _f _g _h; do
  case "$partfile" in
    http*) first_url="$partfile"; break ;;
  esac
done <flash.idx

if [[ -z "$first_url" ]]; then
  echo "ERR: flash.idx has no sideload URL" >&2
  exit 1
fi
probe="${first_url%/*}/"
echo "Waiting for sideload server at $probe..."
server_ok=0
for i in $(seq 1 90); do
  if wget -q -O /dev/null "$probe"; then
    server_ok=1
    break
  fi
  sleep 2
done
if [[ "$server_ok" -ne 1 ]]; then
  echo "ERR: sideload server unreachable: $probe" >&2
  exit 1
fi

echo "Discarding /dev/nvme0n1..."
blkdiscard -f /dev/nvme0n1 || true

steps=$(expr "$(wc -l <flash.idx)" + "1")

erase_bootdev
write_partitions

echo "Aleph NVMe write complete"
echo Finished flashing device
"""
    if main_needle not in text:
        raise SystemExit("main flash flow pattern not found")
    text = text.replace(main_needle, main_repl, 1)

    path.write_text(text)
    print(f"Patched {path} for Aleph NVMe sideloading")


if __name__ == "__main__":
    main()
