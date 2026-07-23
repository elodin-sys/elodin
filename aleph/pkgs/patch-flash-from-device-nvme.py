#!/usr/bin/env python3
"""Patch jetpack flash-from-device.sh to write NVMe partitions (9:0 and 12:*)."""

import sys
from pathlib import Path

# NVIDIA tegraflash maps nvme instance 0 → device 12:0 (mbr_12_0.bin).
# Stock flash_t234_qspi_nvme.xml uses instance 4 → 12:4 (no GPT name map).
# External storage fallback is 9:0. Treat all of these as /dev/nvme0n1.
NVME_TEST = '[[ "$devnum" -eq 9 && "$instnum" -eq 0 ]] || [[ "$devnum" -eq 12 ]]'


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
      file_size=$(stat -c "%s" "$partfile")
      if ! dd if="$partfile" of="/dev/nvme0n1" bs=1M seek="$start_location" oflag=seek_bytes conv=fsync >/dev/null; then
        echo "ERR: failed writing $partfile to /dev/nvme0n1" >&2
        return 1
      fi
    fi
  done <flash.idx
}}"""
    if write_needle not in text:
        write_needle_fsync = write_needle.replace(
            "oflag=seek_bytes >/dev/null",
            "oflag=seek_bytes conv=fsync >/dev/null",
        )
        write_repl_fsync = write_repl.replace(
            'of="/dev/mmcblk0" bs=4096 seek="$start_location" oflag=seek_bytes >/dev/null',
            'of="/dev/mmcblk0" bs=4096 seek="$start_location" oflag=seek_bytes conv=fsync >/dev/null',
        )
        if write_needle_fsync in text:
            text = text.replace(write_needle_fsync, write_repl_fsync, 1)
        else:
            raise SystemExit("write_partitions() pattern not found")
    else:
        text = text.replace(write_needle, write_repl, 1)

    main_needle = """steps=$(expr "$(wc -l <flash.idx)" + "1")

erase_bootdev
write_partitions

echo Finished flashing device
"""
    main_repl = f"""needs_nvme=0
while IFS=", " read -r _pn partloc _a _b _c _d _e _f; do
  devnum=$(echo "$partloc" | cut -d':' -f 1)
  instnum=$(echo "$partloc" | cut -d':' -f 2)
  if {NVME_TEST}; then
    needs_nvme=1
    break
  fi
done <flash.idx

if [[ "$needs_nvme" -eq 1 ]]; then
  for i in $(seq 1 60); do
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
  echo "Discarding /dev/nvme0n1..."
  blkdiscard -f /dev/nvme0n1 || true
fi

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
    print(f"Patched {path} for NVMe (9:0 / 12:*)")


if __name__ == "__main__":
    main()
