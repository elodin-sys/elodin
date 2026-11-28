#!/bin/sh -eu
# Builds build/*.bin: ExpressLRS 4.1.0 Unified_ESP32C3_2400_RX with a Flight
# Controller target (Generic C3 2400 layout), no binding phrase, Wi-Fi auto-on off.
# Uses the official release firmware.zip and its flasher.pyz (needs curl, unzip, python3).
cd "$(dirname "$0")"
URL=https://artifactory.expresslrs.org/ExpressLRS/a9d4a9cb5b5687c4c9d7e9e7fbdf44ad93651da6/firmware.zip
mkdir -p build/dl
[ -f build/dl/firmware.zip ] || curl -fsSL -o build/dl/firmware.zip "$URL"
rm -rf build/dl/fw && mkdir build/dl/fw && unzip -q build/dl/firmware.zip -d build/dl/fw
cd build/dl/fw/firmware
python3 - <<'PY'
import json
p = 'hardware/targets.json'; t = json.load(open(p))
t.setdefault('elodin', {'name': 'Elodin'}).setdefault('rx_2400', {})['flight_controller'] = {
    "product_name": "Aleph Flight Controller 2.4GHz RX", "lua_name": "Aleph FC RX",
    "layout_file": "Generic C3 2400.json", "upload_methods": ["uart", "betaflight"],
    "min_version": "3.5.0", "platform": "esp32-c3", "firmware": "Unified_ESP32C3_2400_RX"}
json.dump(t, open(p, 'w'), indent=4)
PY
python3 flasher.pyz --dir . --target elodin.rx_2400.flight_controller --flash dir --out ../../.. --no-auto-wifi 2>&1 | grep -iE 'error' || true
cp FCC/Unified_ESP32C3_2400_RX/bootloader.bin FCC/Unified_ESP32C3_2400_RX/partitions.bin FCC/Unified_ESP32C3_2400_RX/boot_app0.bin ../../..
ls -l ../../../*.bin
