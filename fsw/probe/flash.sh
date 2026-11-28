#!/bin/sh -eu

# Flashing the Aleph Flight Controller debug probe (U2): USB DFU through the STM32 ROM, or SWD
# from an external probe on the bottom pads TP1 (SWCLK) and TP2 (SWDIO).

OPENOCD=${OPENOCD:-openocd}
ADAPTER_SPEED=${ADAPTER_SPEED:-1000}
RTT_PORT=${RTT_PORT:-9090}
DFU_UTIL=${DFU_UTIL:-dfu-util}
DFU_DEVICE=${DFU_DEVICE:-0483:df11}
DFU_ADDRESS=${DFU_ADDRESS:-0x08000000}
DAP_DEVICE=${DAP_DEVICE:-1209:0001}
DAP_SERIAL=${DAP_SERIAL:-}

# PA14 is both SWCLK and BOOT0; launch the user flash vector explicitly after a
# reset in case a board still has the ROM sampling that pin.
FLASH_LAUNCH='mww 0xE000ED08 0x08000000; set _sp [mrw 0x08000000]; set _pc [expr {[mrw 0x08000004] & 0xfffffffe}]; reg msp $_sp; reg pc $_pc; resume'

# Earlier firmware cleared nBOOT_SEL so the ROM sampled PA14/BOOT0. The board
# no longer has a button and PA14 is SWCLK, so restore the factory setting.
OPT_FACTORY='set _optr [mrw 0x40022020]
if {($_optr & 0xff) != 0xaa} {
    echo [format "option bytes: RDP is not level 0 (FLASH_OPTR=0x%08x); refusing" $_optr]
    shutdown error
}
if {($_optr & 0x01000000) != 0} {
    echo [format "option bytes: factory boot config (FLASH_OPTR=0x%08x)" $_optr]
} else {
    echo [format "option bytes: FLASH_OPTR=0x%08x, setting nBOOT_SEL" $_optr]
    stm32l4x option_write 0 0x20 0x01000000 0x01000000
    stm32l4x option_load 0
}'

cd "$(dirname "$0")"

usage() {
    cat <<'EOF'
usage:
  ./flash.sh dfu [blinky|dap]        flash over USB DFU; default dap
  ./flash.sh flash [target-or-elf]   flash over SWD + verify + run; default dap
  ./flash.sh detach                  send a running dap image to ROM DFU
  ./flash.sh reset                   reset + run already-flashed image
  ./flash.sh optbytes                restore the factory boot config (nBOOT_SEL=1)
  ./flash.sh rtt [target-or-elf]     serve RTT channel 0 on localhost:9090
  ./flash.sh info                    read target ID / flash info / vectors
  ./flash.sh probe                   show OpenOCD version

`dfu` needs only USB-C on the board: it detaches a running dap image into ROM
DFU, or uses the ROM directly (a blank part boots to it). SWD commands need an
external CMSIS-DAP probe on TP1 (SWCLK), TP2 (SWDIO) and GND; the probe's NRST
is not brought out. `blinky` has no USB, so flashing it over DFU leaves only
SWD to get back. With more than one probe on the bus, set DAP_SERIAL to pick
one (both paths honour it).

env:
  OPENOCD=openocd        OpenOCD executable
  ADAPTER_SPEED=1000     SWD speed in kHz
  RTT_PORT=9090          TCP port for RTT channel 0
  DFU_UTIL=dfu-util      dfu-util executable (brew install dfu-util)
  DFU_DEVICE=0483:df11   USB VID:PID of the STM32 ROM bootloader
  DAP_DEVICE=1209:0001   USB VID:PID of the dap image (DFU run-time interface)
  DAP_SERIAL=            USB serial of the probe to use, when several are attached
EOF
}

elf_for() {
    case "${1:-dap}" in
    blinky|dap)
        printf 'build/%s.elf\n' "$1"
        ;;
    *.elf|*/*.elf)
        printf '%s\n' "$1"
        ;;
    *)
        echo "unknown firmware target: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
}

require_elf() {
    elf=$(elf_for "${1:-dap}")
    if [ ! -f "$elf" ]; then
        echo "missing $elf; run ./build.sh first" >&2
        exit 1
    fi
    printf '%s\n' "$elf"
}

require_bin() {
    case "${1:-dap}" in
    blinky|dap) bin="build/$1.bin" ;;
    *)
        echo "dfu needs a raw image: blinky or dap" >&2
        exit 2
        ;;
    esac
    if [ ! -f "$bin" ]; then
        echo "missing $bin; run ./build.sh first" >&2
        exit 1
    fi
    printf '%s\n' "$bin"
}

find_nm() {
    if command -v arm-none-eabi-nm >/dev/null 2>&1; then
        command -v arm-none-eabi-nm
        return
    fi

    for nm in tools/*/bin/arm-none-eabi-nm; do
        if [ -x "$nm" ]; then
            printf '%s\n' "$nm"
            return
        fi
    done

    echo "arm-none-eabi-nm not found; run ./build.sh first" >&2
    exit 1
}

openocd_probe() {
    "$OPENOCD" \
        -f interface/cmsis-dap.cfg \
        ${DAP_SERIAL:+-c "adapter serial $DAP_SERIAL"} \
        -c 'transport select swd' \
        -f target/stm32g0x.cfg \
        -c "adapter speed $ADAPTER_SPEED" \
        "$@"
}

exec_openocd_probe() {
    exec "$OPENOCD" \
        -f interface/cmsis-dap.cfg \
        ${DAP_SERIAL:+-c "adapter serial $DAP_SERIAL"} \
        -c 'transport select swd' \
        -f target/stm32g0x.cfg \
        -c "adapter speed $ADAPTER_SPEED" \
        "$@"
}

probe() {
    "$OPENOCD" --version
}

info() {
    openocd_probe \
        -c 'init' \
        -c 'halt' \
        -c 'flash probe 0' \
        -c 'flash info 0' \
        -c 'reg pc' \
        -c 'reg msp' \
        -c 'mdw 0x08000000 16' \
        -c 'resume' \
        -c 'shutdown'
}

flash() {
    elf=$(require_elf "${1:-dap}")
    openocd_probe \
        -c 'init' \
        -c 'halt' \
        -c "program $elf verify" \
        -c 'reset halt' \
        -c "$FLASH_LAUNCH" \
        -c 'shutdown'
}

optbytes() {
    openocd_probe \
        -c 'init' \
        -c 'halt' \
        -c 'flash probe 0' \
        -c "$OPT_FACTORY" \
        -c 'reset halt' \
        -c "$FLASH_LAUNCH" \
        -c 'shutdown'
}

require_dfu_util() {
    if ! command -v "$DFU_UTIL" >/dev/null 2>&1; then
        echo "$DFU_UTIL not found; brew install dfu-util" >&2
        exit 1
    fi
}

dfu_list() {
    "$DFU_UTIL" -l -d "$1" 2>/dev/null | grep -i "found $2" |
        { if [ -n "$DAP_SERIAL" ]; then grep "serial=\"$DAP_SERIAL\""; else cat; fi; }
}

dfu_present() {
    [ -n "$(dfu_list "$1" "$2")" ]
}

# The ROM keeps the USB port but not the serial, so the port path selects the
# board in both modes.
dap_path() {
    list=$(dfu_list "$DAP_DEVICE" runtime)
    if [ "$(printf '%s\n' "$list" | grep -c .)" -gt 1 ]; then
        echo "several dap images on the bus; set DAP_SERIAL to one of:" >&2
        printf '%s\n' "$list" | sed -n 's/.*serial="\([^"]*\)".*/  \1/p' >&2
        exit 1
    fi
    printf '%s\n' "$list" | sed -n 's/.*path="\([^"]*\)".*/\1/p'
}

# On `:leave` the ROM disconnects and jumps to the image (AN3156 §5.5) before
# dfu-util's last status poll, so a completed download can still exit 74; go by
# the "File downloaded successfully" line.
# -E 1: the ROM enumerates well within a second of a detach (default is 5 s).
dfu_download() {
    log=$(mktemp)
    rc=0
    "$DFU_UTIL" -d "$1" ${2:+-p "$2"} -E 1 -a 0 -s "$DFU_ADDRESS:leave" -D "$bin" >"$log" 2>&1 || rc=$?
    cat "$log"
    if grep -q 'File downloaded successfully' "$log"; then rc=0; fi
    rm -f "$log"
    return "$rc"
}

dfu_wait_dap() {
    tries=0
    while [ $tries -lt 50 ]; do
        if dfu_present "$DAP_DEVICE" runtime; then
            echo "dap image back at $DAP_DEVICE" >&2
            return 0
        fi
        sleep 0.2
        tries=$((tries + 1))
    done
    echo "dap image did not reappear at $DAP_DEVICE within 10 s" >&2
    return 1
}

# ROM DFU over USB, no probe needed. Alt setting 0 is @Internal Flash; `:leave`
# exits DFU into the freshly written image. Given the run-time and DFU-mode IDs
# as a pair, dfu-util detaches a running dap image itself.
dfu() {
    target=${1:-dap}
    bin=$(require_bin "$target")
    require_dfu_util

    if dfu_present "$DAP_DEVICE" runtime; then
        path=$(dap_path)
        echo "dap image at $DAP_DEVICE port $path: detaching into ROM DFU" >&2
        dfu_download "$DAP_DEVICE,$DFU_DEVICE" "$path"
    elif dfu_present "$DFU_DEVICE" dfu; then
        dfu_download "$DFU_DEVICE" ""
    else
        echo "no dap image at $DAP_DEVICE and no ROM DFU at $DFU_DEVICE" >&2
        exit 1
    fi

    if [ "$target" = dap ]; then
        dfu_wait_dap
    fi
}

detach() {
    require_dfu_util

    if ! dfu_present "$DAP_DEVICE" runtime; then
        echo "no dap image at $DAP_DEVICE" >&2
        exit 1
    fi

    path=$(dap_path)
    "$DFU_UTIL" -d "$DAP_DEVICE" -p "$path" -e
}

reset_run() {
    openocd_probe \
        -c 'init' \
        -c 'reset halt' \
        -c "$FLASH_LAUNCH" \
        -c 'shutdown'
}

rtt() {
    elf=$(require_elf "${1:-dap}")
    nm=$(find_nm)
    rtt_addr=0x$("$nm" "$elf" | awk '$3 == "_SEGGER_RTT" { print $1 }')
    if [ "$rtt_addr" = "0x" ]; then
        echo "_SEGGER_RTT not found in $elf" >&2
        exit 1
    fi

    echo "RTT channel 0: nc localhost $RTT_PORT" >&2
    exec_openocd_probe \
        -c 'init' \
        -c "rtt setup $rtt_addr 0x100 {SEGGER RTT}" \
        -c 'rtt polling_interval 20' \
        -c 'rtt start' \
        -c 'rtt channels' \
        -c "rtt server start $RTT_PORT 0"
}

cmd=${1:-help}
case "$cmd" in
probe)
    probe
    ;;
info)
    info
    ;;
flash)
    flash "${2:-dap}"
    ;;
reset)
    reset_run
    ;;
optbytes)
    optbytes
    ;;
dfu)
    dfu "${2:-dap}"
    ;;
detach)
    detach
    ;;
rtt)
    rtt "${2:-dap}"
    ;;
help|-h|--help)
    usage
    ;;
*)
    flash "$cmd"
    ;;
esac
