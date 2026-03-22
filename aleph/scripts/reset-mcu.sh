#!/bin/sh -e

boot0_pidfile="${ALEPH_BOOT0_PIDFILE:-/run/reset-mcu-boot0.pid}"
boot0_chip="${ALEPH_BOOT0_GPIOCHIP:-gpiochip0}"
boot0_line="${ALEPH_BOOT0_GPIOLINE:-144}"

nrst_pidfile="${ALEPH_NRST_PIDFILE:-/run/reset-mcu-nrst.pid}"
nrst_chip="${ALEPH_NRST_GPIOCHIP:-gpiochip0}"
nrst_line="${ALEPH_NRST_GPIOLINE:-106}"

kill_gpio_holder() {
  pidfile="$1"
  pattern="$2"

  if [ -f "$pidfile" ]; then
    gpio_pid="$(tr -d '\n' < "$pidfile")"
    if [ -n "$gpio_pid" ]; then
      kill "$gpio_pid" 2>/dev/null || true
      wait "$gpio_pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  fi

  pkill -f "$pattern" 2>/dev/null || true
}

set_nrst_low() {
  kill_nrst_holder
  gpioset --mode=signal "$nrst_chip" "$nrst_line=0" >/dev/null 2>&1 &
  nrst_pid="$!"
  printf '%s\n' "$nrst_pid" > "$nrst_pidfile"
}

set_nrst_high() {
  kill_nrst_holder
  gpioset --mode=signal "$nrst_chip" "$nrst_line=1" >/dev/null 2>&1 &
  nrst_pid="$!"
  printf '%s\n' "$nrst_pid" > "$nrst_pidfile"
}

kill_boot0_holder() {
  kill_gpio_holder "$boot0_pidfile" "gpioset .*${boot0_chip} ${boot0_line}=1"
}

kill_nrst_holder() {
  kill_gpio_holder "$nrst_pidfile" "gpioset .*${nrst_chip} ${nrst_line}="
}

start_boot0_holder() {
  kill_boot0_holder
  gpioset --mode=signal "$boot0_chip" "$boot0_line=1" >/dev/null 2>&1 &
  boot0_pid="$!"
  printf '%s\n' "$boot0_pid" > "$boot0_pidfile"
}

enter_bootloader() {
  set_nrst_low
  sleep 1
  start_boot0_holder
  sleep 3
  set_nrst_high
  sleep 2
}

boot_application() {
  kill_boot0_holder
  set_nrst_low
  sleep 1
  set_nrst_high
}

case "${1:-}" in
  --bootloader)
    enter_bootloader
    ;;
  --boot0-high)
    start_boot0_holder
    ;;
  --release-boot0)
    kill_boot0_holder
    ;;
  --nrst-low)
    set_nrst_low
    ;;
  --nrst-high)
    set_nrst_high
    ;;
  --app|"")
    boot_application
    ;;
  --help)
    echo "Usage: $0 [--bootloader|--app|--boot0-high|--release-boot0|--nrst-low|--nrst-high]"
    ;;
  *)
    echo "Usage: $0 [--bootloader|--app|--boot0-high|--release-boot0|--nrst-low|--nrst-high]" >&2
    exit 1
    ;;
esac
