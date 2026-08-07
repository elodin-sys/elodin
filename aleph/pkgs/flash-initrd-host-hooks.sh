ALEPH_FLASH_IFACE=enx327005180101

aleph_flash_prepare_host() {
  exec > >(tee /tmp/flash-initrd-$(date +%s).log) 2>&1
  echo 2048 > /sys/module/usbcore/parameters/usbfs_memory_mb || true
  echo -1 > /sys/module/usbcore/parameters/autosuspend || true
  MM_WAS_ACTIVE=0
  if systemctl is-active --quiet ModemManager 2>/dev/null; then
    MM_WAS_ACTIVE=1
    systemctl stop ModemManager || true
  fi
  trap '[ "$MM_WAS_ACTIVE" = 1 ] && systemctl start ModemManager 2>/dev/null; type on_exit >/dev/null 2>&1 && on_exit || true' EXIT
}

aleph_flash_set_link() {
  @ip@ addr replace 192.168.7.1/24 dev "$ALEPH_FLASH_IFACE"
  @ip@ link set "$ALEPH_FLASH_IFACE" up
}

aleph_flash_start_sideload() {
  echo "Starting sideload server for the flash initrd..."
  for _ in {1..90}; do
    @ip@ link show "$ALEPH_FLASH_IFACE" >/dev/null 2>&1 && break
    sleep 1
  done
  @ip@ link show "$ALEPH_FLASH_IFACE" >/dev/null 2>&1 ||
    { echo "ERR: gadget ethernet $ALEPH_FLASH_IFACE did not appear" >&2; exit 3; }
  command -v nmcli >/dev/null 2>&1 && nmcli dev set "$ALEPH_FLASH_IFACE" managed no || true
  aleph_flash_set_link
  (while true; do aleph_flash_set_link 2>/dev/null; sleep 5; done) &
  KEEPER_PID=$!
  @python@ -m http.server 8080 --bind 192.168.7.1 --directory @flashPayload@ &
  HTTP_PID=$!
  trap 'kill $HTTP_PID $KEEPER_PID 2>/dev/null || true; command -v nmcli >/dev/null 2>&1 && nmcli dev set "$ALEPH_FLASH_IFACE" managed yes 2>/dev/null || true; [ "$MM_WAS_ACTIVE" = 1 ] && systemctl start ModemManager 2>/dev/null; type on_exit >/dev/null 2>&1 && on_exit || true' EXIT
}
