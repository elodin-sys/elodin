# Firmware host tests

Run the complete suite from the repository root:

```sh
./firmware/probe/tests/run.sh
```

The script builds and runs five sanitizer-enabled tests:

- `boot.c`: bootloader handover and reset accounting against a modeled
  reset-cause register and system memory vector.
- `cmsis_dap.c`: commands, memory transfers, error recovery, and malformed
  packets against a small DP/MEM-AP model.
- `swd_wire.c`: SWD framing, parity, ACKs, turnaround, clock selection, and
  safe disconnect against a GPIO wire model, with and without a direction pin.
- `usb_model.cpp`: enumeration, bulk packet slots, STM32 endpoint-register
  semantics, class/vendor control-request forwarding, the DFU run-time
  interface, suspend/resume, reconfiguration, and reset.
- `cdc_uart.cpp`: the EP0 OUT data stage, CDC-ACM line coding, and USART
  RX/TX bridging against the USB model and a USART model.

`usb_model.h` holds the USB register and packet-memory model the last two share.

Set `CC` and `CXX` to select another compiler:

```sh
CC=gcc CXX=g++ ./firmware/probe/tests/run.sh
```

Keep tests deterministic and focused on externally visible behavior. Avoid
assertions about internal call counts, allocation offsets, or incidental state.
Add a regression that fails before each bug fix.
