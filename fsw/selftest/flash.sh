#!/bin/sh -e
# Flashes the self-test into the STM32H747 through the onboard debug probe and
# restarts it. Add --connect-under-reset if a blank or crashed part won't attach.
cd "$(dirname "$0")"
probe-rs download --chip STM32H747XI "$@" build/selftest.elf
probe-rs reset --chip STM32H747XI "$@"
