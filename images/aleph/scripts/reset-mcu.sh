#!/bin/sh -e

args="$@"
boot0_pin="gpiochip0 144"
nrst_pin="gpiochip0 106"

# if "--bootloader" is in args, then set boot0 to 1
if echo "$args" | grep -q -- "--bootloader"; then
  gpioset --mode=time --usec=20000 $boot0_pin=1 & sleep 0.001
fi

gpioset --mode=time --usec=100 $nrst_pin=0
