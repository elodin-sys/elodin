#!/bin/sh
set -eu

cd "$(dirname "$0")"

CC="${CC:-cc}"
CFLAGS="-std=c2x -O1 -g -Wall -Wextra -Werror -Wno-unused-function -fsanitize=address,undefined"
CXX="${CXX:-c++}"
CXXFLAGS="-std=gnu++20 -O1 -g -Wall -Wextra -Werror -Wno-unused-function -Wno-c99-extensions -Wno-missing-field-initializers -fsanitize=address,undefined"

mkdir -p build
"$CC" $CFLAGS boot.c -o build/boot
ASAN_OPTIONS=strict_string_checks=1 ./build/boot
"$CC" $CFLAGS cmsis_dap.c -o build/cmsis_dap
ASAN_OPTIONS=strict_string_checks=1 ./build/cmsis_dap
"$CC" $CFLAGS swd_wire.c -o build/swd_wire
ASAN_OPTIONS=strict_string_checks=1 ./build/swd_wire
"$CXX" $CXXFLAGS usb_model.cpp -o build/usb_model
ASAN_OPTIONS=strict_string_checks=1 ./build/usb_model
"$CXX" $CXXFLAGS cdc_uart.cpp -o build/cdc_uart
ASAN_OPTIONS=strict_string_checks=1 ./build/cdc_uart
