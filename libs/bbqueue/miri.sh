#!/bin/bash

# We disable isolation because we use tokio sleep
# We disable leaks because ???
#
# TODO: Can we eliminate some of these limitations for testing?
MIRIFLAGS="-Zmiri-disable-isolation -Zmiri-ignore-leaks" \
    cargo +nightly miri test \
    --target x86_64-unknown-linux-gnu \
    --features=std
