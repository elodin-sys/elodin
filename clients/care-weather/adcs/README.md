# ADCS

The Care Weather ADCS package wraps Elodin's MEKF and Basiliks's mrpFeedback into a complete attitude determination and control system.

## LITL 
To set up LITL (laptop in the loop) testing.
#### Dependencies
- Rust - https://rustup.rs
- Elodin - https://docs.elodin.systems/quickstart.
- Packages - clang & openblas
``` sh
sudo apt install clang libclang-dev libopenblas-dev # ubuntu
sudo dnf install clang libstdc++-static clang-tools-extra openblas-devel # fedora / rhel
```

#### Usage

1. The first step is to plug in the MCU or radio into your laptop. Once that is done, identify the path to the serial port. Usually, this is something like `/dev/tyyACM0` on Linux or `/dev/cu.usbmodem11201` on Mac. Then edit `mcu.path`, located on line 97, in the `config.toml` file with that path. 

2. Next, you can start the dashboard visualizer. Right now, this must be started before running the adcs process

``` sh
elodin editor dash.py
```

3. Now, finally, you can run the `adcs` process with `cargo run`
