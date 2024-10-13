+++
title = "Quick Start"
description = "Install Elodin and start simulating."
draft = false
weight = 101
sort_by = "weight"
template = "get-started/page.html"

[extra]
lead = "Install Elodin and start simulating."
toc = true
top = false
order = 1
icon = ""
+++

## Install

Download the Elodin Client:

| File                                                    | Platform            | Checksum                        |
| ------------------------------------------------------- | ------------------- | ------------------------------- |
| [elodin-aarch64-apple-darwin.tar.gz][elodin-macos]      | Apple Silicon macOS | [sha256][elodin-macos-sha256]   |
| [elodin-x86_64-unknown-linux-gnu.tar.gz][elodin-linux]  | x64 Linux           | [sha256][elodin-linux-sha256]   |
| [elodin-x86_64-pc-windows-msvc.zip][elodin-windows]     | x64 Windows         | [sha256][elodin-windows-sha256] |

[elodin-macos]: https://storage.googleapis.com/elodin-releases/latest/elodin-aarch64-apple-darwin.tar.gz
[elodin-macos-sha256]: https://storage.googleapis.com/elodin-releases/latest/elodin-aarch64-apple-darwin.tar.gz.sha256
[elodin-linux]: https://storage.googleapis.com/elodin-releases/latest/elodin-x86_64-unknown-linux-gnu.tar.gz
[elodin-linux-sha256]: https://storage.googleapis.com/elodin-releases/latest/elodin-x86_64-unknown-linux-gnu.tar.gz.sha256
[elodin-windows]: https://storage.googleapis.com/elodin-releases/latest/elodin-x86_64-pc-windows-msvc.zip
[elodin-windows-sha256]: https://storage.googleapis.com/elodin-releases/latest/elodin-x86_64-pc-windows-msvc.zip.sha256

Install the Elodin Python SDK using `pip`:

{% alert(kind="warning") %}
The SDK is only supported on macOS and Linux distributions with glibc 2.35+ (Ubuntu 22.04+, Debian 12+, Fedora 35+, NixOS 21.11+). Windows users can still use Elodin by installing and running the simulation server in Windows Subsystem for Linux. Install the Elodin Python SDK in WSL, after [installing WSL.](https://docs.microsoft.com/en-us/windows/wsl/install)
{% end %}


```sh
pip install -U elodin
```

## Start Simulating

### Windows (WSL)

To use Elodin on Windows, the simulation server must run in Windows Subsystem for Linux (WSL). The Elodin Client itself can run natively on Windows.

[Video Walkthrough](https://www.loom.com/share/efcbf81e43074863807750d4ad2f8d7a?sid=9403e8c8-7893-4299-824e-2dacb6978120)

In a Windows terminal launch the Elodin app.

```wsl
.\elodin.exe
```

In a WSL terminal download and install `elodin` binary into your path then run:

1. Create a new simulation using the three-body orbit template.
    ```sh
    elodin create --template three-body
    ```
2. Run the simulation server.
    ```sh
    elodin run three-body.py
    ```

### Linux / macOS

1. Create a new simulation using the three-body orbit template.
    ```sh
    elodin create --template three-body
    ```
2. Launch the simulation using the `elodin` CLI.
    ```sh
    elodin editor three-body.py
    ```

## Perform Analysis

To analyze simulation data, use the `Exec` API to run the simulation for some number of ticks and collect the historical component data as a [Polars DataFrame].
The DataFrame can then be used to generate plots or perform other methods of data analysis.

Run the bouncing ball example code to see this in action:

The `ball/plot.py` example depends on `matplotlib`. Install it using `pip`:

```sh
pip install -U matplotlib
```

Then create & run the ball template:
```sh
elodin create --template ball
python3 ball/plot.py
```

For more information on data frames check out
[Polars DataFrame](https://docs.pola.rs/user-guide/concepts/data-structures/#dataframe)

## Monte Carlo

Run Monte Carlo simulations to explore the state space.

1. Create an account at https://app.elodin.systems to receive 60 free minutes of hosted simulation time (per month).
2. Authorize `elodin` to access the Monte Carlo platform.
    ```sh
    elodin login
    ```
3. Create a new simulation from the bouncing ball template, which includes random sampling and asserts.
    ```sh
    elodin create --template ball
    ```
4. Start a 100 sample Monte Carlo run with a maximum sim duration of 15s.
    {% alert(kind="notice") %}Add `--open` to automatically open the dashboard url in the browser.{% end %}
    ```sh
    elodin monte-carlo run --name ball ball/main.py --max-duration 15 --samples 100
    ```

## Next Steps

Try out the following tutorials to learn how to build simulations using Elodin:

{% cardlink(title="Three-Body Orbit Tutorial", icon="planet", href="/get-started/3-body/") %}
Learn how to model a basic stable three-body problem
{% end %}
