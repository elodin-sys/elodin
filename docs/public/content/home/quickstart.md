+++
title = "Quick Start"
description = "Install Elodin and start simulating."
draft = false
weight = 101
sort_by = "weight"

[extra]
lead = "Install Elodin and start simulating."
toc = true
top = false
order = 1
icon = ""
+++

## Install

Download the Elodin Client from the [releases](https://github.com/elodin-sys/elodin/releases) page.

Install the Elodin Python SDK using `pip`:

{% alert(kind="warning") %}
The SDK is only supported on macOS and Linux distributions with glibc 2.35+ (Ubuntu 22.04+, Debian 12+, Fedora 35+, NixOS 21.11+). Windows users can still use Elodin by installing and running the simulation server in Windows Subsystem for Linux. Install the Elodin Python SDK in WSL, after [installing WSL.](https://docs.microsoft.com/en-us/windows/wsl/install)
{% end %}


```sh
pip install -U elodin
```

### Upgrading? 

If upgrading from an old Elodin version, consult the [migration guides](/reference/migration/).

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

## Next Steps

Try out the following tutorials to learn how to build simulations using Elodin:

{% cardlink(title="Three-Body Orbit Tutorial", icon="planet", href="/home/3-body") %}
Learn how to model a basic stable three-body problem
{% end %}
