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

{% alert(kind="warning") %}
The Elodin SDK and CLI are supported on macOS and Linux distributions with glibc 2.35+ (Ubuntu 22.04+, Debian 12+, Fedora 35+, NixOS 21.11+). Windows users run the simulation in Windows Subsystem for Linux ([install WSL](https://docs.microsoft.com/en-us/windows/wsl/install)) and the Editor natively on Windows.
{% end %}

The Elodin toolkit has two parts: the **Elodin CLI** (which bundles the editor, headless runner, and `elodin-db`) and the **Python SDK** (used to author simulations).

### Install the Elodin CLI

On macOS, Linux, or WSL:

```sh
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/elodin-sys/elodin/releases/latest/download/elodin-installer.sh | sh
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/elodin-sys/elodin/releases/latest/download/elodin-db-installer.sh | sh
```

On Windows, download and run the latest `elodin-x86_64-pc-windows-msvc.msi` from the [releases page](https://github.com/elodin-sys/elodin/releases/latest).

Verify the install:

```sh
elodin --version
elodin-db --version
```

### Install the Elodin Python SDK

Install [`uv`](https://docs.astral.sh/uv/) if you don't already have it:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create a virtual environment and install the SDK:

```sh
uv venv && source .venv/bin/activate
uv pip install -U elodin
```

### Upgrading?

If upgrading from an older Elodin version, consult the [migration guides](/reference/migration/).

## Start Simulating

The examples below use [`examples/three-body/main.py`](https://github.com/elodin-sys/elodin/blob/main/examples/three-body/main.py). Clone the elodin repo to follow along:

```sh
git clone https://github.com/elodin-sys/elodin.git && cd elodin
```

### macOS / Linux

Run the simulation and open the Editor in one command:

```sh
elodin editor examples/three-body/main.py
```

### Windows (WSL)

Windows users run the simulation in WSL and connect the natively-installed Editor over `localhost`.

{% alert(kind="info") %}
WSL needs mirrored networking so the Editor on Windows can reach the simulation on `127.0.0.1`. In an admin PowerShell, create `%USERPROFILE%\.wslconfig` containing the snippet below, then run `wsl --shutdown` to apply.
{% end %}

```ini
[wsl2]
networkingMode=mirrored
```

In a WSL terminal, start the simulation:

```sh
elodin run examples/three-body/main.py
```

In a Windows PowerShell terminal, launch the Editor and connect:

```powershell
elodin.exe editor 127.0.0.1:2240
```


## Perform Analysis

To analyze simulation data, use the `Exec` API to run the simulation for some number of ticks and collect the historical component data as a [Polars DataFrame].
The DataFrame can then be used to generate plots or perform other methods of data analysis.

Run the bouncing ball example code to see this in action:

The `ball/plot.py` example depends on `matplotlib`. Install it with `uv`:

```sh
uv pip install -U matplotlib
```

Then run the ball plot example (inside the same activated `.venv`):
```sh
python examples/ball/plot.py
```

For more information on data frames check out
[Polars DataFrame](https://docs.pola.rs/user-guide/concepts/data-structures/#dataframe)

## Next Steps

Try out the following tutorials to learn how to build simulations using Elodin:

{% cardlink(title="Three-Body Orbit Tutorial", icon="planet", href="/home/3-body") %}
Learn how to model a basic stable three-body problem
{% end %}
