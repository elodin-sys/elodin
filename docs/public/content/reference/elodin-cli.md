+++
title = "Elodin CLI"
description = "Elodin CLI"
draft = false
weight = 103
sort_by = "weight"
template = "reference/page.html"

[extra]
toc = true
top = false
icon = ""
order = 3
+++

# Command-Line Help for `elodin`

This document contains the help content for the `elodin` command-line program.

**Command Overview:**

* [`elodin`↴](#elodin)
* [`elodin login`↴](#elodin-login)
* [`elodin monte-carlo`↴](#elodin-monte-carlo)
* [`elodin monte-carlo run`↴](#elodin-monte-carlo-run)
* [`elodin monte-carlo download-results`↴](#elodin-monte-carlo-download-results)
* [`elodin editor`↴](#elodin-editor)
* [`elodin run`↴](#elodin-run)
* [`elodin create`↴](#elodin-create)

## `elodin`

**Usage:** `elodin [OPTIONS] [COMMAND]`

###### **Subcommands:**

* `login` — Obtain access credentials for your user account
* `monte-carlo` — Manage your Monte Carlo runs
* `editor` — Launch the Elodin editor (default)
* `run` — Run an Elodin simulaton in headless mode
* `create` — Create template

###### **Options:**

* `-u`, `--url <URL>`

  Default value: `https://app.elodin.systems`



## `elodin login`

Obtain access credentials for your user account

**Usage:** `elodin login`



## `elodin monte-carlo`

Manage your Monte Carlo runs

**Usage:** `elodin monte-carlo <COMMAND>`

###### **Subcommands:**

* `run` — Create and submit a Monte Carlo run
* `download-results` — Download the results of a Monte Carlo sample



## `elodin monte-carlo run`

Create and submit a Monte Carlo run

**Usage:** `elodin monte-carlo run [OPTIONS] --name <NAME> <FILE>`

###### **Arguments:**

* `<FILE>` — Path to the simulation configuration

###### **Options:**

* `-n`, `--name <NAME>` — Name of the Monte Carlo run
* `-s`, `--samples <SAMPLES>` — Number of samples to run

  Default value: `100`
* `-m`, `--max-duration <MAX_DURATION>` — Max simulation duration in seconds

  Default value: `10`
* `--open` — Open the dashboard in the browser



## `elodin monte-carlo download-results`

Download the results of a Monte Carlo sample

**Usage:** `elodin monte-carlo download-results --run-id <RUN_ID> --batch-number <BATCH_NUMBER> <PATH>`

###### **Arguments:**

* `<PATH>` — Path to download the results to

###### **Options:**

* `-r`, `--run-id <RUN_ID>` — ID of the Monte Carlo run
* `-b`, `--batch-number <BATCH_NUMBER>` — Number of the batch



## `elodin editor`

Launch the Elodin editor (default)

**Usage:** `elodin editor [addr/path]`

###### **Arguments:**

* `<addr/path>`

  Default value: `127.0.0.1:2240`



## `elodin run`

Run an Elodin simulaton in headless mode

**Usage:** `elodin run [addr/path]`

###### **Arguments:**

* `<addr/path>`

  Default value: `127.0.0.1:2240`



## `elodin create`

Create template

**Usage:** `elodin create [OPTIONS] --template <TEMPLATE>`

###### **Options:**

* `-t`, `--template <TEMPLATE>` — Name of the template

  Possible values: `rocket`, `drone`, `cube-sat`, `three-body`, `ball`

* `-p`, `--path <PATH>` — Path where the result will be located

  Default value: `.`



<hr/>

<small><i>
    This document was generated automatically by
    <a href="https://crates.io/crates/clap-markdown"><code>clap-markdown</code></a>.
</i></small>

