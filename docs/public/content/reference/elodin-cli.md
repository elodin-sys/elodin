+++
title = "Elodin CLI"
description = "Elodin CLI"
draft = false
weight = 104
sort_by = "weight"

[extra]
toc = true
top = false
icon = ""
order = 4
+++

# Command-Line Help for `elodin`

This document contains the help content for the `elodin` command-line program.

**Command Overview:**

* [`elodin`↴](#elodin)
* [`elodin login`↴](#elodin-login)
* [`elodin editor`↴](#elodin-editor)
* [`elodin run`↴](#elodin-run)
* [`elodin create`↴](#elodin-create)

## `elodin`

**Usage:** `elodin [OPTIONS] [COMMAND]`

###### **Subcommands:**

* `login` — Obtain access credentials for your user account
* `editor` — Launch the Elodin editor (default)
* `run` — Run an Elodin simulaton in headless mode
* `create` — Create template

###### **Options:**

* `-u`, `--url <URL>`

  Default value: `https://app.elodin.systems`



## `elodin login`

Obtain access credentials for your user account

**Usage:** `elodin login`



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
