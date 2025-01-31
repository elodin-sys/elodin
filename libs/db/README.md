# `elodin-db`

### Install

Install `elodin-db` using the standalone installer script:

```sh
curl -LsSf https://storage.googleapis.com/elodin-releases/install-db.sh | sh
```

Alternatively, you can download the latest portable binary for your platform:

- [macOS (arm64)](https://storage.googleapis.com/elodin-releases/latest/elodin-db-aarch64-apple-darwin.tar.gz)
- [Linux (x86_64)](https://storage.googleapis.com/elodin-releases/latest/elodin-db-x86_64-unknown-linux-musl.tar.gz)
- [Linux (arm64)](https://storage.googleapis.com/elodin-releases/latest/elodin-db-aarch64-unknown-linux-musl.tar.gz)

### Run the database

```sh
# Run elodin-db in the foreground, listening on port 2240 and storing data in the default user data directory
elodin-db run [::]:2240 $HOME/.local/share/elodin/db
```

### Stream data to the database

See [./examples/c](./examples/c) for an example C client that streams fake sensor data to the database. Build and run the client:

```sh
cc examples/c/main.c -lm -o examples/c/db-client
./examples/c/db-client
```

To populate the database with the necessary metadata, the C client first needs to send an initial message that describes the components it will be streaming as well as the on-wire format of the data (vtable). This message can be generated ahead of time and embedded into a C header file by running the following command:

```sh
cargo run --example vtable-gen > examples/c/vtable.h
```

### Connect to the database using the CLI

Launch a LUA REPL to interact with the database:
```sh
elodin-db lua
```

Connect to the database and dump all of the metadata:
```
db ❯❯ client = connect("127.0.0.1:2240")
db ❯❯ client:dump_metadata()
```

Run `:help` in the REPL to see all available commands:
```
db ❯❯ :help
Impeller Lua REPL
- `connect(addr)`
   Connects to a new database and returns a client
- `Client:dump_metadata()`
   Dumps all metadata from the db
...
```

### Connect to the database using the Elodin Editor

Install the [Elodin Editor](https://docs.elodin.systems/get-started/quickstart/#install) if you haven't already. Then, launch the editor by providing the database IP and port:

```sh
elodin editor 127.0.0.1:2240
```

The example C client just streams a sine wave component to entity "1". You can view this in the editor by creating a graph for entity "1" and selecting the only component available for that entity.
