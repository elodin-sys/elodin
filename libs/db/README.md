# `elodin-db`

### Install


Install the elodin-db from the [releases](https://github.com/elodin-sys/elodin/releases) page.


### Run the database

```sh
# Run elodin-db in the foreground:
# - Listening on port 2240
# - Storing data in the default user data directory ($HOME/.local/share/elodin/db)
# - Using the ./examples/db-config.lua config
# - Setting log level to warn (optional)
elodin-db run [::]:2240 $HOME/.local/share/elodin/db --config examples/db-config.lua --log-level warn
```

### Stream data to the database with C

See [./examples/client.c](./examples/client.c) for an example C client that streams fake sensor data to the database. Build and run the client:

```sh
cc examples/client.c -lm -o /tmp/client; /tmp/client
```


### Subscribe to data with C++

[./examples/client.cpp](./examples/client.cpp) includes an example of how to subscribe to data using C++. It can be built and run using:

This example uses C++23, but the library itself is C++20 compatible.

``` sh
c++ -std=c++23 examples/client.cpp -o /tmp/client-cpp; /tmp/client-cpp
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

Install the [Elodin Editor](https://docs.elodin.systems/hello/quickstart/#install) if you haven't already. Then, launch the editor by providing the database IP and port:

```sh
elodin editor 127.0.0.1:2240
```

The example C client just streams a sine wave component to entity "1". You can view this in the editor by creating a graph for entity "1" and selecting the only component available for that entity.

### Follow mode -- replicate data from another database

Follow mode starts a database that replicates all data (components, messages,
metadata) from another running elodin-db instance over a single TCP
connection.  The follower database still accepts its own local connections and
data writers.

Start the source database:

```sh
elodin-db run [::]:2240 $HOME/.local/share/elodin/source-db
```

Start the follower database, pointing it at the source:

```sh
elodin-db run [::]:2241 $HOME/.local/share/elodin/follower-db --follows 127.0.0.1:2240
```

The follower will:
1. Synchronize all existing metadata and schemas from the source.
2. Backfill all historical component time-series data and message logs.
3. Stream real-time updates as they arrive on the source.

Connect the Elodin Editor to the follower to view the replicated data:

```sh
elodin editor 127.0.0.1:2241
```

#### Configurable packet size

By default, the source batches outgoing data into ~1500-byte TCP writes
(standard Ethernet MTU).  This dramatically reduces network overhead when the
source has many components.  You can tune the target packet size:

```sh
elodin-db run [::]:2241 ./follower-db --follows 127.0.0.1:2240 --follow-packet-size 9000
```

#### Dual-source example (video stream)

Run the video-stream example on the source, follow it on the target, and also
connect a local video stream directly to the follower:

```sh
# Source machine
elodin editor examples/video-stream/main.py

# Target machine -- follow the source
elodin-db run [::]:2241 ./follower-db --follows SOURCE_IP:2240 --follow-packet-size 1500

# Target machine -- connect the editor
elodin editor 127.0.0.1:2241

# Target machine -- add a second, local video stream
examples/video-stream/stream-video.sh  # (pointed at 127.0.0.1:2241)
```

Both the replicated video from the source and the locally-streamed video will
be visible in the editor connected to the follower.

If two sources write to the same component, the follower logs a warning to
alert you to potential data corruption.

#### Legacy lua-based downlink

The `examples/downlink.lua` script is still available for custom replication
workflows, but `--follows` is the recommended approach for most use cases.

### Trim a database

Remove data from the beginning or end of a recording. Values are in
microseconds. At least one of `--before` or `--after` must be provided.
Without `--output`, the database is modified in place.

```sh
# Remove the first 3 minutes from a recording
elodin-db trim --from-start 180000000 ./my-db

# Remove the last 2 minutes from a recording
elodin-db trim --from-end 120000000 --output ./trimmed ./my-db

# Trim 1 minute from the start and 2 minutes from the end
elodin-db trim --from-start 60000000 --from-end 120000000 --output ./window ./my-db
```

| Parameter | Example | Purpose |
|-----------|---------|---------|
| `--from-start` | `180000000` | Remove the first N microseconds from the start |
| `--from-end` | `120000000` | Remove the last N microseconds from the end |
| `--output` | `./trimmed` | Write to a new path instead of modifying in place |
| `--dry-run` | | Show what would be trimmed without making changes |
| `-y` | | Skip the confirmation prompt |

### Generate C++ Header

elodin-db ships with a single header C++20 library. The library includes message definitions for communicating with the DB.

> NOTE: Not all definitions have been added yet if you need something ASAP please contact us

You can generate the C++ library by running:

`cargo run gen-cpp > ./examples/db.hpp`

This will generate a C++ header file at `./examples/db.hpp`
