# Elodin-DB C/C++ Client Examples

Examples showing how to send telemetry data to [elodin-db](https://github.com/elodin-sys/elodin) over TCP.

## Which example should I use?

| Example | Pattern | When to use |
|---|---|---|
| **client-batched.cpp** | 1 connection, 1 VTable with N fields, 1 packet/tick | **Recommended.** Best throughput for multi-component workloads. |
| **client-per-component.cpp** | N connections, 1 VTable each, N packets/tick | Simpler code, fine for a few low-frequency components. |
| **rocket-client.cpp** | 1 connection, 1 component | Single-component use case. |
| **client.c** | Raw C, no db.hpp | Minimal C integration without C++ dependencies. |
| **log-client.cpp** | Structured log messages | Sending flight software logs to elodin-db. |
| **rust_client/** | Rust + TUI | Interactive Rust client with discovery and control. |

### Performance

Each packet sent to elodin-db pays a fixed overhead (~6.6µs): protocol parsing, write-lock acquisition, vtable lookup, mmap write. With the **batched** pattern, N components share 1 packet per tick. With **per-component**, each sends its own packet.

| | Batched | Per-component |
|---|---|---|
| 6 components @ 1kHz | 1,000 pkt/s | 6,000 pkt/s |
| 400 components @ 250Hz | 1,000 pkt/s | 100,000 pkt/s |

Run the built-in benchmark for detailed numbers:

```bash
elodin-db-bench --scenario customer --mode batch
elodin-db-bench --scenario customer --mode per-component
```

## Running

All C++ examples use a self-compiling shebang — run them directly:

```bash
# Start elodin-db first
elodin-db run [::]:2240 my-db

# Then run any example
./client-batched.cpp
./client-per-component.cpp
./rocket-client.cpp
```

Or build with make:

```bash
make all
```

## db.hpp

Header-only C++ library implementing the elodin-db wire protocol: postcard serialization, VTable types and builder pattern, packet encoding, and component metadata helpers. Include it and use the `vtable::builder` namespace to construct VTables.
