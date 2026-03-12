# Log Streaming Example

Demonstrates text log ingestion into Elodin DB and live display in the Editor's log viewer panel.

## Quick Start

```bash
elodin editor examples/logstream/main.py
```

This will:
1. Start the simulation (rolling ball with wind)
2. Compile and launch `libs/db/examples/log-client.cpp` via S10
3. Display the incoming log messages in the "Flight Software Log" panel

## How It Works

The C++ log client connects to the database and sends `LogEntry` messages
(a postcard-encoded struct with a level byte and a message string) as MSG
packets. The editor subscribes to the message stream and renders them in a
scrollable, level-filterable log viewer panel.

The log client simulates a flight software boot sequence followed by
repeating flight telemetry cycles with INFO, DEBUG, WARN, and ERROR messages.

## Log Panel Controls

- **Level filters** — click TRACE / DEBUG / INFO / WARN / ERROR to show or
  hide entries at or above that level
- **Auto-scroll** — toggle to pin the view to the latest entry
