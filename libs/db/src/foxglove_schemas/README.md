# Foxglove JSON schemas (vendored)

Verbatim copies of the official Foxglove schema JSON files, embedded into MCAP
channels by `export_mcap.rs`.

- Source: <https://github.com/foxglove/foxglove-sdk> (`schemas/jsonschema/`)
- License: MIT (Copyright Foxglove Technologies Inc)

Do not hand-edit: Foxglove's JSON-channel deserializer derives its base64 →
bytes decoding from these schemas, so nested `contentEncoding: "base64"`
declarations (e.g. `SceneUpdate.entities[].models[].data`) must match the
official definitions exactly or embedded payloads arrive as plain strings.
