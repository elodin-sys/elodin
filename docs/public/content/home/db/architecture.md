+++
title = "Architecture"
description = "Overview of Elodin DB's Architecture"
draft = false
weight = 105
sort_by = "weight"

[extra]
lead = ""
toc = true
top = false
order = 7
icon = ""
+++

Elodin DB is built around the same entity-component concepts that Elodin Sim and Impeller use. In essence entities represent logical "objects" in the system -- think sensors, actuators, spacecraft. Components represent individual pieces of telemetry - think a sensor reading, an attitude estimate, an actuator commanded position, etc.

## Data Model

The core primitive of Elodin DB is the tensor, or the n-dimensional array. A scalar is a 0 dimensional tensor, a vector is a 1 dimensional tensor, a matrix is a 2 dimensional tensor, and so on. Tensors in Elodin DB have a dimension, a primitive type (f64, f32, etc), and their associated data. Tensors are fixed size and dense (non-sparse).

{% alert(kind="info") %}
We are using tensor in the ML framework / programming sense, not the pure mathematical sense.
{% end %}

Elodin DB stores tensors in time-series columns associated with a particular entity-component pair. Conceptually each entity-component pair is a table with each row being a timestamp, and a tensor. Timestamps are stored as i64 microsecond offsets from Unix epoch.

| Time | Data |
|------|------|
| 1742223919e6  | [1.0, 2.0, 3.0] |
| 1742223919    | [4.0, 5.0, 6.0] |


## VTable

Elodin DB uses a dynamic data-extraction system for fast data-ingest. Instead of requiring that data is sent in a fixed format - Elodin DB allows users to generate a vtable that describes the data being ingested. Loosely this can be thought of as a collection of offsets combined with entity and component ids. We borrowed the concept of a vtable from FlatBuffers, which use them in a similar manner.

One of the powers of VTables is that it lets us send raw structs over the wire, without serializing them. For instance take this struct:

```rust
#[repr(C)]
struct Data {
    pub ts: u32,
    pub mag: [f32; 3],
    pub gyro: [f32; 3],
    pub accel: [f32; 3],
    pub mag_temp: f32,
    pub mag_sample: u32,
    pub baro: f32,
    pub baro_temp: f32,
}
```

This struct contains a number of fields each associated with a different component. We can send Elodin DB this struct, unchanged, by formulating a VTable like below.

```lua
local vt = VTableBuilder(1)
local entity_ids = { 1 }
vt:column(ComponentId("ts"), "u32", {}, entity_ids)
vt:column(ComponentId("mag"), "f32", { 3 }, entity_ids)
vt:column(ComponentId("gyro"), "f32", { 3 }, entity_ids)
vt:column(ComponentId("accel"), "f32", { 3 }, entity_ids)
vt:column(ComponentId("mag_temp"), "f32", {}, entity_ids)
vt:column(ComponentId("mag_sample"), "u32", {}, entity_ids)
vt:column(ComponentId("baro"), "f32", {}, entity_ids)
vt:column(ComponentId("baro_temp"), "f32", {}, entity_ids)
```

In this example we are using a Lua script with the VTableBuilder API. Each time we call `column` it increments the internal offset by the size of the tensor. So the final VTable looks like this

| Offset | Component ID | Type | Shape  | Entity IDs |
|--------|--------------|------|------------|------------|
| 0      | ts           | u32  | []         | [1]        |
| 4      | mag          | f32  | [3]        | [1]        |
| 16     | gyro         | f32  | [3]        | [1]        |
| 28     | accel        | f32  | [3]        | [1]        |
| 40     | mag_temp     | f32  | []         | [1]        |
| 44     | mag_sample   | u32  | []         | [1]        |
| 48     | baro         | f32  | []         | [1]        |
| 52     | baro_temp    | f32  | []         | [1]        |


## Messages

Up until now we've discussed how to store telemetry that best fits into a time series of fixed-size tensors. Sometimes our data doesn't fit cleanly into the concept of "telemetry" and fixed sized structures. What if we want to send a single message that contains a variable length sequence of commands? What if we want to send compressed data that is variable length? What if we want to signal a state transition using an enum?

For these cases, Elodin DB supports messages which are variable length, [postcard](https://docs.rs/postcard/latest/postcard/) encoded. This allows you to pack virtually any data structure that postcard can represent.

{% alert(kind="info") %}
Technically, you can store any variable length data in a message, but for all of Elodin DB's features to work postcard is the best choice.
{% end %}

Messages are stored in a table made up of three-columns: a timestamp, an "offset" column, and a "data" column. The offset column is encoded using the [Umbra string format](https://cedardb.com/blog/german_strings/) (also called German Strings). Instead of just storing a length and offset to the data - an Umbra string optionally stores the data inline with the length. This allows for fast path lookups up data less than 12 bytes. Our implementation of this table is designed to be compatible with the Arrow ["Variable-size Binary View Layout"](https://arrow.apache.org/docs/format/Columnar.html#variable-size-binary-view-layout).

## On Disk Layout

Elodin DB uses a collection of memory mapped files to store data. Each separate column in a table maps to a different memory mapped file in its own directory. The files are not meant to be human readable. The directory structure is documented here for information purposes only.

```
- <component_id> - encoded as a 64 bit integer
  - <entity_id> - encodeded as a 64 bit integer
    - index - the timestamps associated with the data
    - data - the list of tensors
- msgs
  - <msg_id>
    - timestamps - the timestamps associated with each message entry
    - offset - the UmbraBuf associated with each message
    - data_log - the actual message data
- db_state - Stores certain database settings
```
