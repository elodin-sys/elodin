# Caller-driven witness C API design

This is an optional library beside `libelodin_model`, not part of model stepping. A customer
orchestrator owns tick order, stage names, external I/O, and which bytes constitute evidence.
The library only writes and validates Elodin witness format version 1.

## Ownership

The caller:

- supplies the exact schema, labels, tick, timestamp, stage order, and record bytes;
- decides when the simulation, controller, hooks, recorder, and wall clock run;
- marks external records, which are retained for replay and excluded from the digest;
- reports run failures in the trailer.

The library:

- validates schema consistency and compact-JSON ordering;
- computes BLAKE3 record, stage, tick, and root digests;
- writes little-endian frames and the trailer;
- optionally reads a witness and returns exact record bytes for caller-owned replay.

It must not call `elodin_model_step`, invoke hooks, start threads, pace time, or interpret FSI,
DIS, flight-software, or `pre_commit` semantics.

## Proposed header

```c
typedef struct ElodinWitnessWriter ElodinWitnessWriter;
typedef struct ElodinWitnessReader ElodinWitnessReader;

ElodinWitnessWriter *elodin_witness_open(
    const char *path, const char *schema_json, const char *labels_json, int keep_raw);
int elodin_witness_begin_tick(ElodinWitnessWriter *, uint64_t tick, int64_t timestamp);
int elodin_witness_record(
    ElodinWitnessWriter *, const char *stage, const char *record, const void *bytes, size_t size);
int elodin_witness_absent(ElodinWitnessWriter *, const char *stage, const char *record);
int elodin_witness_end_tick(ElodinWitnessWriter *);
int elodin_witness_finish(
    ElodinWitnessWriter *, const char *failure, char root_hex[65]);
void elodin_witness_close(ElodinWitnessWriter *);

ElodinWitnessReader *elodin_witness_reader_open(const char *path);
const char *elodin_witness_header_json(const ElodinWitnessReader *);
const char *elodin_witness_trailer_json(const ElodinWitnessReader *);
int elodin_witness_next_tick(ElodinWitnessReader *);
int elodin_witness_record_bytes(
    const ElodinWitnessReader *, const char *stage, const char *record,
    const void **bytes, size_t *size);
void elodin_witness_reader_close(ElodinWitnessReader *);
```

Every status uses a witness-specific thread-local error string and a version/capability query.
Pointers returned by the reader remain valid until the next reader call.

## Implementation boundary

Move the format-only types and digest functions from `nox-py/src/witness/format.rs` into a small
`elodin-witness` crate. `nox-py` and the C wrapper both depend on it. Keep schema serialization
and key order byte-identical, and prove existing witness fixtures retain their roots before
shipping the C API.

`elodin determinism inspect --json` is the first reader surface. It lets an outside integrator
obtain the validated schema and labels without linking this future library.
