/*
 * A simulation exported from Elodin with `JaxSim.export_model`, stepped from C or C++.
 *
 * The library is passive. It works only inside the calls below, never calls back, and has no
 * schedule, pacing or files of its own: the caller decides when to write, step and read.
 *
 * A program holds the model's code and initial state; open it once. An instance holds the state
 * of one run, and each `elodin_model_step` advances it by one simulation tick. Instances of one
 * program are independent, and a program must outlive its instances.
 *
 * Components are addressed by index, in the order of the model's schema.json. Each holds its
 * entities' values as one little-endian array, a row per entity. Reads and writes copy; no
 * pointer into an instance's state is ever handed out.
 *
 * Threads: use an instance from one thread at a time. Different instances may be stepped from
 * different threads at once, and a program may be described from any thread. With the "jit"
 * backend, steps of one program's instances take turns.
 *
 * Stack: a step keeps its temporaries on the calling thread's stack. The examples fit a default
 * stack; a large model needs more, up to `elodin_model_recommended_stack`.
 *
 * Errors: a failed call returns NULL, -1 or a nonzero status, and `elodin_model_error` says why.
 * Strings and shapes returned for a program stay valid until `elodin_model_close`.
 */
#ifndef ELODIN_MODEL_H
#define ELODIN_MODEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ElodinModelProgram ElodinModelProgram;
typedef struct ElodinModelInstance ElodinModelInstance;

/* This interface's version, and the schema.json `abi_version` the library opens. */
#define ELODIN_MODEL_ABI_VERSION 1u

/* Statuses. Zero is success; the nonzero ones only sort failures roughly. */
#define ELODIN_MODEL_OK 0
#define ELODIN_MODEL_ERR_ARGUMENT 1 /* the arguments, or what the model has, do not allow it */
#define ELODIN_MODEL_ERR_STATE 2    /* the library or hook is not in a state for the call */
#define ELODIN_MODEL_ERR_STEP 3     /* a step or hook failed, or the library hit an internal error */

/* Bits of `ElodinModelComponentType.flags`. */
#define ELODIN_MODEL_COMPONENT_WRITTEN_BY_STEP (1u << 0)  /* an output: each step computes it */
#define ELODIN_MODEL_COMPONENT_EXTERNAL_CONTROL (1u << 1) /* an input: only the caller writes it */
#define ELODIN_MODEL_COMPONENT_TRANSIENT (1u << 2)        /* Elodin does not record it */
#define ELODIN_MODEL_COMPONENT_PRIVATE (1u << 3)          /* scratch state Elodin does not export */
#define ELODIN_MODEL_COMPONENT_TICK (1u << 4)             /* the tick: each step adds one */
#define ELODIN_MODEL_COMPONENT_TIME_STEP (1u << 5)        /* seconds the next step integrates */

/* The layout of one component. */
typedef struct {
    size_t struct_size;          /* set to sizeof(ElodinModelComponentType) before the call */
    const char *dtype;           /* "float64", "float32", "int64" ... "uint8", or "bool" */
    const size_t *shape;         /* `rank` dimensions of the whole array */
    size_t rank;
    const size_t *element_shape; /* `element_rank` dimensions of one row */
    size_t element_rank;
    size_t element_bytes;        /* bytes of one `dtype` value */
    size_t rows;                 /* entities that have the component */
    size_t row_bytes;
    size_t total_bytes;          /* rows * row_bytes */
    uint32_t flags;              /* ELODIN_MODEL_COMPONENT_* */
    uint32_t entity_axis;        /* 1: `shape` starts with `rows`; 0: one row, `shape` is its own */
} ElodinModelComponentType;

/* One field of a named interface: chosen rows of one component, in the interface's row order. */
typedef struct {
    size_t struct_size; /* set to sizeof(ElodinModelInterfaceField) before the call */
    const char *name;
    size_t component;   /* the component the rows belong to */
    uint32_t writable;  /* nonzero when `elodin_model_interface_write` accepts the field */
    size_t rows;
    size_t row_bytes;
} ElodinModelInterfaceField;

/* Why this thread's last failed call failed; never NULL. The pointer and its text stay as they
 * are until this thread's next failed call: successful calls leave them alone. */
const char *elodin_model_error(void);
/* The ELODIN_MODEL_ABI_VERSION the library was built with. */
uint32_t elodin_model_abi_version(void);

/* "aot" when the model's code was compiled before delivery, "jit" when compiled at open, "none"
 * when the library was built without a model. */
const char *elodin_model_backend(void);
/* What package.py recorded when it built the library, as compact JSON for a run manifest: the
 * compile time, Elodin revision, toolchain and target, and the hashes of the model's files.
 * "{}" when the library was built without package.py. */
const char *elodin_model_build_info(void);

/* Threads a precompiled step runs its parallel groups on, the caller included. 0, the default,
 * uses the count the model was exported for; results do not depend on it. Fails once a program
 * is open, or to ask for fewer threads than already run: every program in the process shares
 * them, and they are never stopped. */
int elodin_model_set_workers(size_t workers);

/* Opens the model built into the library; the "jit" backend compiles it here. NULL on failure. */
ElodinModelProgram *elodin_model_open(void);
/* Closes a program whose instances are all destroyed. NULL is allowed. */
void elodin_model_close(ElodinModelProgram *program);

/* schema.json's `abi_version`; 0 for a NULL program. */
uint32_t elodin_model_schema_abi(const ElodinModelProgram *program);
/* "elodin.model". */
const char *elodin_model_format(const ElodinModelProgram *program);
/* "little": the byte order of every value. */
const char *elodin_model_endianness(const ElodinModelProgram *program);
/* Threads this program's steps run parallel groups on, the caller included; 1 with "jit". */
size_t elodin_model_effective_workers(const ElodinModelProgram *program);
/* 1 when the model has a tick component, which every step adds one to; otherwise 0. */
uint32_t elodin_model_tick_advanced_by_step(const ElodinModelProgram *program);
/* Bytes of stack that fit a step of any model: what Elodin gives its own simulation threads. */
size_t elodin_model_recommended_stack(void);
/* Seconds of simulated time one step covers from the initial state. */
int elodin_model_nominal_time_step(const ElodinModelProgram *program, double *seconds);

size_t elodin_model_component_count(const ElodinModelProgram *program);
/* NULL when `component` is out of range. */
const char *elodin_model_component_name(const ElodinModelProgram *program, size_t component);
/* Size in bytes; 0 when `component` is out of range. */
size_t elodin_model_component_size(const ElodinModelProgram *program, size_t component);
/* Rows of the component: one per entity that has it; 0 when `component` is out of range. */
size_t elodin_model_component_rows(const ElodinModelProgram *program, size_t component);
/* The entity of a row: its name, or NULL when out of range, and its id, or UINT64_MAX. */
const char *elodin_model_entity_name(const ElodinModelProgram *program, size_t component, size_t row);
uint64_t elodin_model_entity_id(const ElodinModelProgram *program, size_t component, size_t row);
/* -1 when the model has no component `name`. */
int64_t elodin_model_component_index(const ElodinModelProgram *program, const char *name);
/* Fills `output`, whose `struct_size` the caller set. */
int elodin_model_component_type(const ElodinModelProgram *program,
                                size_t component,
                                ElodinModelComponentType *output);

/* Named interfaces the model's author exported. Each has fields of the same number of rows, a
 * field being chosen entities' rows of one component. They describe data only: the caller still
 * decides when to read, write and step. */
size_t elodin_model_interface_count(const ElodinModelProgram *program);
/* NULL when `interface_index` is out of range. */
const char *elodin_model_interface_name(const ElodinModelProgram *program, size_t interface_index);
/* -1 when the model has no interface `name`. */
int64_t elodin_model_interface_index(const ElodinModelProgram *program, const char *name);
/* Rows of each of the interface's fields; 0 when `interface_index` is out of range. */
size_t elodin_model_interface_rows(const ElodinModelProgram *program, size_t interface_index);
/* 0 when `interface_index` is out of range. */
size_t elodin_model_interface_field_count(const ElodinModelProgram *program, size_t interface_index);
/* Fills `output`, whose `struct_size` the caller set. */
int elodin_model_interface_field(const ElodinModelProgram *program,
                                 size_t interface_index,
                                 size_t field,
                                 ElodinModelInterfaceField *output);

/* A run at the model's initial state. NULL on failure. */
ElodinModelInstance *elodin_model_create(const ElodinModelProgram *program);
/* NULL is allowed. */
void elodin_model_destroy(ElodinModelInstance *instance);
/* Back to the initial state, every component included. For the next case of a campaign, reset
 * and then write that case's inputs again. */
int elodin_model_reset(ElodinModelInstance *instance);
/* Advances the run by one tick. A failed step leaves the state as it was. */
int elodin_model_step(ElodinModelInstance *instance);
/* Counts changes to the state: each step, reset and successful write adds one. 0 for NULL. */
uint64_t elodin_model_generation(const ElodinModelInstance *instance);
/* The tick component: steps since the initial state, unless the caller wrote it. Fails when the
 * model has none. */
int elodin_model_current_tick(const ElodinModelInstance *instance, uint64_t *tick);
/* Writes the time-step component, in seconds, for the steps that follow; a reset restores the
 * model's own. Fails when the model has none: its systems then keep the step they were built
 * with. */
int elodin_model_set_time_step(ElodinModelInstance *instance, double seconds);

/* Copy a whole component out of or into the run. `size` must be the component's size, or the
 * call fails and copies nothing. A write is seen by the next step. */
int elodin_model_read(const ElodinModelInstance *instance, size_t component, void *dst, size_t size);
int elodin_model_write(ElodinModelInstance *instance, size_t component, const void *src, size_t size);
/* The same for `count` rows starting at row `first`; `size` must be those rows' bytes. */
int elodin_model_read_rows(const ElodinModelInstance *instance,
                           size_t component,
                           size_t first,
                           size_t count,
                           void *dst,
                           size_t size);
int elodin_model_write_rows(ElodinModelInstance *instance,
                            size_t component,
                            size_t first,
                            size_t count,
                            const void *src,
                            size_t size);
/* The same for one field of an interface: all its rows, in the interface's order. Writing a
 * field that is not writable fails. */
int elodin_model_interface_read(const ElodinModelInstance *instance,
                                size_t interface_index,
                                size_t field,
                                void *dst,
                                size_t size);
int elodin_model_interface_write(ElodinModelInstance *instance,
                                 size_t interface_index,
                                 size_t field,
                                 const void *src,
                                 size_t size);

#ifdef __cplusplus
}
#endif

#endif
