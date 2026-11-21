/*
 * A simulation exported from Elodin with `JaxSim.export_model`, stepped from C or C++.
 *
 * A program holds the model's code and initial state; open it once. An instance holds the
 * state of one run, and each `elodin_model_step` advances it by one simulation tick. Instances
 * of one program are independent, and a program must outlive its instances.
 *
 * Components are addressed by index, in the order of the model's schema.json. Each holds its
 * entities' values as one little-endian array; read and write it whole.
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

#define ELODIN_MODEL_ABI_VERSION 2u

#define ELODIN_MODEL_OK 0
#define ELODIN_MODEL_ERR_ARGUMENT 1
#define ELODIN_MODEL_ERR_STATE 2
#define ELODIN_MODEL_ERR_STEP 3

#define ELODIN_MODEL_COMPONENT_WRITTEN_BY_STEP (1u << 0)
#define ELODIN_MODEL_COMPONENT_EXTERNAL_CONTROL (1u << 1)
#define ELODIN_MODEL_COMPONENT_TRANSIENT (1u << 2)
#define ELODIN_MODEL_COMPONENT_PRIVATE (1u << 3)
#define ELODIN_MODEL_COMPONENT_TICK (1u << 4)
#define ELODIN_MODEL_COMPONENT_TIME_STEP (1u << 5)

typedef struct {
    size_t struct_size; /* Set to sizeof(ElodinModelComponentType). */
    const char *dtype;
    const size_t *shape;
    size_t rank;
    const size_t *element_shape;
    size_t element_rank;
    size_t element_bytes;
    size_t rows;
    size_t row_bytes;
    size_t total_bytes;
    uint32_t flags;
    uint32_t entity_axis; /* 0: none, 1: leading */
} ElodinModelComponentType;

typedef struct {
    size_t struct_size; /* Set to sizeof(ElodinModelInterfaceField). */
    const char *name;
    size_t component;
    uint32_t writable;
    size_t rows;
    size_t row_bytes;
} ElodinModelInterfaceField;

/* Why the last failing call on this thread failed. Valid until the next failure. */
const char *elodin_model_error(void);
uint32_t elodin_model_abi_version(void);

/* "aot" when the model's code was compiled before delivery, "jit" when compiled at open, "none"
 * when the library was built without a model. */
const char *elodin_model_backend(void);
/* Embedded model-export manifest as compact JSON. */
const char *elodin_model_build_info(void);

/* Threads a precompiled step runs its parallel groups on, the caller included; call before the
 * first open. 0, the default, uses the count the model was exported for. Results do not depend
 * on it. */
void elodin_model_set_workers(size_t workers);
/* Checked form; fails if a model is open or an existing process pool would have to shrink. */
int elodin_model_configure_workers(size_t workers);

/* NULL on failure. */
ElodinModelProgram *elodin_model_open(void);
void elodin_model_close(ElodinModelProgram *program);

size_t elodin_model_component_count(const ElodinModelProgram *program);
uint32_t elodin_model_schema_abi(const ElodinModelProgram *program);
const char *elodin_model_format(const ElodinModelProgram *program);
const char *elodin_model_endianness(const ElodinModelProgram *program);
size_t elodin_model_effective_workers(const ElodinModelProgram *program);
uint32_t elodin_model_tick_advanced_by_step(const ElodinModelProgram *program);
size_t elodin_model_recommended_stack(void);
int elodin_model_nominal_time_step(const ElodinModelProgram *program, double *seconds);
/* NULL when `component` is out of range. */
const char *elodin_model_component_name(const ElodinModelProgram *program, size_t component);
/* Size in bytes; 0 when `component` is out of range. */
size_t elodin_model_component_size(const ElodinModelProgram *program, size_t component);
/* Rows of the component: one per entity that has it; 0 when `component` is out of range. */
size_t elodin_model_component_rows(const ElodinModelProgram *program, size_t component);
/* The entity of a row: its name, or NULL, and its id, or UINT64_MAX, when out of range or when
 * the model was exported without entity ids. */
const char *elodin_model_entity_name(const ElodinModelProgram *program, size_t component, size_t row);
uint64_t elodin_model_entity_id(const ElodinModelProgram *program, size_t component, size_t row);
/* -1 when the model has no component `name`. */
int64_t elodin_model_component_index(const ElodinModelProgram *program, const char *name);
int elodin_model_component_type(const ElodinModelProgram *program,
                                size_t component,
                                ElodinModelComponentType *output);

/* Optional named row interfaces exported by the model author. They describe data only; the
 * caller decides when to read, write, step, invoke hooks, or perform I/O. */
size_t elodin_model_interface_count(const ElodinModelProgram *program);
const char *elodin_model_interface_name(const ElodinModelProgram *program, size_t interface_index);
int64_t elodin_model_interface_index(const ElodinModelProgram *program, const char *name);
size_t elodin_model_interface_rows(const ElodinModelProgram *program, size_t interface_index);
size_t elodin_model_interface_field_count(const ElodinModelProgram *program, size_t interface_index);
int elodin_model_interface_field(const ElodinModelProgram *program,
                                 size_t interface_index,
                                 size_t field,
                                 ElodinModelInterfaceField *output);

/* A run at the model's initial state. NULL on failure. */
ElodinModelInstance *elodin_model_create(const ElodinModelProgram *program);
void elodin_model_destroy(ElodinModelInstance *instance);
/* Back to the initial state. */
void elodin_model_reset(ElodinModelInstance *instance);
int elodin_model_reset_checked(ElodinModelInstance *instance);
/* Advances the run by one tick. 0 on success. */
int elodin_model_step(ElodinModelInstance *instance);
uint64_t elodin_model_generation(const ElodinModelInstance *instance);
int elodin_model_current_tick(const ElodinModelInstance *instance, uint64_t *tick);
/* Convenience write to the model's simulation_time_step component. It does not step or pace. */
int elodin_model_set_time_step(ElodinModelInstance *instance, double seconds);

/* 0 on success; nonzero, with nothing copied, if `component` or `size` does not match. */
int elodin_model_read(const ElodinModelInstance *instance, size_t component, void *dst, size_t size);
int elodin_model_write(ElodinModelInstance *instance, size_t component, const void *src, size_t size);
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
