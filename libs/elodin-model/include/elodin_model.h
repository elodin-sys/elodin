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

/* Why the last failing call on this thread failed. Valid until the next failure. */
const char *elodin_model_error(void);

/* "aot" when the model's code was compiled before delivery, "jit" when compiled at open, "none"
 * when the library was built without a model. */
const char *elodin_model_backend(void);

/* Threads a precompiled step runs its parallel groups on, the caller included; call before the
 * first open. 0, the default, uses the count the model was exported for. Results do not depend
 * on it. */
void elodin_model_set_workers(size_t workers);

/* NULL on failure. */
ElodinModelProgram *elodin_model_open(void);
void elodin_model_close(ElodinModelProgram *program);

size_t elodin_model_component_count(const ElodinModelProgram *program);
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

/* A run at the model's initial state. NULL on failure. */
ElodinModelInstance *elodin_model_create(const ElodinModelProgram *program);
void elodin_model_destroy(ElodinModelInstance *instance);
/* Back to the initial state. */
void elodin_model_reset(ElodinModelInstance *instance);
/* Advances the run by one tick. 0 on success. */
int elodin_model_step(ElodinModelInstance *instance);

/* 0 on success; nonzero, with nothing copied, if `component` or `size` does not match. */
int elodin_model_read(const ElodinModelInstance *instance, size_t component, void *dst, size_t size);
int elodin_model_write(ElodinModelInstance *instance, size_t component, const void *src, size_t size);

#ifdef __cplusplus
}
#endif

#endif
