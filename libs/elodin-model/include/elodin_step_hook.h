/*
 * Optional caller-driven adapter for Elodin native pre-commit hook libraries.
 *
 * This API does not step a model, impose hook order, or own time. The caller supplies columns,
 * tick and timestamp whenever its orchestration contract says the hook should run.
 */
#ifndef ELODIN_STEP_HOOK_H
#define ELODIN_STEP_HOOK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ELODIN_STEP_HOOK_ABI_VERSION 1u

typedef struct {
    const char *name;
    uint8_t *data;
    uint64_t row_bytes;
    uint32_t writable;
} ElodinColumn;

typedef struct {
    uint32_t abi_version;
    uint64_t tick;
    int64_t timestamp;
    uint64_t rows;
    uint64_t row_start;
    uint64_t row_end;
    const ElodinColumn *columns;
    uint64_t column_count;
    void *state;
    char *error;
    uint64_t error_capacity;
} ElodinStep;

typedef struct ElodinHook ElodinHook;

/* Load `symbol`, and optional `{symbol}_init` and `{symbol}_destroy`. */
ElodinHook *elodin_hook_open(const char *library, const char *symbol);

/* Initialize once over every row. Columns and their data need only remain live for the call. */
int elodin_hook_initialize(ElodinHook *hook,
                           uint64_t rows,
                           const ElodinColumn *columns,
                           uint64_t column_count);

/* Invoke once over every row with caller-owned logical time. */
int elodin_hook_call(ElodinHook *hook,
                     uint64_t tick,
                     int64_t timestamp,
                     const ElodinColumn *columns,
                     uint64_t column_count);

/* Calls the hook's optional destroy symbol, then unloads the library. NULL is allowed. */
void elodin_hook_close(ElodinHook *hook);

/* Failures use elodin_model_error() from elodin_model.h. */

#ifdef __cplusplus
}
#endif

#endif
