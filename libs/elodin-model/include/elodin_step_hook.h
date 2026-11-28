/*
 * Elodin's native step hook interface, version 1, and a loader for it that the caller drives.
 *
 * A hook is a C function in a shared library that reads and writes rows of named columns, as
 * the `pre_commit` hooks of an Elodin simulation do. A hook library includes this header for
 * the structs and symbol types; a program that runs one includes it for the loader, which is
 * part of libelodin_model.
 *
 * The loader does not step a model, order hooks or own time. The caller passes the columns,
 * tick and timestamp whenever its own schedule says the hook runs.
 */
#ifndef ELODIN_STEP_HOOK_H
#define ELODIN_STEP_HOOK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ELODIN_STEP_HOOK_ABI_VERSION 1u

/* One column: `ElodinStep.rows` rows of `row_bytes` bytes, back to back. */
typedef struct {
    const char *name;
    uint8_t *data;
    uint64_t row_bytes;
    uint32_t writable; /* nonzero when the hook may write the column */
} ElodinColumn;

/* What one call of a hook sees. */
typedef struct {
    uint32_t abi_version;    /* ELODIN_STEP_HOOK_ABI_VERSION */
    uint64_t tick;
    int64_t timestamp;       /* the caller's time for the tick; Elodin passes microseconds */
    uint64_t rows;           /* rows of every column */
    uint64_t row_start;      /* the call owns rows row_start up to row_end, and writes no others */
    uint64_t row_end;
    const ElodinColumn *columns;
    uint64_t column_count;
    void *state;             /* what the init symbol stored, or NULL */
    char *error;             /* where a failing call writes a NUL-terminated message */
    uint64_t error_capacity; /* bytes at `error` */
} ElodinStep;

/* A hook library defines `symbol`, and may define `{symbol}_init`, which runs once before the
 * first call and can store state, and `{symbol}_destroy`, which frees that state. The first two
 * return 0, or nonzero after writing the reason to `step->error`. */
typedef int32_t ElodinStepFn(const ElodinStep *step);
typedef int32_t ElodinStepInitFn(const ElodinStep *step, void **state);
typedef void ElodinStepDestroyFn(void *state);

/* A loaded hook. Use it from one thread at a time. */
typedef struct ElodinHook ElodinHook;

/* Failures below return NULL or a nonzero ELODIN_MODEL_ERR_* status, and `elodin_model_error`
 * of elodin_model.h says why. */

/* Loads `symbol` from the shared library at `library`, with its init and destroy symbols when
 * the library has them. */
ElodinHook *elodin_hook_open(const char *library, const char *symbol);

/* Runs the init symbol once, over `rows` rows, with the tick and timestamp of the first call, as
 * Elodin does. Call it even for a hook without one. The columns need only last for the call. */
int elodin_hook_initialize(ElodinHook *hook,
                           uint64_t rows,
                           uint64_t tick,
                           int64_t timestamp,
                           const ElodinColumn *columns,
                           uint64_t column_count);

/* Runs the hook once over every row. Elodin can instead share the rows between several calls. */
int elodin_hook_call(ElodinHook *hook,
                     uint64_t tick,
                     int64_t timestamp,
                     const ElodinColumn *columns,
                     uint64_t column_count);

/* Runs the destroy symbol of an initialized hook, then unloads the library. NULL is allowed. */
void elodin_hook_close(ElodinHook *hook);

#ifdef __cplusplus
}
#endif

#endif
