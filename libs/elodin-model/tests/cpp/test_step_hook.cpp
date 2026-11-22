// A hook library for hook.cpp, written against elodin_step_hook.h alone: each call doubles the
// `value` column and adds the tick.
#include "elodin_step_hook.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

extern "C" {
ElodinStepInitFn double_step_init;
ElodinStepFn double_step;
ElodinStepDestroyFn double_step_destroy;
}

namespace {

int fail(const ElodinStep* step, const char* message)
{
    std::snprintf(step->error, step->error_capacity, "%s", message);
    return 1;
}

} // namespace

// Stores a call counter, and insists on the tick and timestamp hook.cpp initializes with.
int32_t double_step_init(const ElodinStep* step, void** state)
{
    if (step->abi_version != ELODIN_STEP_HOOK_ABI_VERSION)
        return fail(step, "unsupported hook ABI");
    if (step->tick != 5 || step->timestamp != 5000)
        return fail(step, "init did not get the first call's tick and timestamp");
    *state = new std::uint64_t(0);
    return 0;
}

int32_t double_step(const ElodinStep* step)
{
    if (!step->state || step->row_start != 0 || step->row_end != step->rows)
        return fail(step, "expected one call over every row, after init");
    if (step->tick == 13)
        return fail(step, "tick 13 is refused");
    ++*static_cast<std::uint64_t*>(step->state);
    for (std::uint64_t i = 0; i < step->column_count; ++i) {
        const ElodinColumn& column = step->columns[i];
        if (std::strcmp(column.name, "value") != 0 || !column.writable || column.row_bytes != sizeof(double))
            continue;
        for (std::uint64_t row = 0; row < step->rows; ++row) {
            double value;
            std::memcpy(&value, column.data + row * sizeof value, sizeof value);
            value = value * 2.0 + static_cast<double>(step->tick);
            std::memcpy(column.data + row * sizeof value, &value, sizeof value);
        }
        return 0;
    }
    return fail(step, "no writable value column");
}

void double_step_destroy(void* state)
{
    delete static_cast<std::uint64_t*>(state);
}
