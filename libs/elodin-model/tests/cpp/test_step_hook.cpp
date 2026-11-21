#include "elodin_step_hook.h"

#include <cstdint>
#include <cstring>

extern "C" int double_step_init(const ElodinStep* step, void** state)
{
    if (!step || !state || step->abi_version != ELODIN_STEP_HOOK_ABI_VERSION)
        return 1;
    *state = new std::uint64_t(0);
    return 0;
}

extern "C" int double_step(const ElodinStep* step)
{
    if (!step || !step->state || step->rows != 1 || step->row_start != 0 || step->row_end != 1)
        return 1;
    auto* calls = static_cast<std::uint64_t*>(step->state);
    ++*calls;
    for (std::uint64_t i = 0; i < step->column_count; ++i) {
        const ElodinColumn& column = step->columns[i];
        if (std::strcmp(column.name, "value") == 0 && column.writable && column.row_bytes == 8) {
            double value;
            std::memcpy(&value, column.data, sizeof value);
            value = value * 2.0 + static_cast<double>(step->tick);
            std::memcpy(column.data, &value, sizeof value);
            return 0;
        }
    }
    return 1;
}

extern "C" void double_step_destroy(void* state)
{
    delete static_cast<std::uint64_t*>(state);
}
