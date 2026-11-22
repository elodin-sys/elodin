// The hook loader of elodin_step_hook.h, driven by the caller: it runs the hook only when asked,
// with the columns, tick and timestamp it is given.
//
//     hook LIBRARY
#include "elodin_model.h"
#include "elodin_step_hook.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

int failures = 0;

void expect(bool condition, const char* what)
{
    if (!condition) {
        std::fprintf(stderr, "expected %s (%s)\n", what, elodin_model_error());
        ++failures;
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::fputs("usage: hook LIBRARY\n", stderr);
        return 2;
    }
    expect(elodin_hook_open("no-such-library", "double_step") == nullptr, "a missing library to fail");
    expect(std::strstr(elodin_model_error(), "no-such-library") != nullptr, "the library in the error");
    expect(elodin_hook_open(argv[1], "no_such_symbol") == nullptr, "a missing symbol to fail");

    ElodinHook* hook = elodin_hook_open(argv[1], "double_step");
    if (!hook) {
        std::fprintf(stderr, "open %s: %s\n", argv[1], elodin_model_error());
        return 1;
    }
    double values[2] = { 3.0, 4.0 };
    ElodinColumn column {};
    column.name = "value";
    column.data = reinterpret_cast<std::uint8_t*>(values);
    column.row_bytes = sizeof(double);
    column.writable = 1;

    expect(elodin_hook_call(hook, 5, 5000, &column, 1) == ELODIN_MODEL_ERR_STEP, "a call before init to fail");
    expect(elodin_hook_initialize(hook, 2, 4, 4000, &column, 1) == ELODIN_MODEL_ERR_STATE,
        "the hook's init to see the tick it is given");
    expect(std::strstr(elodin_model_error(), "first call's tick") != nullptr, "the hook's own message");
    expect(elodin_hook_initialize(hook, 2, 5, 5000, &column, 1) == ELODIN_MODEL_OK, "init to succeed");
    expect(elodin_hook_initialize(hook, 2, 5, 5000, &column, 1) == ELODIN_MODEL_ERR_STATE, "one init only");

    expect(elodin_hook_call(hook, 5, 5000, &column, 1) == ELODIN_MODEL_OK, "the call to succeed");
    expect(values[0] == 11.0 && values[1] == 13.0, "each value doubled plus the tick");
    expect(elodin_hook_call(hook, 13, 13000, &column, 1) == ELODIN_MODEL_ERR_STEP, "the hook's failure");
    expect(std::strcmp(elodin_model_error(), "tick 13 is refused") == 0, "the hook's message");
    expect(values[0] == 11.0, "a failed call to leave the column alone");

    elodin_hook_close(hook);
    elodin_hook_close(nullptr);
    if (failures == 0)
        std::puts("the hook loader runs a hook library on the caller's schedule");
    return failures != 0;
}
