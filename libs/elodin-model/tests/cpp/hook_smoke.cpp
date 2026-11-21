#include "elodin_model.h"
#include "elodin_step_hook.h"

#include <cmath>
#include <cstdint>
#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "usage: hook_smoke LIBRARY\n";
        return 2;
    }
    ElodinHook* hook = elodin_hook_open(argv[1], "double_step");
    if (!hook) {
        std::cerr << elodin_model_error() << "\n";
        return 1;
    }
    double value = 3.0;
    ElodinColumn column {};
    column.name = "value";
    column.data = reinterpret_cast<std::uint8_t*>(&value);
    column.row_bytes = sizeof value;
    column.writable = 1;
    if (elodin_hook_initialize(hook, 1, &column, 1)
        || elodin_hook_call(hook, 5, 1234, &column, 1)) {
        std::cerr << elodin_model_error() << "\n";
        elodin_hook_close(hook);
        return 1;
    }
    elodin_hook_close(hook);
    if (std::abs(value - 11.0) > 1.0e-12) {
        std::cerr << "hook produced " << value << ", not 11\n";
        return 1;
    }
    std::cout << "caller-driven hook smoke passed\n";
    return 0;
}
