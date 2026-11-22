// How much stack this model's steps use, measured, for sizing the thread that steps it. Works
// on any packaged model, from its initial state.
//
//     stack_probe [STEPS]
#include "elodin_model.h"

#include <pthread.h>
#include <sys/mman.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

// What the stack holds until a frame writes over it.
constexpr unsigned char UNUSED = 0xA5;

struct Probe {
    const ElodinModelProgram* program;
    std::size_t steps;
    bool ok;
};

void* step_instance(void* argument)
{
    Probe& probe = *static_cast<Probe*>(argument);
    ElodinModelInstance* instance = elodin_model_create(probe.program);
    probe.ok = instance != nullptr;
    for (std::size_t i = 0; probe.ok && i < probe.steps; ++i)
        probe.ok = elodin_model_step(instance) == ELODIN_MODEL_OK;
    elodin_model_destroy(instance);
    return nullptr;
}

} // namespace

int main(int argc, char** argv)
{
    ElodinModelProgram* program = elodin_model_open();
    if (!program) {
        std::fprintf(stderr, "open: %s\n", elodin_model_error());
        return 1;
    }
    Probe probe { program, argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 100, false };

    const std::size_t size = elodin_model_recommended_stack();
    void* const mapped = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (mapped == MAP_FAILED) {
        std::perror("map a stack");
        return 1;
    }
    std::memset(mapped, UNUSED, size);
    pthread_attr_t attributes;
    pthread_attr_init(&attributes);
    pthread_attr_setstack(&attributes, mapped, size);
    pthread_t thread;
    if (pthread_create(&thread, &attributes, step_instance, &probe) != 0) {
        std::fputs("cannot start a thread\n", stderr);
        return 1;
    }
    pthread_join(thread, nullptr);
    pthread_attr_destroy(&attributes);
    elodin_model_close(program);
    if (!probe.ok) {
        std::fputs("a step failed\n", stderr);
        return 1;
    }

    // A stack grows down from its top, so the first byte written over is the deepest one used.
    const unsigned char* const stack = static_cast<const unsigned char*>(mapped);
    std::size_t untouched = 0;
    while (untouched < size && stack[untouched] == UNUSED)
        ++untouched;
    std::printf("%zu steps reached %zu KiB of stack; a thread that steps this model needs that plus"
                " the caller's own frames\n",
        probe.steps, (size - untouched + 1023) / 1024);
    munmap(mapped, size);
    return 0;
}
