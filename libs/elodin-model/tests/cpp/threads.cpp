// Separate instances of one program, stepped from several threads at once, must each end where
// they end when stepped alone. Works on any packaged model: thread i takes STEPS + i steps.
//
//     threads [WORKERS]
#include "elodin_model.h"

#include <pthread.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr std::size_t THREADS = 8;
constexpr std::size_t STEPS = 2000;
// Enough for any model the examples export; a default thread stack is not.
constexpr std::size_t STACK = std::size_t { 64 } << 20;

struct Run {
    const ElodinModelProgram* program;
    std::size_t steps;
    // Every component's bytes after the last step.
    std::vector<unsigned char> state;
    // Why the run failed; errors are per thread.
    std::string error;
};

void* step_instance(void* argument)
{
    Run& run = *static_cast<Run*>(argument);
    ElodinModelInstance* instance = elodin_model_create(run.program);
    bool ok = instance != nullptr;
    for (std::size_t i = 0; ok && i < run.steps; ++i)
        ok = elodin_model_step(instance) == ELODIN_MODEL_OK;
    const std::size_t components = elodin_model_component_count(run.program);
    for (std::size_t c = 0; ok && c < components; ++c) {
        const std::size_t size = elodin_model_component_size(run.program, c);
        const std::size_t at = run.state.size();
        run.state.resize(at + size);
        ok = elodin_model_read(instance, c, run.state.data() + at, size) == ELODIN_MODEL_OK;
    }
    if (!ok)
        run.error = elodin_model_error();
    elodin_model_destroy(instance);
    return nullptr;
}

void start(pthread_t& thread, Run& run)
{
    pthread_attr_t attributes;
    pthread_attr_init(&attributes);
    pthread_attr_setstacksize(&attributes, STACK);
    if (pthread_create(&thread, &attributes, step_instance, &run) != 0) {
        std::fputs("cannot start a thread\n", stderr);
        std::exit(1);
    }
    pthread_attr_destroy(&attributes);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc > 1 && elodin_model_set_workers(std::strtoul(argv[1], nullptr, 10)) != ELODIN_MODEL_OK) {
        std::fprintf(stderr, "set workers: %s\n", elodin_model_error());
        return 1;
    }
    ElodinModelProgram* program = elodin_model_open();
    if (!program) {
        std::fprintf(stderr, "open: %s\n", elodin_model_error());
        return 1;
    }
    const std::size_t workers = elodin_model_effective_workers(program);
    std::vector<Run> alone(THREADS), together(THREADS);
    std::vector<pthread_t> threads(THREADS);
    for (std::size_t i = 0; i < THREADS; ++i) {
        alone[i] = Run { program, STEPS + i, {}, {} };
        together[i] = alone[i];
        start(threads[i], alone[i]);
        pthread_join(threads[i], nullptr);
    }
    for (std::size_t i = 0; i < THREADS; ++i)
        start(threads[i], together[i]);
    for (pthread_t thread : threads)
        pthread_join(thread, nullptr);
    elodin_model_close(program);

    int status = 0;
    for (std::size_t i = 0; i < THREADS; ++i) {
        const std::string& error = alone[i].error.empty() ? together[i].error : alone[i].error;
        if (!error.empty()) {
            std::fprintf(stderr, "run %zu failed: %s\n", i, error.c_str());
            status = 1;
        } else if (alone[i].state != together[i].state) {
            std::fprintf(stderr, "run %zu differs when other threads step other instances\n", i);
            status = 1;
        }
    }
    if (alone.front().state == alone.back().state) {
        std::fputs("the state does not change with the step count, so this proves nothing\n", stderr);
        status = 1;
    }
    if (status == 0)
        std::printf("%zu threads stepping separate instances match the same runs alone, %zu workers\n",
            THREADS, workers);
    return status;
}
