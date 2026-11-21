// Steps the ball model exported from examples/ball-cpp and writes its trajectory as CSV.
//
//     ball_cpp [out.csv] [ticks] [seed]
#include "elodin_model.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>

namespace {

using Program = std::unique_ptr<ElodinModelProgram, decltype(&elodin_model_close)>;
using Instance = std::unique_ptr<ElodinModelInstance, decltype(&elodin_model_destroy)>;

[[noreturn]] void fail(const char* what)
{
    std::fprintf(stderr, "%s: %s\n", what, elodin_model_error());
    std::exit(1);
}

size_t component(const ElodinModelProgram* program, const char* name)
{
    const int64_t index = elodin_model_component_index(program, name);
    if (index < 0) {
        std::fprintf(stderr, "the model has no %s component\n", name);
        std::exit(1);
    }
    return static_cast<size_t>(index);
}

} // namespace

int main(int argc, char** argv)
{
    const char* path = argc > 1 ? argv[1] : "ball_cpp.csv";
    const long ticks = argc > 2 ? std::strtol(argv[2], nullptr, 10) : 1200;
    const uint64_t seed = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 0;

    Program program(elodin_model_open(), elodin_model_close);
    if (!program)
        fail("open");
    std::printf("%s model with %zu components:\n", elodin_model_backend(),
        elodin_model_component_count(program.get()));
    for (size_t i = 0; i < elodin_model_component_count(program.get()); ++i)
        std::printf("  %-22s %3zu bytes\n", elodin_model_component_name(program.get(), i),
            elodin_model_component_size(program.get(), i));

    Instance ball(elodin_model_create(program.get()), elodin_model_destroy);
    if (!ball)
        fail("create");
    if (elodin_model_write(ball.get(), component(program.get(), "seed"), &seed, sizeof seed))
        fail("write seed");

    const size_t world_pos = component(program.get(), "world_pos");
    const size_t world_vel = component(program.get(), "world_vel");
    std::FILE* out = std::fopen(path, "w");
    if (!out) {
        std::perror(path);
        return 1;
    }
    std::fprintf(out, "tick");
    for (int i = 0; i < 7; ++i)
        std::fprintf(out, ",world_pos_%d", i);
    for (int i = 0; i < 6; ++i)
        std::fprintf(out, ",world_vel_%d", i);
    std::fprintf(out, "\n");

    double pos[7];
    double vel[6];
    for (long tick = 0;; ++tick) {
        if (elodin_model_read(ball.get(), world_pos, pos, sizeof pos)
            || elodin_model_read(ball.get(), world_vel, vel, sizeof vel))
            fail("read");
        // %.17g round-trips every double, so the CSV holds the exact state.
        std::fprintf(out, "%ld", tick);
        for (double value : pos)
            std::fprintf(out, ",%.17g", value);
        for (double value : vel)
            std::fprintf(out, ",%.17g", value);
        std::fprintf(out, "\n");
        if (tick == ticks)
            break;
        if (elodin_model_step(ball.get()))
            fail("step");
    }
    std::fclose(out);

    std::printf("after %ld ticks the ball is at (%.6f, %.6f, %.6f); wrote %s\n", ticks, pos[4],
        pos[5], pos[6], path);
    return 0;
}
