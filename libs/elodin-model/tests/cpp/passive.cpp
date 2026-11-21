#include "elodin_model.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#ifdef __linux__
#include <dirent.h>

std::size_t proc_status_value(const char* key)
{
    std::ifstream input("/proc/self/status");
    std::string line;
    while (std::getline(input, line)) {
        if (line.rfind(key, 0) != 0)
            continue;
        std::istringstream value(line.substr(std::char_traits<char>::length(key)));
        std::size_t parsed;
        if (value >> parsed)
            return parsed;
    }
    throw std::runtime_error(std::string("missing /proc status field ") + key);
}

std::size_t open_fds()
{
    DIR* directory = opendir("/proc/self/fd");
    if (!directory)
        throw std::runtime_error("cannot read /proc/self/fd");
    std::size_t count = 0;
    while (readdir(directory))
        ++count;
    closedir(directory);
    return count;
}
#endif

int main()
{
    try {
        if (elodin_model_abi_version() != ELODIN_MODEL_ABI_VERSION)
            throw std::runtime_error("model ABI version mismatch");
        const std::string build_info = elodin_model_build_info();
        if (build_info.empty() || build_info.find("elodin.model-build") == std::string::npos)
            throw std::runtime_error("model build info is missing its format");

        elodin::check(elodin_model_configure_workers(1), "configure workers");
        elodin::Program program;
        if (elodin_model_effective_workers(program.get()) != 1)
            throw std::runtime_error("caller worker setting was not applied");
        if (elodin_model_configure_workers(2) != ELODIN_MODEL_ERR_STATE)
            throw std::runtime_error("worker count changed after open");
        if (elodin_model_schema_abi(program.get()) != 2)
            throw std::runtime_error("model schema is not version 2");
        double nominal_dt = 0.0;
        elodin::check(
            elodin_model_nominal_time_step(program.get(), &nominal_dt),
            "nominal time step");
        if (std::abs(nominal_dt - 1.0 / 120.0) > 1.0e-15)
            throw std::runtime_error("unexpected nominal time step");

        const std::size_t world_pos = program.component("world_pos");
        const elodin::ComponentType type = program.component_type(world_pos);
        if (type.dtype != "float64" || type.element_shape != std::vector<std::size_t> { 7 })
            throw std::runtime_error("unexpected world_pos type");

        elodin::Instance run(program);
        const auto initial = run.read_as<double>(world_pos);
        run.step(); // Warm allocator and lazy runtime state before resource accounting.
        run.reset();
#ifdef __linux__
        const std::size_t threads_before = proc_status_value("Threads:");
        const std::size_t rss_before_kib = proc_status_value("VmRSS:");
        const std::size_t fds_before = open_fds();
#endif
        std::vector<double> first_step;
        for (int case_index = 0; case_index < 1000; ++case_index) {
            run.reset();
            if (run.read_as<double>(world_pos) != initial)
                throw std::runtime_error("reset did not restore the initial state");
            run.step();
            const auto stepped = run.read_as<double>(world_pos);
            if (case_index == 0)
                first_step = stepped;
            else if (stepped != first_step)
                throw std::runtime_error("same-process reset changed the result");
        }
#ifdef __linux__
        if (proc_status_value("Threads:") != threads_before)
            throw std::runtime_error("same-process cases leaked threads");
        if (open_fds() != fds_before)
            throw std::runtime_error("same-process cases leaked file descriptors");
        if (proc_status_value("VmRSS:") > rss_before_kib + 16 * 1024)
            throw std::runtime_error("same-process cases grew RSS by more than 16 MiB");
#endif

        std::cout << "passive model ABI smoke passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
