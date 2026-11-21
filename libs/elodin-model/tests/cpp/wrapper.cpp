// elodin_model.hpp against the rooms model of tests/fixtures/rooms, the way a caller's own class
// would hold it: no C calls, no Elodin base class, and many cases in one process.
#include "elodin_model.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __linux__
#include <dirent.h>
#include <fstream>
#include <sstream>
#endif

namespace {

void expect(bool condition, const char* what)
{
    if (!condition)
        throw std::runtime_error(std::string("expected ") + what);
}

// `call` must throw a message that contains `text`.
void expect_throw(const std::function<void()>& call, const char* text)
{
    try {
        call();
    } catch (const std::exception& error) {
        if (std::string(error.what()).find(text) == std::string::npos)
            throw std::runtime_error(std::string("expected an error about ") + text + ", got: " + error.what());
        return;
    }
    throw std::runtime_error(std::string("expected an error about ") + text);
}

// A thermostat that owns the rooms model: the shape of a caller's adapter class.
class Thermostat {
public:
    // Starts the next case from the model's initial state.
    void begin_case(double first, double second)
    {
        run_.reset();
        commands_ = { first, second };
    }

    void update()
    {
        run_.write_interface(exchange_, command_, commands_.data(), commands_.size());
        run_.step();
        run_.read_interface(exchange_, measured_, measurements_.data(), measurements_.size());
    }

    // Temperatures in the interface's order: room_1, then room_0.
    const std::array<double, 2>& measurements() const { return measurements_; }
    std::uint64_t tick() const { return run_.tick(); }

private:
    elodin::Program model_;
    elodin::Instance run_ { model_ };
    std::size_t exchange_ = model_.interface("rooms.exchange");
    std::size_t command_ = model_.field(exchange_, "command");
    std::size_t measured_ = model_.field(exchange_, "measured");
    std::array<double, 2> commands_ {};
    std::array<double, 2> measurements_ {};
};

#ifdef __linux__
std::size_t proc_status(const char* key)
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

std::size_t open_files()
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

// Successive cases in one process repeat exactly and leave no threads, files or memory behind.
void repeats_cases()
{
    Thermostat thermostat;
    thermostat.begin_case(30.0, 22.0);
    thermostat.update(); // Warm the allocator before counting resources.
#ifdef __linux__
    const std::size_t threads = proc_status("Threads:");
    const std::size_t resident_kib = proc_status("VmRSS:");
    const std::size_t files = open_files();
#endif
    for (int repeat = 0; repeat < 1000; ++repeat) {
        thermostat.begin_case(30.0, 22.0);
        thermostat.update();
        expect(thermostat.measurements() == std::array<double, 2> { 26.5, 22.5 }, "the same first step");
        thermostat.update();
        expect(thermostat.measurements() == std::array<double, 2> { 25.375, 24.375 }, "the same second step");
        expect(thermostat.tick() == 2, "two ticks since the reset");
    }
#ifdef __linux__
    expect(proc_status("Threads:") == threads, "no new threads over 1000 cases");
    expect(open_files() == files, "no new open files over 1000 cases");
    expect(proc_status("VmRSS:") <= resident_kib + 16 * 1024, "under 16 MiB of growth over 1000 cases");
#endif
}

void describes_the_model(const elodin::Program& model)
{
    expect(model.nominal_time_step() == 0.25, "a quarter-second step");
    expect(model.component_count() == 4, "four components");
    const std::size_t temperature = model.component("temperature");
    expect(std::string(model.component_name(temperature)) == "temperature", "the component's name");
    const elodin::ComponentType type = model.component_type(temperature);
    expect(type.dtype == "float64" && type.shape == std::vector<std::size_t> { 2 }, "two float64 rows");
    expect(type.element_shape.empty() && type.row_bytes == 8 && type.total_bytes == 16, "scalar rows");
    expect((type.flags & ELODIN_MODEL_COMPONENT_WRITTEN_BY_STEP) != 0, "an output");
    const std::uint32_t tick_flags = model.component_type(model.component("tick")).flags;
    expect((tick_flags & ELODIN_MODEL_COMPONENT_TICK) != 0, "the tick flag");
    expect((tick_flags & ELODIN_MODEL_COMPONENT_WRITTEN_BY_STEP) != 0, "the tick as an output");

    expect(model.rows(temperature) == 2, "two rooms");
    expect(model.row(temperature, "room_1") == 1, "room_1 in row 1");
    expect(model.entity_id(temperature, 1) == 2, "room_1's id");
    expect(std::string(model.entity_name(temperature, 0)) == "room_0", "room_0 in row 0");

    expect(model.interface_count() == 1, "one interface");
    const std::size_t exchange = model.interface("rooms.exchange");
    expect(std::string(model.interface_name(exchange)) == "rooms.exchange", "the interface's name");
    expect(model.field_count(exchange) == 2, "two fields");
    const elodin::InterfaceField command = model.interface_field(exchange, model.field(exchange, "command"));
    expect(command.name == "command" && command.writable, "a writable command");
    expect(command.component == model.component("ambient") && command.rows == 2 && command.row_bytes == 8,
        "the command's rows");
    expect(!model.interface_field(exchange, model.field(exchange, "measured")).writable, "a read-only field");
    expect(std::string(elodin::build_info()).find("elodin.model-build") != std::string::npos, "build info");

    expect_throw([&] { model.component("humidity"); }, "no component named humidity");
    expect_throw([&] { model.interface("attic"); }, "no interface named attic");
    expect_throw([&] { model.field(exchange, "setpoint"); }, "rooms.exchange has no field setpoint");
    expect_throw([&] { model.row(temperature, "attic"); }, "temperature has no entity attic");
    expect_throw([&] { model.component_name(9); }, "no such component");
    expect_throw([&] { model.component_type(9); }, "no component 9");
}

void reads_and_writes_typed_values(const elodin::Program& model)
{
    elodin::Instance run(model);
    const std::size_t temperature = model.component("temperature");
    const std::size_t ambient = model.component("ambient");
    expect(run.read<double>(temperature) == std::vector<double> { 20.0, 28.0 }, "the initial temperatures");
    run.write(ambient, std::vector<double> { 30.0, 22.0 });
    run.step();
    std::array<double, 2> values {};
    run.read(temperature, values.data(), values.size());
    expect(values == std::array<double, 2> { 22.5, 26.5 }, "one step toward the new ambient");
    expect(run.tick() == 1 && run.generation() == 2, "one write and one step");

    double row = 0.0;
    run.read_rows(temperature, 1, 1, &row, 1);
    expect(row == 26.5, "row 1 alone");
    row = 18.0;
    run.write_rows(ambient, 0, 1, &row, 1);
    expect(run.read<double>(ambient) == std::vector<double> { 18.0, 22.0 }, "row 0 alone written");

    const std::size_t exchange = model.interface("rooms.exchange");
    expect(run.read_interface<double>(exchange, model.field(exchange, "measured"))
            == std::vector<double> { 26.5, 22.5 },
        "the interface's row order");
    run.write_interface(exchange, model.field(exchange, "command"), std::vector<double> { 20.0, 20.0 });
    expect(run.read<double>(ambient) == std::vector<double> { 20.0, 20.0 }, "the command written");

    // Raw bytes hold any component.
    std::array<std::byte, 16> bytes {};
    run.read(temperature, bytes.data(), bytes.size());
    expect(std::memcmp(bytes.data(), values.data(), bytes.size()) == 0, "the same bytes");
    std::uint64_t tick = 0;
    run.read(model.component("tick"), &tick, 1);
    expect(tick == 1, "the tick as uint64");

    run.set_time_step(0.5);
    run.reset();
    run.set_time_step(0.5);
    run.step();
    expect(run.read<double>(temperature) == std::vector<double> { 24.0, 24.0 }, "a half-second step");

    // A buffer of the wrong type or length is refused before anything is copied.
    std::array<float, 4> floats {};
    std::array<bool, 16> flags {};
    expect_throw([&] { run.read(temperature, floats.data(), floats.size()); }, "temperature holds float64, not float32");
    expect_throw([&] { run.read(temperature, flags.data(), flags.size()); }, "temperature holds float64, not bool");
    expect_throw([&] { run.read(temperature, values.data(), 1); }, "read component");
    expect_throw([&] { run.read_rows(temperature, 1, 2, values.data(), 2); }, "read rows");
    expect_throw([&] { run.read<double>(9); }, "no component 9");
    expect_throw(
        [&] { run.write_interface(exchange, model.field(exchange, "measured"), values.data(), values.size()); },
        "read-only");
    expect_throw([&] { run.set_time_step(0.0); }, "set model time step");
}

} // namespace

int main()
{
    try {
        elodin::set_workers(1);
        {
            const elodin::Program model;
            expect(model.workers() == 1, "the worker count that was set");
            expect_throw([] { elodin::set_workers(2); }, "set workers");
            describes_the_model(model);
            reads_and_writes_typed_values(model);
        }
        repeats_cases();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    std::puts("the C++ wrapper matches the rooms model");
    return 0;
}
