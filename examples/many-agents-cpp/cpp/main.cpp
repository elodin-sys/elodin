// The many-agents SITL as one C++ program: the simulation step exported from Elodin, each
// agent's flight software, and the tick loop Elodin runs around them, in the same order:
//
//   1. step the simulation;
//   2. send every agent's state to its flight software and write the commands it returns;
//   3. record what changed;
//   4. wait for the next tick when paced.
//
// With --witness it records the same deterministic witness Elodin records for the `lockstep`
// mode of examples/many-agents, so `elodin determinism compare` checks the two runs byte for byte.
#include "exchange.hpp"
#include "model.hpp"
#include "pacing.hpp"
#include "protocol.hpp"
#include "witness.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int64_t TICK_US = 1000;

struct Options {
    uint64_t ticks = 60000;
    bool realtime = true;
    std::string fsw = "in-process";
    std::string fsw_binary;
    size_t workers = 0;
    uint16_t port_base = 13000;
    long timeout_ms = 250;
    std::string witness;
    bool witness_raw = false;
    std::string csv;
    std::string receipt;
    std::optional<uint64_t> fault_tick;
    size_t fault_agent = 7;
    size_t fault_byte = 3;
    uint8_t fault_mask = 1;
};

const char* USAGE = R"(usage: many_agents_cpp [options]
  --ticks N             ticks to run (60000)
  --realtime 0|1        keep wall-clock time at 1 kHz, or run as fast as possible (1)
  --fsw in-process|processes
                        call each agent's flight software directly, or run one many_agents_fsw
                        process per agent over UDP (in-process)
  --fsw-binary PATH     the flight-software process (many_agents_fsw next to this program)
  --port-base N         first UDP port; each agent uses two (13000)
  --timeout-ms N        how long to wait for the flight software (250)
  --workers N           threads of the simulation step; 0 for the exported count (0)
  --witness PATH        record the run's deterministic witness
  --witness-raw         keep every tick's exact bytes in the witness
  --csv PATH            write each agent's position, velocity and command at every tick
  --receipt PATH        write the run's timings as JSON
  --fault-tick N        flip bits of one agent's response at tick N
  --fault-agent N       the agent of the fault (7)
  --fault-byte N        the byte of its response (3)
  --fault-mask N        the bits that flip (1)
)";

Options parse(int argc, char** argv)
{
    Options options;
    const std::string program = argv[0];
    const size_t slash = program.rfind('/');
    options.fsw_binary = (slash == std::string::npos ? "." : program.substr(0, slash)) + "/many_agents_fsw";
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--witness-raw") {
            options.witness_raw = true;
            continue;
        }
        if (flag == "--help" || i + 1 >= argc) {
            std::fputs(USAGE, flag == "--help" ? stdout : stderr);
            std::exit(flag == "--help" ? 0 : 2);
        }
        const std::string value = argv[++i];
        const auto integer = [&] { return std::strtoull(value.c_str(), nullptr, 10); };
        if (flag == "--ticks")
            options.ticks = integer();
        else if (flag == "--realtime")
            options.realtime = value != "0";
        else if (flag == "--fsw")
            options.fsw = value;
        else if (flag == "--fsw-binary")
            options.fsw_binary = value;
        else if (flag == "--port-base")
            options.port_base = static_cast<uint16_t>(integer());
        else if (flag == "--timeout-ms")
            options.timeout_ms = static_cast<long>(integer());
        else if (flag == "--workers")
            options.workers = integer();
        else if (flag == "--witness")
            options.witness = value;
        else if (flag == "--csv")
            options.csv = value;
        else if (flag == "--receipt")
            options.receipt = value;
        else if (flag == "--fault-tick")
            options.fault_tick = integer();
        else if (flag == "--fault-agent")
            options.fault_agent = integer();
        else if (flag == "--fault-byte")
            options.fault_byte = integer();
        else if (flag == "--fault-mask")
            options.fault_mask = static_cast<uint8_t>(integer());
        else {
            std::fprintf(stderr, "unknown option %s\n%s", flag.c_str(), USAGE);
            std::exit(2);
        }
    }
    return options;
}

// The components the loop reads and writes, by index in the model.
struct Components {
    size_t position, velocity, sensor_frame, diagnostic_bank, command;
    size_t object_position, object_velocity, object_signal;

    explicit Components(const Model& model)
        : position(model.component("position"))
        , velocity(model.component("velocity"))
        , sensor_frame(model.component("sensor_frame"))
        , diagnostic_bank(model.component("diagnostic_bank"))
        , command(model.component("command"))
        , object_position(model.component("object_position"))
        , object_velocity(model.component("object_velocity"))
        , object_signal(model.component("object_signal"))
    {
    }
};

// One component's values at the moment, with the row width that splits them by entity.
struct Column {
    size_t width;
    std::vector<double> values;

    const double* row(size_t index) const { return values.data() + index * width; }
};

// The rows `rows` of `column`, back to back: a field of a witnessed batch.
std::vector<double> gather(const Column& column, const std::vector<size_t>& rows)
{
    std::vector<double> out;
    for (size_t row : rows)
        out.insert(out.end(), column.row(row), column.row(row) + column.width);
    return out;
}

// Each row's last recorded value, to decide which rows a commit records. Elodin records a row
// only when its bytes differ from the last value it recorded, starting from the initial state.
class Recorder {
public:
    explicit Recorder(const Column& initial)
        : last_(initial.values)
        , width_(initial.width)
    {
    }

    // One flag per row: 1 when this commit records it.
    std::vector<uint8_t> commit(const Column& now)
    {
        std::vector<uint8_t> recorded(last_.size() / width_);
        for (size_t row = 0; row < recorded.size(); ++row) {
            double* last = last_.data() + row * width_;
            if (std::memcmp(last, now.row(row), width_ * sizeof(double)) != 0) {
                std::memcpy(last, now.row(row), width_ * sizeof(double));
                recorded[row] = 1;
            }
        }
        return recorded;
    }

private:
    std::vector<double> last_;
    size_t width_;
};

std::vector<uint8_t> gather_flags(const std::vector<uint8_t>& flags, const std::vector<size_t>& rows)
{
    std::vector<uint8_t> out;
    for (size_t row : rows)
        out.push_back(flags[row]);
    return out;
}

template <typename T>
void add(witness::Writer& writer, const std::vector<T>& values)
{
    writer.add(values.data(), values.size() * sizeof(T));
}

int run(const Options& options)
{
    Model model(options.workers);
    const Components c(model);
    const size_t agents = model.rows(c.position);
    const size_t objects = model.rows(c.object_position);
    if (options.fault_tick && (options.fault_agent >= agents || options.fault_byte >= protocol::RESPONSE_SIZE))
        throw std::runtime_error("the fault needs an agent below " + std::to_string(agents)
            + " and a byte below " + std::to_string(protocol::RESPONSE_SIZE));
    auto column = [&](size_t component, size_t width) {
        return Column { width, model.read(component) };
    };

    // The rows of each batch Elodin's example witnesses: `exchange` has a row per agent with the
    // object it follows, `objects` a row per object.
    std::vector<size_t> agent_rows, observed_rows, object_rows;
    for (size_t agent = 0; agent < agents; ++agent) {
        agent_rows.push_back(agent);
        observed_rows.push_back(agent % objects);
    }
    for (size_t object = 0; object < objects; ++object)
        object_rows.push_back(object);

    std::unique_ptr<witness::Writer> witness;
    if (!options.witness.empty()) {
        std::map<uint64_t, std::string> names;
        auto entities = [&](size_t component, const std::vector<size_t>& rows) {
            std::vector<uint64_t> ids;
            for (size_t row : rows) {
                ids.push_back(model.entity_id(component, row));
                names[ids.back()] = model.entity_name(component, row);
            }
            return ids;
        };
        auto field = [&](const char* name, size_t component, uint64_t width, bool write,
                         const std::vector<size_t>& rows) {
            return witness::Field { name, name, { width }, width * 8, write, entities(component, rows) };
        };
        std::vector<witness::Batch> batches = {
            { "exchange",
                {
                    field("position", c.position, 3, false, agent_rows),
                    field("velocity", c.velocity, 3, false, agent_rows),
                    field("sensor_frame", c.sensor_frame, 16, false, agent_rows),
                    field("diagnostic_bank", c.diagnostic_bank, 32, false, agent_rows),
                    field("object_position", c.object_position, 3, false, observed_rows),
                    field("object_velocity", c.object_velocity, 3, false, observed_rows),
                    field("object_signal", c.object_signal, 16, false, observed_rows),
                    field("command", c.command, 3, true, agent_rows),
                } },
            { "objects",
                {
                    field("object_position", c.object_position, 3, false, object_rows),
                    field("object_velocity", c.object_velocity, 3, false, object_rows),
                    field("object_signal", c.object_signal, 16, false, object_rows),
                } },
        };
        std::vector<witness::Channel> channels = {
            { "controller.ingress", 0, protocol::STATE_SIZE, false },
            { "controller.response", 0, protocol::RESPONSE_SIZE, false },
            { "controller.reply", 0, protocol::COMMAND_SIZE, true },
        };
        // The exchange stands in for the example's native hook, under its name, so Elodin can
        // replay it from this witness; the second hook writes nothing until the last tick.
        std::vector<witness::Hook> hooks = {
            { "many_agent_step", "exchange", { { 0, agents } } },
            { "finish_step", "", {} },
        };
        witness = std::make_unique<witness::Writer>(options.witness, batches, channels, hooks, names,
            options.witness_raw);
    }

    std::FILE* csv = nullptr;
    if (!options.csv.empty()) {
        csv = std::fopen(options.csv.c_str(), "w");
        if (!csv)
            throw std::runtime_error("cannot create " + options.csv);
        std::fputs("tick,agent,x,y,z,vx,vy,vz,cx,cy,cz\n", csv);
    }

    std::vector<Recorder> recorders;
    const std::vector<std::pair<size_t, size_t>> recorded_columns = {
        { c.position, 3 },
        { c.velocity, 3 },
        { c.sensor_frame, 16 },
        { c.diagnostic_bank, 32 },
        { c.command, 3 },
        { c.object_position, 3 },
        { c.object_velocity, 3 },
        { c.object_signal, 16 },
    };
    for (const auto& [component, width] : recorded_columns)
        recorders.emplace_back(column(component, width));

    const auto exchange = make_exchange(options.fsw, options.fsw_binary, agents, options.port_base,
        std::chrono::milliseconds(options.timeout_ms));
    std::vector<uint8_t> states(agents * protocol::STATE_SIZE, 0);
    std::vector<uint8_t> replies(agents * protocol::COMMAND_SIZE);
    std::vector<uint8_t> responses(agents * protocol::RESPONSE_SIZE);
    std::vector<double> command(agents * 3);
    Phase step_phase, exchange_phase, record_phase, cycle_phase;
    Pacer pacer(std::chrono::microseconds(TICK_US), options.realtime);

    const auto started = Clock::now();
    for (uint64_t tick = 0; tick < options.ticks; ++tick) {
        const auto cycle_start = Clock::now();
        model.step();
        const auto stepped = Clock::now();

        const Column position = column(c.position, 3), velocity = column(c.velocity, 3);
        const Column sensor_frame = column(c.sensor_frame, 16);
        const Column diagnostic_bank = column(c.diagnostic_bank, 32);
        const Column object_position = column(c.object_position, 3);
        const Column object_velocity = column(c.object_velocity, 3);
        const Column object_signal = column(c.object_signal, 16);
        const Column previous_command = column(c.command, 3);
        if (witness) {
            witness->begin(tick, static_cast<int64_t>(tick) * TICK_US);
            add(*witness, position.values);
            add(*witness, velocity.values);
            add(*witness, sensor_frame.values);
            add(*witness, diagnostic_bank.values);
            add(*witness, gather(object_position, observed_rows));
            add(*witness, gather(object_velocity, observed_rows));
            add(*witness, gather(object_signal, observed_rows));
            add(*witness, previous_command.values);
            add(*witness, object_position.values);
            add(*witness, object_velocity.values);
            add(*witness, object_signal.values);
        }

        for (size_t agent = 0; agent < agents; ++agent) {
            uint8_t* state = states.data() + agent * protocol::STATE_SIZE;
            const size_t object = observed_rows[agent];
            protocol::write_header(state, protocol::STATE_MAGIC, tick);
            size_t at = 0;
            auto put = [&](const Column& from, size_t row) {
                for (size_t i = 0; i < from.width; ++i)
                    protocol::write_f64(state, at++, from.row(row)[i]);
            };
            put(position, agent);
            put(velocity, agent);
            put(object_position, object);
            put(object_velocity, object);
            put(sensor_frame, agent);
            put(diagnostic_bank, agent);
            put(object_signal, object);
        }
        exchange->run(states.data(), replies.data());
        for (size_t agent = 0; agent < agents; ++agent) {
            const uint8_t* reply = replies.data() + agent * protocol::COMMAND_SIZE;
            uint64_t reply_tick;
            if (!protocol::read_header(reply, protocol::COMMAND_MAGIC, &reply_tick) || reply_tick != tick)
                throw std::runtime_error("invalid command packet header");
            std::memcpy(responses.data() + agent * protocol::RESPONSE_SIZE, reply + protocol::HEADER_SIZE,
                protocol::RESPONSE_SIZE);
        }
        if (options.fault_tick == tick)
            responses[options.fault_agent * protocol::RESPONSE_SIZE + options.fault_byte] ^= options.fault_mask;
        for (size_t agent = 0; agent < agents; ++agent)
            std::memcpy(command.data() + agent * 3, responses.data() + agent * protocol::RESPONSE_SIZE,
                3 * sizeof(double));
        model.write(c.command, command);
        const auto exchanged = Clock::now();

        if (witness) {
            // After the exchange: what it sent, what it took from the replies, the commands it
            // wrote, no deferred writes, and the replies themselves, kept but never hashed.
            witness->add(states.data(), states.size());
            add(*witness, responses);
            add(*witness, command);
            witness->add(nullptr, 0);
            add(*witness, replies);
            // After the second hook, which fills no channel.
            witness->add(nullptr, 0);
            witness->add(nullptr, 0);
            add(*witness, command);
            witness->add(nullptr, 0);
            witness->add(nullptr, 0);
        }
        const Column written_command { 3, command };
        const std::vector<const Column*> committed = { &position, &velocity, &sensor_frame,
            &diagnostic_bank, &written_command, &object_position, &object_velocity, &object_signal };
        std::vector<std::vector<uint8_t>> flags;
        for (size_t i = 0; i < recorders.size(); ++i)
            flags.push_back(recorders[i].commit(*committed[i]));
        if (witness) {
            // Which rows the commit recorded, per witnessed field.
            add(*witness, flags[0]);
            add(*witness, flags[1]);
            add(*witness, flags[2]);
            add(*witness, flags[3]);
            add(*witness, gather_flags(flags[5], observed_rows));
            add(*witness, gather_flags(flags[6], observed_rows));
            add(*witness, gather_flags(flags[7], observed_rows));
            add(*witness, flags[4]);
            add(*witness, flags[5]);
            add(*witness, flags[6]);
            add(*witness, flags[7]);
            witness->end();
        }
        if (csv)
            for (size_t agent = 0; agent < agents; ++agent) {
                const double* p = position.row(agent);
                const double* v = velocity.row(agent);
                const double* u = command.data() + agent * 3;
                std::fprintf(csv,
                    "%llu,%zu,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                    static_cast<unsigned long long>(tick), agent, p[0], p[1], p[2], v[0], v[1], v[2],
                    u[0], u[1], u[2]);
            }
        const auto recorded = Clock::now();

        step_phase.add(stepped - cycle_start);
        exchange_phase.add(exchanged - stepped);
        record_phase.add(recorded - exchanged);
        cycle_phase.add(recorded - cycle_start);
        pacer.wait();
    }
    const double wall_s = std::chrono::duration<double>(Clock::now() - started).count();

    std::string root;
    if (witness)
        root = witness->finish();
    if (csv)
        std::fclose(csv);
    const double hz = options.ticks / wall_s;
    std::printf("%zu agents, %llu ticks in %.2f s (%.1f Hz), %llu late, %llu schedule resets%s%s\n",
        agents, static_cast<unsigned long long>(options.ticks), wall_s, hz,
        static_cast<unsigned long long>(pacer.late()), static_cast<unsigned long long>(pacer.resets()),
        root.empty() ? "" : ", witness ", root.c_str());
    if (!options.receipt.empty()) {
        std::FILE* out = std::fopen(options.receipt.c_str(), "w");
        if (!out)
            throw std::runtime_error("cannot create " + options.receipt);
        std::fprintf(out,
            "{\"agents\":%zu,\"ticks\":%llu,\"fsw\":\"%s\",\"workers\":%zu,\"realtime\":%s,"
            "\"wall_s\":%.6f,\"effective_hz\":%.3f,\"late_ticks\":%llu,\"late_fraction\":%.6f,"
            "\"pacing_resets\":%llu,\"step\":%s,\"exchange\":%s,\"record\":%s,\"cycle\":%s,"
            "\"witness_root\":\"%s\"}\n",
            agents, static_cast<unsigned long long>(options.ticks), options.fsw.c_str(), options.workers,
            options.realtime ? "true" : "false", wall_s, hz, static_cast<unsigned long long>(pacer.late()),
            options.ticks ? static_cast<double>(pacer.late()) / options.ticks : 0.0,
            static_cast<unsigned long long>(pacer.resets()), step_phase.json().c_str(),
            exchange_phase.json().c_str(), record_phase.json().c_str(), cycle_phase.json().c_str(),
            root.c_str());
        std::fclose(out);
    }
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        return run(parse(argc, argv));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "many_agents_cpp: %s\n", error.what());
        return 1;
    }
}
