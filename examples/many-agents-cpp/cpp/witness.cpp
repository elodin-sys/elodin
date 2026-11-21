#include "witness.hpp"

#include <cstring>
#include <stdexcept>

namespace witness {

namespace {

    constexpr uint32_t VERSION = 1;
    constexpr uint8_t TICK = 1, TRAILER = 2;
    // Elodin starts the witness clock at the run's start timestamp, which its example sets to 0.
    constexpr int64_t START_TIMESTAMP = 0;

    std::string quote(const std::string& text)
    {
        std::string out = "\"";
        for (char c : text) {
            if (c == '"' || c == '\\') {
                out += '\\';
                out += c;
            } else if (static_cast<unsigned char>(c) < 0x20) {
                char escaped[8];
                std::snprintf(escaped, sizeof escaped, "\\u%04x", c);
                out += escaped;
            } else {
                out += c;
            }
        }
        return out + "\"";
    }

    template <typename T, typename F>
    std::string list(const std::vector<T>& items, F each)
    {
        std::string out = "[";
        for (size_t i = 0; i < items.size(); ++i)
            out += (i ? "," : "") + each(items[i]);
        return out + "]";
    }

    std::string number(uint64_t value) { return std::to_string(value); }

    const char* kind_name(Kind kind)
    {
        switch (kind) {
        case Kind::Field:
            return "field";
        case Kind::Channel:
            return "channel";
        case Kind::External:
            return "external";
        case Kind::Deferred:
            return "deferred";
        case Kind::Recorded:
            return "recorded";
        }
        return "";
    }

    // Elodin's stages and records (`records` in libs/nox-py/src/witness/mod.rs): after the systems
    // every field; after each hook its channels, its write fields, its deferred writes and its
    // external channels; after the commit a recorded flag per row of every field.
    std::vector<Record> layout(const std::vector<Batch>& batches, const std::vector<Channel>& channels,
        const std::vector<std::string>& stages)
    {
        std::vector<Record> records;
        for (size_t stage = 0; stage < stages.size(); ++stage) {
            const bool systems = stages[stage] == "after_systems";
            const bool commit = stages[stage] == "after_commit";
            auto fields = [&](Kind kind, bool writes_only) {
                for (size_t b = 0; b < batches.size(); ++b)
                    for (size_t f = 0; f < batches[b].fields.size(); ++f) {
                        const Field& field = batches[b].fields[f];
                        if (writes_only && !field.write)
                            continue;
                        records.push_back({ stage, kind, batches[b].name + "." + field.name,
                            static_cast<long>(b), f, kind == Kind::Recorded ? 1 : field.row_bytes });
                    }
            };
            auto channel_records = [&](bool external) {
                for (size_t c = 0; c < channels.size(); ++c)
                    if (channels[c].external == external)
                        records.push_back({ stage, external ? Kind::External : Kind::Channel,
                            channels[c].name, static_cast<long>(channels[c].batch), c,
                            channels[c].row_bytes });
            };
            if (systems) {
                fields(Kind::Field, false);
            } else if (commit) {
                fields(Kind::Recorded, false);
            } else {
                channel_records(false);
                fields(Kind::Field, true);
                records.push_back({ stage, Kind::Deferred, "deferred", -1, 0, 0 });
                channel_records(true);
            }
        }
        return records;
    }

    // The schema as serde_json writes Elodin's `Schema`: compact, fields in declaration order. Its
    // BLAKE3 is checked when Elodin opens the file.
    std::string schema_json(const std::map<uint64_t, std::string>& entities,
        const std::vector<Batch>& batches, const std::vector<Channel>& channels,
        const std::vector<std::string>& stages, const std::vector<Record>& records)
    {
        std::string json = "{\"entities\":[";
        bool first = true;
        for (const auto& [id, name] : entities) {
            json += (first ? "" : ",") + std::string("[") + number(id) + "," + quote(name) + "]";
            first = false;
        }
        json += "],\"batches\":" + list(batches, [](const Batch& batch) {
            return "{\"name\":" + quote(batch.name) + ",\"fields\":" + list(batch.fields, [](const Field& f) {
                return "{\"name\":" + quote(f.name) + ",\"component\":" + quote(f.component)
                    + ",\"dtype\":\"float64\",\"shape\":" + list(f.shape, number)
                    + ",\"row_bytes\":" + number(f.row_bytes) + ",\"access\":"
                    + (f.write ? "\"write\"" : "\"read\"") + ",\"entities\":" + list(f.entities, number)
                    + "}";
            }) + "}";
        });
        json += ",\"channels\":" + list(channels, [](const Channel& c) {
            return "{\"name\":" + quote(c.name) + ",\"batch\":" + number(c.batch) + ",\"row_bytes\":"
                + number(c.row_bytes) + ",\"external\":" + (c.external ? "true" : "false") + "}";
        });
        json += ",\"stages\":" + list(stages, quote);
        json += ",\"records\":" + list(records, [](const Record& r) {
            return "{\"stage\":" + number(r.stage) + ",\"kind\":\"" + kind_name(r.kind)
                + "\",\"name\":" + quote(r.name) + ",\"batch\":"
                + (r.batch < 0 ? "null" : number(static_cast<uint64_t>(r.batch))) + ",\"index\":"
                + number(r.index) + ",\"row_bytes\":" + number(r.row_bytes) + "}";
        });
        return json + "}";
    }

    void put(std::FILE* out, const void* bytes, size_t size)
    {
        if (std::fwrite(bytes, 1, size, out) != size)
            throw std::runtime_error("cannot write the witness");
    }

    template <typename T>
    void put_value(std::FILE* out, T value) { put(out, &value, sizeof value); }

} // namespace

Writer::Writer(const std::string& path, std::vector<Batch> batches, std::vector<Channel> channels,
    std::vector<Hook> hooks, const std::map<uint64_t, std::string>& entity_names, bool raw)
    : raw_(raw)
{
    std::vector<std::string> stages = { "after_systems" };
    for (size_t hook = 0; hook < hooks.size(); ++hook)
        stages.push_back("after_hook:" + std::to_string(hook));
    stages.push_back("after_commit");
    records_ = layout(batches, channels, stages);
    stage_hashed_.assign(stages.size(), 0);
    for (const Record& record : records_)
        stage_hashed_[record.stage] += record.kind != Kind::External;

    std::map<uint64_t, std::string> entities;
    for (const Batch& batch : batches)
        for (const Field& field : batch.fields)
            for (uint64_t id : field.entities)
                entities[id] = entity_names.at(id);
    const std::string schema = schema_json(entities, batches, channels, stages, records_);
    const Digest schema_digest = Blake3::hash(schema.data(), schema.size());
    std::string labels = "{\"build\":\"many_agents_cpp\",\"start_timestamp\":"
        + std::to_string(START_TIMESTAMP) + ",\"hooks\":" + list(hooks, [](const Hook& hook) {
              return "{\"name\":" + quote(hook.name) + ",\"batch\":"
                  + (hook.batch.empty() ? "null" : quote(hook.batch)) + ",\"partitions\":"
                  + list(hook.partitions, [](const std::pair<uint64_t, uint64_t>& rows) {
                        return "[" + number(rows.first) + "," + number(rows.second) + "]";
                    })
                  + "}";
          })
        + "}";
    const std::string header = "{\"format\":\"elodin.witness\",\"version\":1,"
                               "\"hash\":\"blake3-256 over little-endian bytes\",\"schema_digest\":\""
        + hex(schema_digest) + "\",\"schema\":" + schema + ",\"labels\":" + labels + "}";

    out_ = std::fopen(path.c_str(), "wb");
    if (!out_)
        throw std::runtime_error("cannot create " + path);
    put(out_, "ELWITNES", 8);
    put_value(out_, VERSION);
    put_value(out_, static_cast<uint32_t>(header.size()));
    put(out_, header.data(), header.size());
    root_.update(schema_digest.data(), schema_digest.size());
}

Writer::~Writer()
{
    if (out_)
        std::fclose(out_);
}

void Writer::begin(uint64_t tick, int64_t timestamp)
{
    tick_ = tick;
    timestamp_ = timestamp;
    bytes_.clear();
    ends_.clear();
}

void Writer::add(const void* bytes, size_t size)
{
    const uint8_t* data = static_cast<const uint8_t*>(bytes);
    bytes_.insert(bytes_.end(), data, data + size);
    ends_.push_back(bytes_.size());
}

void Writer::end()
{
    if (ends_.size() != records_.size())
        throw std::logic_error("a witness tick takes every record of the schema");
    std::vector<Digest> digests;
    for (size_t i = 0; i < records_.size(); ++i) {
        if (records_[i].kind == Kind::External)
            continue;
        const size_t start = i ? ends_[i - 1] : 0;
        const size_t size = ends_[i] - start;
        // A channel no hook filled is all zeros, not the digest of no bytes.
        digests.push_back(records_[i].kind == Kind::Channel && size == 0
                ? Digest {}
                : Blake3::hash(bytes_.data() + start, size));
    }
    // A stage's digest covers its records' digests; the tick's covers the tick, its simulation
    // time and its stages'; the root covers the schema and every tick.
    Blake3 tick;
    tick.update(&tick_, 8);
    const int64_t sim_time = timestamp_ - START_TIMESTAMP;
    tick.update(&sim_time, 8);
    size_t next = 0;
    for (size_t hashed : stage_hashed_) {
        const Digest stage = Blake3::hash(digests.data() + next, hashed * sizeof(Digest));
        tick.update(stage.data(), stage.size());
        next += hashed;
    }
    const Digest tick_digest = tick.finalize();
    root_.update(tick_digest.data(), tick_digest.size());

    put_value(out_, TICK);
    put_value(out_, tick_);
    put_value(out_, timestamp_);
    put(out_, digests.data(), digests.size() * sizeof(Digest));
    put_value(out_, static_cast<uint8_t>(raw_));
    if (raw_)
        for (size_t i = 0; i < records_.size(); ++i) {
            const size_t start = i ? ends_[i - 1] : 0;
            put_value(out_, static_cast<uint32_t>(ends_[i] - start));
            put(out_, bytes_.data() + start, ends_[i] - start);
        }
    ++ticks_;
}

std::string Writer::finish()
{
    const std::string root = hex(root_.finalize());
    const std::string trailer
        = "{\"ticks\":" + std::to_string(ticks_) + ",\"root\":\"" + root + "\",\"failure\":null}";
    put_value(out_, TRAILER);
    put_value(out_, static_cast<uint32_t>(trailer.size()));
    put(out_, trailer.data(), trailer.size());
    std::fclose(out_);
    out_ = nullptr;
    return root;
}

} // namespace witness
