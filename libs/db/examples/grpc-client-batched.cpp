#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <grpcpp/grpcpp.h>
#include <openssl/evp.h>

#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

#include "elodin/db/v1/ingest.grpc.pb.h"

namespace v1 = elodin::db::v1;
using namespace std::chrono_literals;

namespace {

constexpr std::size_t kComponentCount = 8;
constexpr std::string_view kMessageName = "GrpcCppReference";

struct Options {
    std::string address = "127.0.0.1:2242";
    std::uint64_t ticks = 100;
    double frequency_hz = 100.0;
};

Options parse_options(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--help") {
            std::cout << "Usage: grpc-client-batched [--address HOST:PORT] "
                         "[--ticks COUNT] [--frequency HZ]\n";
            std::exit(0);
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value for " + std::string(arg));
        }
        const std::string value(argv[++i]);
        if (arg == "--address") {
            options.address = value;
        } else if (arg == "--ticks") {
            options.ticks = std::stoull(value);
        } else if (arg == "--frequency") {
            options.frequency_hz = std::stod(value);
        } else {
            throw std::invalid_argument("unknown option " + std::string(arg));
        }
    }
    if (options.address.empty() || options.ticks == 0 || options.frequency_hz <= 0.0) {
        throw std::invalid_argument("address, ticks, and frequency must be positive");
    }
    return options;
}

std::string deterministic_bytes(const google::protobuf::MessageLite& message)
{
    std::string bytes;
    {
        google::protobuf::io::StringOutputStream output(&bytes);
        google::protobuf::io::CodedOutputStream coded(&output);
        coded.SetSerializationDeterministic(true);
        if (!message.SerializeToCodedStream(&coded)) {
            throw std::runtime_error("failed to serialize SchemaSet");
        }
    }
    return bytes;
}

std::string sha256(std::string_view bytes)
{
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest {};
    unsigned int digest_len = 0;
    auto context = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>(
        EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1
        || EVP_DigestUpdate(context.get(), bytes.data(), bytes.size()) != 1
        || EVP_DigestFinal_ex(context.get(), digest.data(), &digest_len) != 1) {
        throw std::runtime_error("failed to compute schema fingerprint");
    }
    return { reinterpret_cast<const char*>(digest.data()), digest_len };
}

v1::SchemaSet make_schema()
{
    v1::SchemaSet schema;
    auto* message = schema.add_messages();
    message->set_name(kMessageName);
    message->set_encoding(v1::ROW_ENCODING_PACKED);
    message->set_packed_size(kComponentCount * sizeof(double));
    for (std::size_t i = 0; i < kComponentCount; ++i) {
        auto* component = message->add_components();
        component->set_name("grpc_cpp.reference.signal_" + std::to_string(i));
        component->set_prim_type(v1::PRIM_TYPE_F64);
        component->set_packed_offset(i * sizeof(double));
    }
    return schema;
}

std::string make_payload(std::uint64_t tick)
{
    static_assert(std::endian::native == std::endian::little);
    std::array<double, kComponentCount> values {};
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = std::sin(static_cast<double>(tick) * 0.01 + i);
    }
    std::string payload(sizeof(values), '\0');
    std::memcpy(payload.data(), values.data(), sizeof(values));
    return payload;
}

std::int64_t monotonic_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int run(const Options& options)
{
    auto channel = grpc::CreateChannel(options.address, grpc::InsecureChannelCredentials());
    if (!channel->WaitForConnected(std::chrono::system_clock::now() + 5s)) {
        throw std::runtime_error("timed out connecting to " + options.address);
    }
    auto stub = v1::IngestService::NewStub(channel);
    grpc::ClientContext context;
    auto stream = stub->Ingest(&context);

    const auto schema = make_schema();
    v1::IngestRequest request;
    auto* open = request.mutable_open();
    open->set_client_name("grpc-cpp-reference");
    open->set_client_instance_id(
        "grpc-cpp-reference-" + std::to_string(monotonic_ns()));
    open->mutable_schema()->CopyFrom(schema);
    open->set_schema_fingerprint(sha256(deterministic_bytes(schema)));
    open->mutable_ack_policy()->set_max_unacked_rows(32);
    open->mutable_ack_policy()->set_max_ack_delay_ms(20);
    if (!stream->Write(request)) {
        throw std::runtime_error("server closed during SessionOpen");
    }

    v1::IngestResponse response;
    if (!stream->Read(&response)) {
        const auto status = stream->Finish();
        throw std::runtime_error("SessionOpen failed: " + status.error_message());
    }
    if (response.has_reject()) {
        throw std::runtime_error("session rejected: " + response.reject().detail());
    }
    if (!response.has_accept()) {
        throw std::runtime_error("expected SessionAccept");
    }
    const auto handle_it = response.accept().message_handles().find(std::string(kMessageName));
    if (handle_it == response.accept().message_handles().end()) {
        throw std::runtime_error("SessionAccept omitted the message handle");
    }
    const auto message_handle = handle_it->second;

    std::atomic<std::uint64_t> through_seq { response.accept().resume_from_seq() };
    std::atomic<std::uint64_t> row_errors { 0 };
    std::thread reader([&] {
        v1::IngestResponse incoming;
        while (stream->Read(&incoming)) {
            if (incoming.has_ack()) {
                through_seq.store(incoming.ack().through_seq(), std::memory_order_relaxed);
            } else if (incoming.has_error()) {
                ++row_errors;
                std::cerr << "RowError seq=" << incoming.error().seq()
                          << " component=" << incoming.error().component()
                          << " detail=" << incoming.error().detail() << '\n';
            } else if (incoming.has_reject()) {
                ++row_errors;
                std::cerr << "SessionReject: " << incoming.reject().detail() << '\n';
            }
        }
    });

    const auto interval = std::chrono::duration<double>(1.0 / options.frequency_hz);
    auto deadline = std::chrono::steady_clock::now();
    std::uint64_t sent = 0;
    for (std::uint64_t tick = 0; tick < options.ticks; ++tick) {
        v1::IngestRequest write;
        auto* batch = write.mutable_batch();
        batch->set_first_seq(tick + 1);
        auto* row = batch->add_rows();
        row->set_message_handle(message_handle);
        row->set_time_monotonic_ns(monotonic_ns());
        row->set_packed(make_payload(tick));
        if (!stream->Write(write)) {
            break;
        }
        ++sent;
        deadline += std::chrono::duration_cast<std::chrono::steady_clock::duration>(interval);
        std::this_thread::sleep_until(deadline);
    }

    stream->WritesDone();
    reader.join();
    const auto status = stream->Finish();
    if (!status.ok()) {
        std::cerr << "gRPC failure: " << status.error_code() << ": "
                  << status.error_message() << '\n';
        return 1;
    }
    if (row_errors.load() != 0 || through_seq.load() < sent) {
        std::cerr << "incomplete ingest: sent=" << sent
                  << " acked_through=" << through_seq.load()
                  << " row_errors=" << row_errors.load() << '\n';
        return 1;
    }
    std::cout << "sent " << sent << " batches, acked through sequence "
              << through_seq.load() << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv)
try {
    return run(parse_options(argc, argv));
} catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
}
