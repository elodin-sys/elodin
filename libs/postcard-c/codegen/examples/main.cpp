#include "foo.hpp"
#include <cstdint>
#include <print>
#include <unordered_map>

int main() {
    Foo foo =  {
        .id = 123,
        .name = std::string("PostcardTest"),
        .values = std::vector<int64_t> {-10, 20, -30},
        .byte_arr = std::vector<uint8_t> { 1, 2, 3 },
        .is_active = true,
        .metadata = std::unordered_map<std::string, int32_t> { {"key1", 2}, {"key2", -1}},
    };
    auto out = foo.encode_vec();
    auto out_span = std::span(reinterpret_cast<uint8_t*>(out.data()), out.size());
    std::println("encoded data: {}", out_span);

    auto input = std::span<const std::byte>(out.begin(), out.end());
    std::println("input len {}", input.size());

    Foo decoded;
    auto result = decoded.decode(input);
    std::println("result: {}", static_cast<uint32_t>(result));
    std::println("id: {}", decoded.id);
    std::println("name: {}", decoded.name);
    std::println("values: {}", decoded.values);
    std::println("byte_arr: {}", decoded.byte_arr);
    std::println("is_active: {}", decoded.is_active);
    std::println("metadata: {}", decoded.metadata);
}
