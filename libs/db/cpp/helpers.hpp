#ifndef ELO_DB_HELPERS_H
#define ELO_DB_HELPERS_H

#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>
#include <vector>

inline uint32_t fnv1a_hash_32(const std::string_view str)
{
    uint32_t hash = 0x811c9dc5;
    size_t i = 0;
    for (auto c : str) {
        if (++i >= 32) {
            break;
        }
        hash ^= static_cast<uint8_t>(c);
        hash *= 0x01000193;
    }
    return hash;
}

inline uint64_t fnv1a_hash_64(const std::string_view str)
{
    uint64_t hash = 0xcbf29ce484222325;
    size_t i = 0;
    for (auto c : str) {
        if (++i >= 64) {
            break;
        }
        hash ^= static_cast<uint8_t>(c);
        hash *= 0x00000100000001B3;
    }
    return hash;
}

inline uint16_t fnv1a_hash_16_xor(const std::string_view str)
{
    auto hash = fnv1a_hash_32(str);
    uint16_t upper = static_cast<uint16_t>((hash >> 16) & 0xFFFF);
    uint16_t lower = static_cast<uint16_t>(hash & 0xFFFF);
    return upper ^ lower;
}

inline std::array<uint8_t, 2> msg_id(const std::string_view str)
{
    auto hash = fnv1a_hash_16_xor(str);
    return { static_cast<uint8_t>(hash & 0xff), static_cast<uint8_t>((hash >> 8) & 0xff) };
}

inline uint64_t component_id(const std::string_view str)
{
    auto hash = fnv1a_hash_64(str) & ~(1ul << 63);
    return hash;
}

enum class PacketType : uint8_t {
    MSG = 0,
    TABLE = 1,
    TIME_SERIES = 2
};

struct PacketHeader {
    uint32_t len;
    PacketType ty;
    std::array<uint8_t, 2> packet_id;
    uint8_t request_id;
};

template <typename T>
class Msg {
    PacketHeader header;
    T payload;

public:
    Msg(T p)
    {
        auto packet_id = msg_id(T::TYPE_NAME);
        header = PacketHeader {
            .len = 0,
            .ty = PacketType::MSG,
            .packet_id = packet_id,
            .request_id = 0,
        };
        payload = p;
    }

    std::vector<uint8_t> encode_vec()
    {
        auto t_size = payload.encoded_size();
        header.len = t_size + 4;
        auto header_len = sizeof(PacketHeader);
        auto buf = std::vector<uint8_t>(t_size + header_len);
        std::memcpy(buf.data(), &header, header_len);
        auto span = std::span<uint8_t>(buf).subspan(header_len, t_size);
        postcard_slice_t slice;
        postcard_init_slice(&slice, span.data(), span.size());
        auto res = payload.encode_raw(&slice);
        if (res == POSTCARD_SUCCESS) {
            buf.resize(slice.len + header_len);
        } else {
            buf.clear();
        }

        return buf;
    }
};

SetComponentMetadata set_component_name(std::string name)
{
    return SetComponentMetadata(ComponentMetadata {
        .component_id = component_id(name),
        .name = std::move(name),
    });
}

#endif
