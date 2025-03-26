#ifndef ELO_DB_HELPERS_H
#define ELO_DB_HELPERS_H

#include <cstring>
#include <span>
#include <vector>

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
    Msg(std::array<uint8_t, 2> packet_id, T p)
    {
        header = PacketHeader {
            .len = 0,
            .ty = PacketType::MSG,
            .packet_id = packet_id,
            .request_id = 0,
        };
        payload = p;
    }

    std::vector<std::byte> encode_vec()
    {
        auto t_size = payload.encoded_size();
        header.len = t_size + 4;
        auto header_len = sizeof(PacketHeader);
        auto buf = std::vector<std::byte>(t_size + header_len);
        std::memcpy(buf.data(), &header, header_len);
        auto span = std::span<std::byte>(buf).subspan(header_len, t_size);
        postcard_slice_t slice;
        postcard_init_slice(&slice, reinterpret_cast<uint8_t*>(span.data()), span.size());
        auto res = payload.encode_raw(&slice);
        if (res == POSTCARD_SUCCESS) {
            buf.resize(slice.len + header_len);
        } else {
            buf.clear();
        }

        return buf;
    }
};

#endif
