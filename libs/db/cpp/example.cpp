#include "db.hpp"
#include "vtable.hpp"
#include <cstdio>

using namespace vtable;

int main() {
    auto table = builder::vtable({
        builder::raw_field(0, 8, builder::schema(PrimType::F64(), {}, builder::pair(1, "time"))),
        builder::raw_field(8, 16, builder::schema(PrimType::F64(), {}, builder::pair(1, "mag")))
    });
    auto vtable_msg = VTableMsg {
        .id = {1,0},
        .vtable = table,
    };
    auto msg = Msg<VTableMsg>(vtable_msg);
    auto out = msg.encode_vec();
    std::println("vtable msg {}", out);
    return 0;
}
