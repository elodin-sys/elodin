#include "db.hpp"
#include "vtable.hpp"
#include <cstdio>

using namespace vtable;

struct Foo {
    double time;
    double mag;
};

int main() {
    auto table = builder::vtable({
        builder::field<Foo, &Foo::time>(builder::schema(PrimType::F64(), {}, builder::pair(1, "time"))),
        builder::field<Foo, &Foo::mag>(builder::schema(PrimType::F64(), {}, builder::pair(1, "mag")))
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
