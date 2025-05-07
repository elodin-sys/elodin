sensor_vt =  vtable_msg(1, {
    field(0, 8, schema("f64", {}, pair(1, "time"))),
    field(8, 16, schema("f64", {}, pair(1, "mag"))),
})
print(sensor_vt:msg())
