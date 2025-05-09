local mag_id = ComponentId("mag")
local gyro_id = ComponentId("gyro")
local accel_id = ComponentId("accel")
local temp_id = ComponentId("temp")
local pressure_id = ComponentId("pressure")
local humidity_id = ComponentId("humidity")

sensor_vt =  vtable_msg(1, {
    field(8, 12, schema("f32", {3}, pair(1, "mag"))),
    field(20, 12, schema("f32", {3}, pair(1, "gyro"))),
    field(32, 12, schema("f32", {3}, pair(1, "accel"))),
    field(44, 4, schema("f32", {}, pair(1, "temp"))),
    field(48, 4, schema("f32", {}, pair(1, "pressure"))),
    field(52, 4, schema("f32", {}, pair(1, "humidity"))),
})

msgs = {
	SetEntityMetadata({ entity_id = 1, name = "Vehicle" }),
	SetComponentMetadata({ component_id = mag_id, name = "mag", metadata = { priority = "99" } }),
	SetComponentMetadata({ component_id = gyro_id, name = "gyro", metadata = { priority = "98" } }),
	SetComponentMetadata({ component_id = accel_id, name = "accel", metadata = { priority = "97" } }),
	SetComponentMetadata({ component_id = temp_id, name = "temp", metadata = { priority = "96" } }),
	SetComponentMetadata({ component_id = pressure_id, name = "pressure", metadata = { priority = "95" } }),
	SetComponentMetadata({ component_id = humidity_id, name = "humidity", metadata = { priority = "94" } }),
	sensor_vt,
}

client = connect("127.0.0.1:2240")
client:send_msgs(msgs)
