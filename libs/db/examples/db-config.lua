local entity_ids = { 1 }

local time_id = ComponentId("time")
local mag_id = ComponentId("mag")
local gyro_id = ComponentId("gyro")
local accel_id = ComponentId("accel")
local temp_id = ComponentId("temp")
local pressure_id = ComponentId("pressure")
local humidity_id = ComponentId("humidity")

local sensor_vt = VTableBuilder(1)
sensor_vt:column(time_id, "i64", {}, entity_ids)
sensor_vt:column(mag_id, "f32", { 3 }, entity_ids)
sensor_vt:column(gyro_id, "f32", { 3 }, entity_ids)
sensor_vt:column(accel_id, "f32", { 3 }, entity_ids)
sensor_vt:column(temp_id, "f32", {}, entity_ids)
sensor_vt:column(pressure_id, "f32", {}, entity_ids)
sensor_vt:column(humidity_id, "f32", {}, entity_ids)

msgs = {
	SetEntityMetadata({ entity_id = entity_ids[1], name = "Vehicle" }),
	SetComponentMetadata({ component_id = time_id, name = "time", metadata = { priority = "100" } }),
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
