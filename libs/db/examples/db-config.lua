local entity_ids = { 1 }

local time_id = ComponentId("time")
local mag_id = ComponentId("mag")
local gyro_id = ComponentId("gyro")
local accel_id = ComponentId("accel")
local temp_id = ComponentId("temp")
local pressure_id = ComponentId("pressure")
local humidity_id = ComponentId("humidity")

local sensor_vt = VTableBuilder(1)
sensor_vt:column(time_id, "I64", {}, entity_ids)
sensor_vt:column(mag_id, "F32", { 3 }, entity_ids)
sensor_vt:column(gyro_id, "F32", { 3 }, entity_ids)
sensor_vt:column(accel_id, "F32", { 3 }, entity_ids)
sensor_vt:column(temp_id, "F32", {}, entity_ids)
sensor_vt:column(pressure_id, "F32", {}, entity_ids)
sensor_vt:column(humidity_id, "F32", {}, entity_ids)

msgs = {
	sensor_vt:msg(),
	SetEntityMetadata({ entity_id = entity_ids[1], name = "Vehicle" }):msg(),
	SetComponentMetadata({ component_id = time_id, name = "time", metadata = { priority = "100" } }):msg(),
	SetComponentMetadata({ component_id = mag_id, name = "mag", metadata = { priority = "99" } }):msg(),
	SetComponentMetadata({ component_id = gyro_id, name = "gyro", metadata = { priority = "98" } }):msg(),
	SetComponentMetadata({ component_id = accel_id, name = "accel", metadata = { priority = "97" } }):msg(),
	SetComponentMetadata({ component_id = temp_id, name = "temp", metadata = { priority = "96" } }):msg(),
	SetComponentMetadata({ component_id = pressure_id, name = "pressure", metadata = { priority = "95" } }):msg(),
	SetComponentMetadata({ component_id = humidity_id, name = "humidity", metadata = { priority = "94" } }):msg(),
}

client = connect("127.0.0.1:2240")
client:send_msgs(msgs)
