local remote_addr = os.getenv("FC_ADDR") or "127.0.0.1:2240"
local ground_station_addr = os.getenv("GROUND_STATION_ADDR") or "127.0.0.1:2241"

local remote_client = connect(remote_addr)
local local_client = connect(ground_station_addr)

local metadata_dump = remote_client:dump_metadata()
local metadata_msgs = {}
for _, entity_metadata in pairs(metadata_dump["entity_metadata"]) do
	table.insert(metadata_msgs, SetEntityMetadata(entity_metadata))
end
for _, component_metadata in pairs(metadata_dump["component_metadata"]) do
	table.insert(metadata_msgs, SetComponentMetadata(component_metadata))
end
local_client:send_msgs(metadata_msgs)
remote_client:send_msg(UdpUnicast({ stream = { id = 2 }, addr = ground_station_addr }))
