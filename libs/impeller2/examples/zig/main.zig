const vtable = @embedFile("./vtable.bin");
const std = @import("std");
const net = std.net;

const PacketTy = enum(u8) { msg, table, timeSeries };
const PacketHeader = extern struct {
    len: u64,
    ty: PacketTy,
    packetId: [3]u8,
    reqId: u32 = 0,
};
const MsgId = enum([3]u8) {
    vtable = [_]u8{ 224, 0, 0 },
};

pub fn main() !void {
    const addr = try net.Address.parseIp4("127.0.0.1", 2240);
    const stream = try net.tcpConnectToAddress(addr);
    defer stream.close();
    var writer = stream.writer();
    const vtableHeader = PacketHeader{ .len = vtable.len + 8, .ty = PacketTy.msg, .packetId = .{ 224, 0, 0 } };
    try writer.writeAll(std.mem.asBytes(&vtableHeader));
    try writer.writeAll(vtable);
    var val: f64 = 1.0;
    while (true) {
        const sin = std.math.sin(val);
        const bytes = std.mem.asBytes(&sin);
        const tableHeader = PacketHeader{ .len = 8 + 8, .ty = PacketTy.table, .packetId = .{ 1, 0, 0 } };
        try writer.writeAll(std.mem.asBytes(&tableHeader));
        try writer.writeAll(bytes);
        val += 0.000001;
    }
}
