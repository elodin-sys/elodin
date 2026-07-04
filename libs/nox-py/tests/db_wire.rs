//! Reproduces the TableWriter wire sequence (metadata, vtable, table) over a
//! real TCP connection to an embedded server, mirroring writer.rs exactly.

use impeller2::types::{ComponentId, LenPacket, PrimType};
use impeller2::vtable::builder::{component, raw_field, raw_table, schema, timestamp, vtable};
use impeller2_stellar::Client;
use impeller2_wkt::{SetComponentMetadata, VTableMsg};

#[test]
fn wire_two_field_write() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .try_init();
    let dir = tempfile::tempdir().unwrap();
    let addr: std::net::SocketAddr = "127.0.0.1:22499".parse().unwrap();
    let server = elodin_db::Server::new(dir.path().join("db"), addr).unwrap();
    let db = server.db.clone();
    let _server_thread = stellarator::struc_con::stellar(move || server.run());
    std::thread::sleep(std::time::Duration::from_millis(300));

    let db_in = db.clone();
    stellarator::run(move || async move {
        let db = db_in;
        let mut c = Client::connect(addr).await.unwrap();
        for name in ["t.a", "t.b"] {
            let msg = SetComponentMetadata::new(ComponentId::new(name), name);
            let (r, _) = c.send(&msg).await;
            r.unwrap();
        }
        let time_field = raw_table(0, 8);
        let specs = [("t.a", 8usize, 24usize, vec![3u64]), ("t.b", 32, 16, vec![2])];
        let vtable_def = vtable(specs.iter().map(|(name, offset, size, dims)| {
            raw_field(
                *offset as u16,
                *size as u16,
                schema(
                    PrimType::F64,
                    dims,
                    timestamp(time_field.clone(), component(ComponentId::new(*name))),
                ),
            )
        }));
        let vtable_msg = VTableMsg {
            id: [77, 1],
            vtable: vtable_def,
        };
        let bytes = postcard::to_allocvec(&vtable_msg).unwrap();
        println!("allocvec bytes: {bytes:?}");
        let reparsed: VTableMsg = postcard::from_bytes(&bytes).unwrap();
        println!("direct reinsert: {:?}", db.insert_vtable(reparsed).map(|_| "ok"));
        let (r, _) = c.send(&vtable_msg).await;
        r.unwrap();

        let mut row = Vec::with_capacity(48);
        row.extend_from_slice(&1_000_000i64.to_le_bytes());
        for v in [1.0f64, 2.0, 3.0, 4.0, 5.0] {
            row.extend_from_slice(&v.to_le_bytes());
        }
        let mut packet = LenPacket::table([77, 1], row.len());
        packet.extend_from_slice(&row);
        let (r, _) = c.send(packet).await;
        r.unwrap();
        stellarator::sleep(std::time::Duration::from_millis(500)).await;
    });

    let ok = db.with_state(|state| {
        state.get_component(ComponentId::new("t.a")).is_some()
            && state.get_component(ComponentId::new("t.b")).is_some()
    });
    assert!(ok, "both components should have time series after the write");
}
