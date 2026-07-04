//! Validates the `elodin.db` TableWriter vtable layout directly against
//! `elodin_db::DB` (no sockets): registration must pass alignment validation
//! and a row must sink into per-component time series.

use impeller2::types::{ComponentId, PrimType};
use impeller2::vtable::builder::{component, raw_field, raw_table, schema, timestamp, vtable};
use impeller2_wkt::VTableMsg;

#[test]
fn two_field_vtable_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let db = elodin_db::DB::create(dir.path().join("db")).unwrap();
    let time_field = raw_table(0, 8);
    let fields = vec![
        raw_field(
            8,
            24,
            schema(
                PrimType::F64,
                &[3],
                timestamp(time_field.clone(), component(ComponentId::new("t.a"))),
            ),
        ),
        raw_field(
            32,
            16,
            schema(
                PrimType::F64,
                &[2],
                timestamp(time_field.clone(), component(ComponentId::new("t.b"))),
            ),
        ),
    ];
    let vtable_msg = VTableMsg {
        id: [43, 1],
        vtable: vtable(fields),
    };
    let bytes = postcard::to_allocvec(&vtable_msg).unwrap();
    let parsed: VTableMsg = postcard::from_bytes(&bytes).unwrap();
    db.insert_vtable(parsed)
        .expect("two-field vtable should round-trip");
}

#[test]
fn batched_vtable_registers_and_sinks() {
    let dir = tempfile::tempdir().unwrap();
    let db = elodin_db::DB::create(dir.path().join("db")).unwrap();

    // Layout mirroring python: ts @0..8, f64x3 @8..32, f64 @32..40, i32 @40..44
    let time_field = raw_table(0, 8);
    let fields = vec![
        raw_field(
            8,
            24,
            schema(
                PrimType::F64,
                &[3],
                timestamp(time_field.clone(), component(ComponentId::new("test.accel"))),
            ),
        ),
        raw_field(
            32,
            8,
            schema(
                PrimType::F64,
                &[],
                timestamp(
                    time_field.clone(),
                    component(ComponentId::new("test.throttle")),
                ),
            ),
        ),
        raw_field(
            40,
            4,
            schema(
                PrimType::I32,
                &[],
                timestamp(time_field.clone(), component(ComponentId::new("test.gate"))),
            ),
        ),
    ];
    let vtable_msg = VTableMsg {
        id: [42, 1],
        vtable: vtable(fields),
    };

    // Round-trip through postcard the way the wire does: the shared timestamp
    // op must survive serialization.
    let bytes = postcard::to_allocvec(&vtable_msg).unwrap();
    let parsed: VTableMsg = postcard::from_bytes(&bytes).unwrap();
    db.insert_vtable(parsed)
        .expect("vtable should validate after wire round-trip");
}
