//! Integration tests for the Foxglove-compatible MCAP export.
//!
//! Builds a small database (pose component + scalar + vector + msg log +
//! schematic asset), exports it, and re-reads the MCAP with the `mcap` crate
//! to verify channels, schemas, message ordering, attachments, and the
//! generated Foxglove layout.

use std::collections::HashMap;
use std::path::PathBuf;

use elodin_db::export_mcap::{McapExportOptions, run};
use elodin_db::{ComponentSchema, DB};
use impeller2::types::{ComponentId, PrimType, Timestamp, msg_id};
use impeller2_wkt::{ComponentMetadata, MsgMetadata, log_entry_msg_schema};

const TS_BASE: i64 = 1_700_000_000_000_000; // µs epoch
const TS_STEP: i64 = 10_000; // 100 Hz
const NUM_ROWS: usize = 25;

fn tmp_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("elodin_mcap_{label}_{}", fastrand::u64(..)));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn build_fixture(path: PathBuf) -> DB {
    let db = DB::create(path.clone()).expect("DB::create");

    let specs: &[(&str, PrimType, &[usize], Option<&str>)] = &[
        (
            "drone.world_pos",
            PrimType::F64,
            &[7],
            Some("q0,q1,q2,q3,x,y,z"),
        ),
        ("drone.gyro", PrimType::F64, &[3], Some("x,y,z")),
        ("drone.thrust", PrimType::F64, &[], None),
        ("Globals.tick", PrimType::U64, &[], None),
    ];

    db.with_state_mut(|s| {
        for (name, prim, dim, element_names) in specs {
            let cid = ComponentId::new(name);
            let mut metadata = HashMap::new();
            if let Some(en) = element_names {
                metadata.insert("element_names".to_string(), en.to_string());
            }
            s.set_component_metadata(
                ComponentMetadata {
                    component_id: cid,
                    name: name.to_string(),
                    metadata,
                },
                &path,
            )
            .expect("set_component_metadata");
            s.insert_component(cid, ComponentSchema::new(*prim, dim), &path)
                .expect("insert_component");
        }
    });

    for step in 0..NUM_ROWS {
        let ts = Timestamp(TS_BASE + TS_STEP * step as i64);
        db.with_state(|s| {
            let t = step as f64;
            let pose: Vec<u8> = [0.0, 0.0, 0.0, 1.0, t * 0.1, t * 0.2, 2.0]
                .iter()
                .flat_map(|v: &f64| v.to_le_bytes())
                .collect();
            let gyro: Vec<u8> = [t, -t, 0.5 * t]
                .iter()
                .flat_map(|v: &f64| v.to_le_bytes())
                .collect();
            let thrust = (t * 0.25f64).to_le_bytes().to_vec();
            let tick = (step as u64).to_le_bytes().to_vec();
            for (name, buf) in [
                ("drone.world_pos", pose),
                ("drone.gyro", gyro),
                ("drone.thrust", thrust),
                ("Globals.tick", tick),
            ] {
                let c = s.get_component(ComponentId::new(name)).expect("component");
                c.time_series.push_buf(ts, &buf).expect("push_buf");
            }
        });
    }

    // A LogEntry message log.
    let log_id = msg_id("fsw.log");
    db.with_state_mut(|s| {
        s.set_msg_metadata(
            log_id,
            MsgMetadata {
                name: "fsw.log".to_string(),
                schema: log_entry_msg_schema(),
                metadata: Default::default(),
            },
            &path,
        )
        .expect("set_msg_metadata");
    });
    for step in 0..3usize {
        let entry = impeller2_wkt::LogEntry {
            level: 2,
            message: format!("log line {step}"),
        };
        let bytes = postcard::to_allocvec(&entry).expect("postcard");
        db.push_msg(Timestamp(TS_BASE + TS_STEP * step as i64), log_id, &bytes)
            .expect("push_msg");
    }

    // Minimal schematic + GLB asset so scene/layout generation kicks in.
    let assets = path.join("assets");
    std::fs::create_dir_all(assets.join("schematics")).expect("mkdir assets");
    // Tiny valid-enough GLB payload (magic + version + length header only).
    let mut glb = b"glTF".to_vec();
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&12u32.to_le_bytes());
    std::fs::write(assets.join("drone.glb"), &glb).expect("write glb");
    std::fs::write(
        assets.join("schematics/main.kdl"),
        r#"tabs {
    hsplit name=Viewport {
        viewport name=Viewport pos="drone.world_pos + (0,0,0,0, 2,2,2)" look_at=drone.world_pos show_grid=#true active=#true
        vsplit share=0.4 {
            graph drone.gyro name=Gyro
            graph "drone.world_pos.q0, drone.thrust"
        }
    }
    vsplit name="Monitors" {
        component_monitor component_name=drone.gyro
    }
}
vector_arrow "(1, 0, 0)" origin=drone.world_pos name="Drone X" body_frame=#true {
    color white
}
object_3d drone.world_pos {
    glb path=db:drone.glb
}"#,
    )
    .expect("write schematic");
    db.set_active_schematic("schematics/main.kdl")
        .expect("set_active_schematic");

    db.flush_all().expect("flush_all");
    db
}

#[test]
fn mcap_export_roundtrip() {
    let db_path = tmp_dir("db");
    let out = tmp_dir("out");
    let db = build_fixture(db_path.clone());
    drop(db);

    run(db_path.clone(), out.clone(), McapExportOptions::default()).expect("mcap export");

    let db_name = db_path.file_name().unwrap().to_str().unwrap();
    let mcap_path = out.join(format!("{db_name}.mcap"));
    let layout_path = out.join(format!("{db_name}.foxglove-layout.json"));
    let mapped = std::fs::read(&mcap_path).expect("read mcap");

    // --- summary / channels ---
    let summary = mcap::Summary::read(&mapped)
        .expect("summary parse")
        .expect("summary present");
    let topics: std::collections::HashSet<String> =
        summary.channels.values().map(|c| c.topic.clone()).collect();
    for expected in [
        "/drone/world_pos",
        "/drone/gyro",
        "/drone/thrust",
        "/Globals/tick",
        "/tf",
        "/scene/drone-model",
        "/scene/drone-arrows",
        "/log/fsw.log",
    ] {
        assert!(
            topics.contains(expected),
            "missing topic {expected}: {topics:?}"
        );
    }

    let tf_channel = summary
        .channels
        .values()
        .find(|c| c.topic == "/tf")
        .unwrap();
    assert_eq!(
        tf_channel.schema.as_ref().unwrap().name,
        "foxglove.FrameTransforms"
    );

    // The scene schema must be the full official foxglove.SceneUpdate schema:
    // Foxglove's JSON deserializer only base64-decodes bytes fields that the
    // channel schema declares with `contentEncoding: "base64"`. A trimmed-down
    // schema silently leaves `models[].data` as a string and GLBs fail to load.
    let scene_channel = summary
        .channels
        .values()
        .find(|c| c.topic == "/scene/drone-model")
        .unwrap();
    let scene_schema: serde_json::Value =
        serde_json::from_slice(&scene_channel.schema.as_ref().unwrap().data).unwrap();
    assert_eq!(
        scene_schema["properties"]["entities"]["items"]["properties"]["models"]["items"]["properties"]
            ["data"]["contentEncoding"],
        "base64",
        "SceneUpdate schema must declare models[].data as base64 bytes"
    );
    assert_eq!(tf_channel.message_encoding, "json");
    let log_channel = summary
        .channels
        .values()
        .find(|c| c.topic == "/log/fsw.log")
        .unwrap();
    assert_eq!(log_channel.schema.as_ref().unwrap().name, "foxglove.Log");

    // --- messages: counts, monotonic log_time, JSON content ---
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut last_log_time = 0u64;
    let mut world_pos_first: Option<serde_json::Value> = None;
    let mut tf_first: Option<serde_json::Value> = None;
    for message in mcap::MessageStream::new(&mapped).expect("stream") {
        let message = message.expect("message");
        assert!(
            message.log_time >= last_log_time,
            "log_time went backwards on {}",
            message.channel.topic
        );
        last_log_time = message.log_time;
        *counts.entry(message.channel.topic.clone()).or_default() += 1;
        if message.channel.topic == "/drone/world_pos" && world_pos_first.is_none() {
            world_pos_first = Some(serde_json::from_slice(&message.data).unwrap());
        }
        if message.channel.topic == "/tf" && tf_first.is_none() {
            tf_first = Some(serde_json::from_slice(&message.data).unwrap());
        }
    }
    assert_eq!(counts["/drone/world_pos"], NUM_ROWS);
    assert_eq!(counts["/tf"], NUM_ROWS);
    assert_eq!(counts["/drone/thrust"], NUM_ROWS);
    assert_eq!(counts["/log/fsw.log"], 3);
    // Per-entity scene topics: exactly one message each.
    assert_eq!(counts["/scene/drone-model"], 1);
    assert_eq!(counts["/scene/drone-arrows"], 1);

    let world_pos = world_pos_first.unwrap();
    assert_eq!(world_pos["q3"], 1.0);
    assert_eq!(world_pos["z"], 2.0);
    let tf = tf_first.unwrap();
    assert_eq!(tf["transforms"][0]["child_frame_id"], "drone");
    assert_eq!(tf["transforms"][0]["translation"]["z"], 2.0);
    assert_eq!(tf["transforms"][0]["rotation"]["w"], 1.0);

    // --- attachments: schematic + referenced GLB ---
    let attachment_names: Vec<String> = summary
        .attachment_indexes
        .iter()
        .map(|a| a.name.clone())
        .collect();
    assert!(attachment_names.contains(&"schematics/main.kdl".to_string()));
    assert!(attachment_names.contains(&"drone.glb".to_string()));

    // --- metadata records ---
    let metadata_names: Vec<String> = summary
        .metadata_indexes
        .iter()
        .map(|m| m.name.clone())
        .collect();
    assert!(metadata_names.contains(&"elodin.db_state".to_string()));
    assert!(metadata_names.contains(&"elodin.components".to_string()));

    // --- layout ---
    let layout: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&layout_path).expect("layout file")).unwrap();
    let config_by_id = layout["configById"].as_object().unwrap();
    let root = layout["layout"].as_str().unwrap();
    assert!(root.starts_with("Tab!"));
    let tabs = config_by_id[root]["tabs"].as_array().unwrap();
    let titles: Vec<&str> = tabs.iter().map(|t| t["title"].as_str().unwrap()).collect();
    assert_eq!(titles, ["Viewport", "Monitors"]);

    // A 3D panel following the drone, with every scene topic enabled.
    let three_d = config_by_id
        .values()
        .find(|v| v.get("followTf").is_some())
        .expect("3D panel config");
    assert_eq!(three_d["followTf"], "drone");
    assert_eq!(three_d["topics"]["/scene/drone-model"]["visible"], true);
    assert_eq!(three_d["topics"]["/scene/drone-arrows"]["visible"], true);
    // Camera derived from the viewport pos offset (2,2,2), in *degrees* —
    // Foxglove's 3D panel reads phi/thetaOffset/fovy as degrees; radians here
    // produce a near-top-down view (phi ~1 deg).
    let camera = &three_d["cameraState"];
    assert!((camera["distance"].as_f64().unwrap() - 12.0f64.sqrt()).abs() < 1e-6);
    assert!((camera["phi"].as_f64().unwrap() - 54.7356).abs() < 1e-3);
    assert!((camera["thetaOffset"].as_f64().unwrap() - 45.0).abs() < 1e-6);
    assert_eq!(camera["fovy"], 45.0);

    // Plot series resolved from EQL, including explicit element access.
    let plots: Vec<&serde_json::Value> = config_by_id
        .values()
        .filter(|v| v.get("paths").is_some())
        .collect();
    assert_eq!(plots.len(), 2);
    let all_paths: Vec<String> = plots
        .iter()
        .flat_map(|p| p["paths"].as_array().unwrap())
        .map(|s| s["value"].as_str().unwrap().to_string())
        .collect();
    assert!(all_paths.contains(&"/drone/gyro.x".to_string()));
    assert!(all_paths.contains(&"/drone/world_pos.q0".to_string()));
    assert!(all_paths.contains(&"/drone/thrust.value".to_string()));

    // Raw messages panel on the monitored component.
    let raw = config_by_id
        .values()
        .find(|v| v.get("topicPath").is_some())
        .expect("RawMessages config");
    assert_eq!(raw["topicPath"], "/drone/gyro");

    let _ = std::fs::remove_dir_all(&db_path);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn mcap_export_pattern_filters_components() {
    let db_path = tmp_dir("db_filter");
    let out = tmp_dir("out_filter");
    let db = build_fixture(db_path.clone());
    drop(db);

    run(
        db_path.clone(),
        out.clone(),
        McapExportOptions {
            pattern: Some("drone.*".to_string()),
            ..Default::default()
        },
    )
    .expect("mcap export");

    let db_name = db_path.file_name().unwrap().to_str().unwrap();
    let mapped = std::fs::read(out.join(format!("{db_name}.mcap"))).expect("read mcap");
    let summary = mcap::Summary::read(&mapped).unwrap().unwrap();
    let topics: Vec<String> = summary.channels.values().map(|c| c.topic.clone()).collect();
    assert!(topics.iter().any(|t| t == "/drone/gyro"));
    assert!(!topics.iter().any(|t| t == "/Globals/tick"));

    let _ = std::fs::remove_dir_all(&db_path);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn mcap_export_epoch_offset_auto() {
    let db_path = tmp_dir("db_offset_auto");
    let out = tmp_dir("out_offset_auto");
    let db = DB::create(db_path.clone()).expect("DB::create");

    let specs: &[(&str, PrimType, &[usize], Option<&str>)] = &[
        ("sat.alt", PrimType::F64, &[], None),
    ];

    db.with_state_mut(|s| {
        for (name, prim, dim, element_names) in specs {
            let cid = ComponentId::new(name);
            let mut metadata = HashMap::new();
            if let Some(en) = element_names {
                metadata.insert("element_names".to_string(), en.to_string());
            }
            s.set_component_metadata(
                ComponentMetadata {
                    component_id: cid,
                    name: name.to_string(),
                    metadata,
                },
                &db_path,
            )
            .expect("set_component_metadata");
            s.insert_component(cid, ComponentSchema::new(*prim, dim), &db_path)
                .expect("insert_component");
        }
    });

    for step in 0..10 {
        let ts = Timestamp(-100_000 + step as i64 * 10_000);
        db.with_state(|s| {
            let val = (step as f64 * 100.0).to_le_bytes().to_vec();
            let c = s.get_component(ComponentId::new("sat.alt")).expect("component");
            c.time_series.push_buf(ts, &val).expect("push_buf");
        });
    }
    db.flush_all().expect("flush_all");
    drop(db);

    run(db_path.clone(), out.clone(), McapExportOptions::default()).expect("mcap export");

    let db_name = db_path.file_name().unwrap().to_str().unwrap();
    let mapped = std::fs::read(out.join(format!("{db_name}.mcap"))).expect("read mcap");

    let mut min_log_time = u64::MAX;
    for message in mcap::MessageStream::new(&mapped).expect("stream") {
        let message = message.expect("message");
        min_log_time = min_log_time.min(message.log_time);
    }
    assert_eq!(min_log_time, 0, "auto-rebased earliest should be 0 ns");

    let summary = mcap::Summary::read(&mapped).unwrap().unwrap();
    let db_state_meta = summary.metadata_indexes.iter().find(|m| m.name == "elodin.db_state");
    assert!(db_state_meta.is_some(), "db_state metadata should exist");

    let _ = std::fs::remove_dir_all(&db_path);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn mcap_export_epoch_offset_manual() {
    let db_path = tmp_dir("db_offset_manual");
    let out = tmp_dir("out_offset_manual");
    let db = DB::create(db_path.clone()).expect("DB::create");

    let specs: &[(&str, PrimType, &[usize], Option<&str>)] = &[
        ("sat.alt", PrimType::F64, &[], None),
    ];

    db.with_state_mut(|s| {
        for (name, prim, dim, element_names) in specs {
            let cid = ComponentId::new(name);
            let mut metadata = HashMap::new();
            if let Some(en) = element_names {
                metadata.insert("element_names".to_string(), en.to_string());
            }
            s.set_component_metadata(
                ComponentMetadata {
                    component_id: cid,
                    name: name.to_string(),
                    metadata,
                },
                &db_path,
            )
            .expect("set_component_metadata");
            s.insert_component(cid, ComponentSchema::new(*prim, dim), &db_path)
                .expect("insert_component");
        }
    });

    for step in 0..5 {
        let ts = Timestamp(1000 + step as i64 * 1000);
        db.with_state(|s| {
            let val = (step as f64).to_le_bytes().to_vec();
            let c = s.get_component(ComponentId::new("sat.alt")).expect("component");
            c.time_series.push_buf(ts, &val).expect("push_buf");
        });
    }
    db.flush_all().expect("flush_all");
    drop(db);

    let manual_offset: i64 = 500_000;
    run(
        db_path.clone(),
        out.clone(),
        McapExportOptions {
            epoch_offset_us: Some(manual_offset),
            ..Default::default()
        },
    )
    .expect("mcap export");

    let db_name = db_path.file_name().unwrap().to_str().unwrap();
    let mapped = std::fs::read(out.join(format!("{db_name}.mcap"))).expect("read mcap");

    let mut first_log_time = None;
    for message in mcap::MessageStream::new(&mapped).expect("stream") {
        let message = message.expect("message");
        if first_log_time.is_none() {
            first_log_time = Some(message.log_time);
        }
    }
    let expected_ns = (1000i64 + manual_offset) as u64 * 1000;
    assert_eq!(first_log_time.unwrap(), expected_ns);

    let _ = std::fs::remove_dir_all(&db_path);
    let _ = std::fs::remove_dir_all(&out);
}
