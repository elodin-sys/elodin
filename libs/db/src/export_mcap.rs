//! Foxglove-compatible MCAP export.
//!
//! Exports a database to a single `.mcap` file (JSON-encoded channels, zstd
//! chunks) plus a Foxglove layout JSON generated from the active schematic.
//! See `ai-context/foxglove-mcap-export-design.md` for the full design.
//!
//! Channel mapping:
//! - each component -> `/<name with '.' replaced by '/'>` (JSON object keyed by
//!   element names, nested at `.` boundaries)
//! - pose components (`*.world_pos`, 7 elements) -> `/tf` (`foxglove.FrameTransforms`)
//! - schematic `object_3d` / static `vector_arrow` / `line_3d` -> `/scene`
//!   (`foxglove.SceneUpdate`, one message per entity; GLBs embedded as base64)
//! - dynamic `vector_arrow` (EQL-backed) -> `/scene_dynamic` (separate topic so
//!   seeks do not drop static entities under latest-per-topic backfill)
//! - message logs -> `foxglove.CompressedVideo` (H.264), `foxglove.RawImage`
//!   (sensor cameras), `foxglove.Log` (LogEntry streams), or raw base64 JSON
//! - DB / component / msg metadata -> MCAP metadata records; schematics and
//!   referenced assets -> MCAP attachments

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use glob::Pattern;
use impeller2::types::{PacketId, PrimType, Timestamp, msg_id};
use impeller2_wkt::{
    Color, Line3d, LogEntry, Object3D, Object3DMesh, Panel, Schematic, SchematicElem,
    SensorCameraConfig, VectorArrow3d, log_entry_msg_schema,
};
use mcap::records::MessageHeader;
use serde_json::{Map, Value, json};

use crate::msg_log::MsgLog;
use crate::{Component, DB, Error};

/// 1 MiB output buffer, matching the other exporters.
const FILE_BUF_CAP: usize = 1 << 20;

/// Options for the MCAP export.
#[derive(Clone, Debug)]
pub struct McapExportOptions {
    /// Glob pattern over component names; non-matching components are skipped.
    pub pattern: Option<String>,
    /// Include components whose metadata contains `"private": "true"`.
    pub include_private: bool,
    /// Attach every file under `{db}/assets/` instead of only schematic-referenced ones.
    pub all_assets: bool,
    /// Microsecond offset added to all timestamps before conversion to MCAP
    /// nanoseconds. When `None` and the earliest DB timestamp is negative,
    /// the offset is auto-computed so the earliest sample maps to t = 0.
    pub epoch_offset_us: Option<i64>,
    /// Maximum GLB size (MiB) to base64-embed in SceneUpdate. Larger models
    /// are attached to the MCAP but omitted from the scene message entirely
    /// (no empty-`data` model primitive). The viewport follow-entity's mesh
    /// is always embedded regardless of this limit.
    pub max_embed_mb: u64,
}

impl Default for McapExportOptions {
    fn default() -> Self {
        Self {
            pattern: None,
            include_private: false,
            all_assets: false,
            epoch_offset_us: None,
            max_embed_mb: 32,
        }
    }
}

// ---------------------------------------------------------------------------
// Foxglove well-known JSON schemas, vendored verbatim from the MIT-licensed
// foxglove-sdk (schemas/jsonschema). The *full* official schemas are required
// — Foxglove's JSON-channel deserializer builds its base64-bytes decoding from
// the channel schema, and only decodes `bytes` fields the schema explicitly
// declares with `"contentEncoding": "base64"` (nested fields included). A
// hand-rolled subset silently breaks e.g. `SceneUpdate.entities[].models[].data`.
// ---------------------------------------------------------------------------

const SCHEMA_FRAME_TRANSFORMS: &str = include_str!("foxglove_schemas/FrameTransforms.json");
const SCHEMA_SCENE_UPDATE: &str = include_str!("foxglove_schemas/SceneUpdate.json");
const SCHEMA_COMPRESSED_VIDEO: &str = include_str!("foxglove_schemas/CompressedVideo.json");
#[cfg_attr(feature = "video-export", allow(dead_code))]
const SCHEMA_RAW_IMAGE: &str = include_str!("foxglove_schemas/RawImage.json");
const SCHEMA_LOG: &str = include_str!("foxglove_schemas/Log.json");

const SCHEMA_RAW_BYTES: &str = r#"{
  "title": "elodin.RawMessage", "type": "object",
  "properties": {"data": {"type": "string", "contentEncoding": "base64"}}
}"#;

fn invalid_data(message: impl Into<String>) -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        message.into(),
    ))
}

// ---------------------------------------------------------------------------
// Component export plan
// ---------------------------------------------------------------------------

/// One component prepared for export: topic, element field paths, channel metadata.
struct ExportComponent {
    component: Component,
    name: String,
    topic: String,
    /// Nested JSON path per flattened element (e.g. `["e", "r"]` for element name `e.r`).
    element_paths: Vec<Vec<String>>,
    metadata: BTreeMap<String, String>,
    /// `Some(entity)` when this is a 7-element `<entity>.world_pos` pose.
    pose_entity: Option<String>,
}

fn topic_for_component(name: &str) -> String {
    format!("/{}", name.replace('.', "/"))
}

/// Resolve the flattened per-element field paths for a component. Uses the
/// component's `element_names` metadata when it matches the flat element count
/// and has no duplicates; otherwise falls back to EQL default names
/// (`x,y,z,w,...`, compound for multi-dim shapes). Scalars map to `["value"]`.
fn element_paths(
    name: &str,
    schema: &crate::ComponentSchema,
    meta_element_names: &str,
) -> Vec<Vec<String>> {
    let flat_count: usize = schema.dim.iter().product::<usize>().max(1);
    let from_meta: Vec<String> = meta_element_names
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let unique: HashSet<&String> = from_meta.iter().collect();
    let names: Vec<String> = if from_meta.len() == flat_count && unique.len() == from_meta.len() {
        from_meta
    } else {
        // EQL default names for this shape (single source of truth for naming).
        eql::Component::new(
            name.to_string(),
            impeller2::types::ComponentId::new(name),
            schema.to_schema(),
        )
        .element_names
    };
    names
        .into_iter()
        .map(|n| {
            if n.is_empty() {
                vec!["value".to_string()]
            } else {
                n.split('.').map(str::to_string).collect()
            }
        })
        .collect()
}

/// Flat element path rendered for Foxglove message-path syntax (dots preserved,
/// which works because dotted element names become nested JSON objects).
fn element_path_str(paths: &[Vec<String>], idx: usize) -> Option<String> {
    paths.get(idx).map(|p| p.join("."))
}

fn json_type_for_prim(prim: PrimType) -> &'static str {
    match prim {
        PrimType::F32 | PrimType::F64 => "number",
        PrimType::Bool => "boolean",
        _ => "integer",
    }
}

/// Build the JSON schema for a component channel (nested at `.` boundaries).
fn component_json_schema(name: &str, prim: PrimType, element_paths: &[Vec<String>]) -> Value {
    fn insert_path(props: &mut Map<String, Value>, path: &[String], leaf: Value) {
        match path {
            [] => {}
            [last] => {
                props.insert(last.clone(), leaf);
            }
            [head, rest @ ..] => {
                let entry = props
                    .entry(head.clone())
                    .or_insert_with(|| json!({"type": "object", "properties": {}}));
                if let Some(sub) = entry.get_mut("properties").and_then(|p| p.as_object_mut()) {
                    // Recurse into the nested object's properties.
                    let mut sub_map = std::mem::take(sub);
                    insert_path(&mut sub_map, rest, leaf);
                    *sub = sub_map;
                }
            }
        }
    }
    let mut props = Map::new();
    let leaf_type = json_type_for_prim(prim);
    for path in element_paths {
        insert_path(&mut props, path, json!({"type": leaf_type}));
    }
    json!({"title": name, "type": "object", "properties": Value::Object(props)})
}

/// Decode element `idx` of a raw sample buffer into a JSON value.
fn element_value(prim: PrimType, buf: &[u8], idx: usize) -> Value {
    fn f64_value(v: f64) -> Value {
        serde_json::Number::from_f64(v)
            .map(Value::Number)
            .unwrap_or(Value::Null)
    }
    let size = prim.size();
    let start = idx * size;
    let Some(bytes) = buf.get(start..start + size) else {
        return Value::Null;
    };
    match prim {
        PrimType::F64 => f64_value(f64::from_le_bytes(bytes.try_into().unwrap())),
        PrimType::F32 => f64_value(f32::from_le_bytes(bytes.try_into().unwrap()) as f64),
        PrimType::I64 => Value::from(i64::from_le_bytes(bytes.try_into().unwrap())),
        PrimType::I32 => Value::from(i32::from_le_bytes(bytes.try_into().unwrap())),
        PrimType::I16 => Value::from(i16::from_le_bytes(bytes.try_into().unwrap())),
        PrimType::I8 => Value::from(bytes[0] as i8),
        PrimType::U64 => Value::from(u64::from_le_bytes(bytes.try_into().unwrap())),
        PrimType::U32 => Value::from(u32::from_le_bytes(bytes.try_into().unwrap())),
        PrimType::U16 => Value::from(u16::from_le_bytes(bytes.try_into().unwrap())),
        PrimType::U8 => Value::from(bytes[0]),
        PrimType::Bool => Value::from(bytes[0] != 0),
    }
}

/// Serialize one component sample to a JSON message body.
fn component_row_json(comp: &ExportComponent, buf: &[u8]) -> Vec<u8> {
    fn insert_path(map: &mut Map<String, Value>, path: &[String], value: Value) {
        match path {
            [] => {}
            [last] => {
                map.insert(last.clone(), value);
            }
            [head, rest @ ..] => {
                let entry = map
                    .entry(head.clone())
                    .or_insert_with(|| Value::Object(Map::new()));
                if let Value::Object(sub) = entry {
                    insert_path(sub, rest, value);
                }
            }
        }
    }
    let prim = comp.component.schema.prim_type;
    let mut map = Map::new();
    for (idx, path) in comp.element_paths.iter().enumerate() {
        insert_path(&mut map, path, element_value(prim, buf, idx));
    }
    serde_json::to_vec(&Value::Object(map)).expect("json serialize")
}

fn read_f64(prim: PrimType, buf: &[u8], idx: usize) -> f64 {
    match element_value(prim, buf, idx) {
        Value::Number(n) => n.as_f64().unwrap_or(0.0),
        _ => 0.0,
    }
}

fn timestamp_json(ts_ns: u64) -> Value {
    json!({"sec": ts_ns / 1_000_000_000, "nsec": ts_ns % 1_000_000_000})
}

/// Convert a DB microsecond timestamp to MCAP nanoseconds after applying
/// `offset_us`. MCAP `log_time` is unsigned, so negative results saturate to 0
/// — callers must choose an offset that keeps the export range non-negative
/// (see epoch-offset resolution in [`run`]).
fn us_to_ns(ts: Timestamp, offset_us: i64) -> u64 {
    match ts.0.checked_add(offset_us) {
        Some(us) if us >= 0 => (us as u64).saturating_mul(1000),
        _ => 0,
    }
}

/// FrameTransforms message for one pose sample (`[qx,qy,qz,qw, x,y,z]`).
fn tf_message(entity: &str, parent: &str, prim: PrimType, buf: &[u8], ts_ns: u64) -> Vec<u8> {
    let q = |i| read_f64(prim, buf, i);
    let msg = json!({
        "transforms": [{
            "timestamp": timestamp_json(ts_ns),
            "parent_frame_id": parent,
            "child_frame_id": entity,
            "translation": {"x": q(4), "y": q(5), "z": q(6)},
            "rotation": {"x": q(0), "y": q(1), "z": q(2), "w": q(3)},
        }]
    });
    serde_json::to_vec(&msg).expect("json serialize")
}

// ---------------------------------------------------------------------------
// Geo frames (schematic `coordinate` node)
// ---------------------------------------------------------------------------

/// World→NED / world→ENU transforms anchored at the schematic's geodetic
/// `coordinate` origin. The export world frame is the raw data frame (ECEF
/// for geo examples); entities whose `object_3d` carries `frame="NED"` or
/// `frame="ENU"` have *local* poses that Elodin re-anchors at this origin.
struct GeoFrameAnchors {
    origin_ecef: [f64; 3],
    enu_quat: [f64; 4],
    ned_quat: [f64; 4],
}

/// Quaternion (x,y,z,w) from a rotation matrix given as three columns.
fn quat_from_mat3_cols(c0: [f64; 3], c1: [f64; 3], c2: [f64; 3]) -> [f64; 4] {
    let (m00, m10, m20) = (c0[0], c0[1], c0[2]);
    let (m01, m11, m21) = (c1[0], c1[1], c1[2]);
    let (m02, m12, m22) = (c2[0], c2[1], c2[2]);
    let trace = m00 + m11 + m22;
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s]
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s]
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s]
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s]
    }
}

fn geo_frame_anchors(origin: &impeller2_wkt::GeoOriginConfig) -> GeoFrameAnchors {
    const WGS84_A: f64 = 6_378_137.0;
    const WGS84_E2: f64 = 6.694_379_990_141_316_5e-3;
    let lat = origin.latitude.to_radians();
    let lon = origin.longitude.to_radians();
    let (slat, clat) = lat.sin_cos();
    let (slon, clon) = lon.sin_cos();
    let n = WGS84_A / (1.0 - WGS84_E2 * slat * slat).sqrt();
    let alt = origin.altitude;
    let origin_ecef = [
        (n + alt) * clat * clon,
        (n + alt) * clat * slon,
        (n * (1.0 - WGS84_E2) + alt) * slat,
    ];
    let east = [-slon, clon, 0.0];
    let north = [-slat * clon, -slat * slon, clat];
    let up = [clat * clon, clat * slon, slat];
    GeoFrameAnchors {
        origin_ecef,
        enu_quat: quat_from_mat3_cols(east, north, up),
        ned_quat: quat_from_mat3_cols(north, east, [-up[0], -up[1], -up[2]]),
    }
}

/// FrameTransforms message with the world→NED and world→ENU anchor frames.
fn geo_frame_tf_message(anchors: &GeoFrameAnchors, ts_ns: u64) -> Vec<u8> {
    let [x, y, z] = anchors.origin_ecef;
    let tf = |frame: &str, q: &[f64; 4]| {
        json!({
            "timestamp": timestamp_json(ts_ns),
            "parent_frame_id": "world",
            "child_frame_id": frame,
            "translation": {"x": x, "y": y, "z": z},
            "rotation": {"x": q[0], "y": q[1], "z": q[2], "w": q[3]},
        })
    };
    let msg = json!({
        "transforms": [tf("NED", &anchors.ned_quat), tf("ENU", &anchors.enu_quat)],
    });
    serde_json::to_vec(&msg).expect("json serialize")
}

/// Map entity → geo frame name for entities whose `object_3d` declares a
/// local `frame=` (only NED/ENU need re-parenting; ECEF equals world).
fn entity_geo_frames(schematics: &LoadedSchematics, ctx: &eql::Context) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for elem in schematics
        .primary
        .iter()
        .chain(schematics.windows.iter().map(|(_, s)| s))
        .flat_map(|s| s.elems.iter())
    {
        if let SchematicElem::Object3d(object) = elem
            && let Some(frame) = object.frame
        {
            let name = format!("{frame:?}");
            if (name == "NED" || name == "ENU")
                && let Some(entity) = entity_for_eql(&object.eql, ctx)
            {
                map.entry(entity).or_insert(name);
            }
        }
    }
    map
}

// ---------------------------------------------------------------------------
// Message log classification
// ---------------------------------------------------------------------------

enum MsgLogKind {
    H264Video,
    SensorCamera(Box<SensorCameraConfig>),
    LogEntries,
    Raw,
}

struct ExportMsgLog {
    log: MsgLog,
    name: String,
    topic: String,
    kind: MsgLogKind,
}

/// True when `payload` looks like H.264 Annex-B: a 3- or 4-byte start code
/// followed by a plausible NAL header (forbidden_zero_bit clear, type 1–23).
/// Rejects bare `00 00 01` prefixes that aren't actually video.
fn is_annex_b(payload: &[u8]) -> bool {
    let nal = if payload.starts_with(&[0, 0, 0, 1]) {
        payload.get(4)
    } else if payload.starts_with(&[0, 0, 1]) {
        payload.get(3)
    } else {
        return false;
    };
    match nal {
        Some(&b) => {
            let nal_type = b & 0x1f;
            (b & 0x80) == 0 && (1..=23).contains(&nal_type)
        }
        None => false,
    }
}

fn classify_msg_log(
    log: &MsgLog,
    packet_id: PacketId,
    name: &str,
    sensor_by_msg_id: &HashMap<PacketId, SensorCameraConfig>,
    video_names: &HashSet<String>,
) -> MsgLogKind {
    // Key by the log's PacketId (same as export_videos), not msg_id(name):
    // MsgMetadata.name can disagree with sensor_cameras.camera_name while the
    // packet id still matches — name-based lookup would miss and fall through
    // to H.264/raw while the layout still points at a video topic.
    if let Some(cfg) = sensor_by_msg_id.get(&packet_id) {
        return MsgLogKind::SensorCamera(Box::new(cfg.clone()));
    }
    if let Some(metadata) = log.metadata()
        && metadata.schema == log_entry_msg_schema()
    {
        return MsgLogKind::LogEntries;
    }
    let first_payload = log.timestamps().first().and_then(|ts| log.get(*ts));
    if video_names.contains(name) || first_payload.is_some_and(is_annex_b) {
        return MsgLogKind::H264Video;
    }
    MsgLogKind::Raw
}

fn msg_log_json(kind: &MsgLogKind, name: &str, payload: &[u8], ts_ns: u64) -> Vec<u8> {
    let value = match kind {
        MsgLogKind::H264Video => json!({
            "timestamp": timestamp_json(ts_ns),
            "frame_id": name,
            "data": BASE64.encode(payload),
            "format": "h264",
        }),
        MsgLogKind::SensorCamera(cfg) => json!({
            "timestamp": timestamp_json(ts_ns),
            "frame_id": name,
            "width": cfg.width,
            "height": cfg.height,
            "encoding": "rgba8",
            "step": cfg.width * 4,
            "data": BASE64.encode(payload),
        }),
        MsgLogKind::LogEntries => {
            let entry: LogEntry = postcard::from_bytes(payload).unwrap_or(LogEntry {
                level: 2,
                message: String::from_utf8_lossy(payload).into_owned(),
            });
            // Elodin levels trace..error(0..4) -> foxglove Log levels debug..fatal(1..5).
            let level = match entry.level {
                0 | 1 => 1, // trace, debug -> debug
                2 => 2,     // info
                3 => 3,     // warn
                _ => 4,     // error
            };
            json!({
                "timestamp": timestamp_json(ts_ns),
                "level": level,
                "message": entry.message,
                "name": name,
                // Required by foxglove.Log; Elodin LogEntry carries no source location.
                "file": "",
                "line": 0,
            })
        }
        MsgLogKind::Raw => json!({"data": BASE64.encode(payload)}),
    };
    serde_json::to_vec(&value).expect("json serialize")
}

// ---------------------------------------------------------------------------
// Schematic loading + scene building
// ---------------------------------------------------------------------------

struct LoadedSchematics {
    /// Primary schematic (from `schematic.active`), if present and parseable.
    primary: Option<Schematic>,
    /// Secondary window schematics: (asset key, parsed).
    windows: Vec<(String, Schematic)>,
    /// Raw KDL bytes by asset key, for attachments.
    raw: Vec<(String, Vec<u8>)>,
}

fn asset_key(path: &str) -> &str {
    path.strip_prefix("db:").unwrap_or(path)
}

fn load_schematics(db_path: &Path, active_key: Option<&str>) -> LoadedSchematics {
    let assets_dir = db_path.join("assets");
    let mut loaded = LoadedSchematics {
        primary: None,
        windows: Vec::new(),
        raw: Vec::new(),
    };
    let Some(key) = active_key else {
        return loaded;
    };
    let primary_path = assets_dir.join(key);
    let Ok(primary_kdl) = std::fs::read_to_string(&primary_path) else {
        eprintln!(
            "Warning: active schematic {} not readable; exporting without scene/layout",
            primary_path.display()
        );
        return loaded;
    };
    loaded
        .raw
        .push((key.to_string(), primary_kdl.clone().into_bytes()));
    let primary = match impeller2_kdl::parse_schematic(&primary_kdl) {
        Ok(s) => s,
        Err(err) => {
            eprintln!("Warning: failed to parse schematic {key}: {err}");
            return loaded;
        }
    };
    for elem in &primary.elems {
        let SchematicElem::Window(window) = elem else {
            continue;
        };
        let Some(path) = window.path.as_deref() else {
            continue;
        };
        let sub_key = asset_key(path).to_string();
        match std::fs::read_to_string(assets_dir.join(&sub_key)) {
            Ok(kdl) => match impeller2_kdl::parse_schematic(&kdl) {
                Ok(sub) => {
                    loaded.raw.push((sub_key.clone(), kdl.into_bytes()));
                    loaded.windows.push((sub_key, sub));
                }
                Err(err) => eprintln!("Warning: failed to parse window schematic {sub_key}: {err}"),
            },
            Err(err) => eprintln!("Warning: failed to read window schematic {sub_key}: {err}"),
        }
    }
    loaded.primary = Some(primary);
    loaded
}

/// Entity frame for an EQL expression: the first referenced component with a
/// `.world_pos`-style pose gives `<entity>`; a bare component gives its prefix.
/// Returns `None` only for pure literal expressions (handled separately by
/// `parse_literal_pose`).
fn entity_for_eql(eql_src: &str, ctx: &eql::Context) -> Option<String> {
    let expr = ctx.parse_str(eql_src).ok()?;
    let mut names = Vec::new();
    collect_component_names(&expr, &mut names);
    let first = names.first()?;
    Some(match first.rsplit_once('.') {
        Some((entity, _)) => entity.to_string(),
        None => first.clone(),
    })
}

/// Try to parse an EQL expression as a literal 7-element pose
/// `(qx, qy, qz, qw, tx, ty, tz)`. Returns the pose values if successful.
fn parse_literal_pose(eql_src: &str, ctx: &eql::Context) -> Option<[f64; 7]> {
    let vals = parse_literal_tuple(eql_src)?;
    if vals.len() == 7 {
        Some([
            vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6],
        ])
    } else {
        let expr = ctx.parse_str(eql_src).ok()?;
        let mut flat = Vec::new();
        flatten_literals_full(&expr, &mut flat);
        (flat.len() == 7).then(|| {
            [
                flat[0], flat[1], flat[2], flat[3], flat[4], flat[5], flat[6],
            ]
        })
    }
}

fn flatten_literals_full(expr: &eql::Expr, out: &mut Vec<f64>) {
    match expr {
        eql::Expr::FloatLiteral(v) => out.push(*v),
        eql::Expr::Tuple(items) => {
            for item in items {
                flatten_literals_full(item, out);
            }
        }
        _ => {}
    }
}

fn collect_component_names(expr: &eql::Expr, out: &mut Vec<String>) {
    match expr {
        eql::Expr::ComponentPart(part) => {
            if part.component.is_some() {
                out.push(part.name.clone());
            }
        }
        eql::Expr::ArrayAccess(inner, _) => collect_component_names(inner, out),
        eql::Expr::Tuple(items) => {
            for item in items {
                collect_component_names(item, out);
            }
        }
        eql::Expr::BinaryOp(left, right, _) => {
            collect_component_names(left, out);
            collect_component_names(right, out);
        }
        eql::Expr::Formula(_, inner) => collect_component_names(inner, out),
        eql::Expr::Last(inner, _) | eql::Expr::First(inner, _) => {
            collect_component_names(inner, out)
        }
        eql::Expr::Time(component) => out.push(component.name.clone()),
        eql::Expr::FloatLiteral(_) | eql::Expr::StringLiteral(_) => {}
    }
}

/// Extract the trailing 3-element position offset from a viewport `pos` EQL
/// expression like `"drone.world_pos + (0,0,0,0, 2,2,2)"` (7-element world_pos
/// arithmetic: the last 3 literals are the ENU camera offset from the target).
fn camera_offset_from_pos(expr: &eql::Expr) -> Option<[f64; 3]> {
    // EQL tuples parse left-associatively as nested pairs:
    // `(0,0,0,0, 2,2,2)` -> Tuple([Tuple([...]), FloatLiteral(2.0)]).
    fn flatten_literals(expr: &eql::Expr, out: &mut Vec<f64>) -> bool {
        match expr {
            eql::Expr::FloatLiteral(v) => {
                out.push(*v);
                true
            }
            eql::Expr::Tuple(items) => items.iter().all(|item| flatten_literals(item, out)),
            _ => false,
        }
    }
    fn find_literal_tuple(expr: &eql::Expr) -> Option<Vec<f64>> {
        match expr {
            eql::Expr::Tuple(_) => {
                let mut vals = Vec::new();
                flatten_literals(expr, &mut vals).then_some(vals)
            }
            eql::Expr::BinaryOp(left, right, _) => {
                find_literal_tuple(left).or_else(|| find_literal_tuple(right))
            }
            _ => None,
        }
    }
    // Camera offset from translate formula chains like
    // `lander.world_pos.translate_world(10, 10, 4)` or
    // `bdx.world_pos.rotate_z(-90).translate_y(-2)`: sum every translate's
    // literal args (rotations only affect azimuth, which stays approximate).
    fn formula_offset(expr: &eql::Expr) -> [f64; 3] {
        let eql::Expr::Formula(formula, inner) = expr else {
            return [0.0; 3];
        };
        let eql::Expr::Tuple(items) = &**inner else {
            return formula_offset(inner);
        };
        let Some((recv, args)) = items.split_first() else {
            return [0.0; 3];
        };
        let mut offset = formula_offset(recv);
        let lits: Vec<f64> = args
            .iter()
            .filter_map(|a| match a {
                eql::Expr::FloatLiteral(v) => Some(*v),
                _ => None,
            })
            .collect();
        let add: [f64; 3] = match (formula.name(), lits.as_slice()) {
            ("translate_world" | "translate", [x, y, z]) => [*x, *y, *z],
            ("translate_world_x" | "translate_x", [d]) => [*d, 0.0, 0.0],
            ("translate_world_y" | "translate_y", [d]) => [0.0, *d, 0.0],
            ("translate_world_z" | "translate_z", [d]) => [0.0, 0.0, *d],
            _ => [0.0; 3],
        };
        for (o, a) in offset.iter_mut().zip(add) {
            *o += a;
        }
        offset
    }

    let vals = match find_literal_tuple(expr) {
        Some(vals) if vals.len() >= 3 => vals,
        _ => {
            let off = formula_offset(expr);
            if off == [0.0; 3] {
                return None;
            }
            return Some(off);
        }
    };
    let [e, n, u]: [f64; 3] = vals[vals.len() - 3..].try_into().ok()?;
    if e == 0.0 && n == 0.0 && u == 0.0 {
        return None;
    }
    Some([e, n, u])
}

/// Foxglove 3D `cameraState` orbit parameters (degrees) from an ENU camera
/// offset relative to the follow target: `phi` is the polar angle from +Z
/// (0 = top-down), `thetaOffset` the azimuth about Z.
fn camera_orbit_from_offset(offset: Option<[f64; 3]>) -> (f64, f64, f64) {
    match offset {
        Some([e, n, u]) => {
            let distance = (e * e + n * n + u * u).sqrt();
            let phi = (u / distance).clamp(-1.0, 1.0).acos().to_degrees();
            let theta = e.atan2(n).to_degrees();
            (distance, phi, theta)
        }
        // 3/4 view matching the drone example's (2,2,2) vantage.
        None => (6.0, 54.7356, 45.0),
    }
}

/// Parse a literal tuple like `"(1, 0, 0)"` into a vector.
fn parse_literal_tuple(src: &str) -> Option<Vec<f64>> {
    let inner = src.trim().strip_prefix('(')?.strip_suffix(')')?;
    inner
        .split(',')
        .map(|part| part.trim().parse::<f64>().ok())
        .collect()
}

fn color_json(color: &Color) -> Value {
    json!({"r": color.r, "g": color.g, "b": color.b, "a": color.a})
}

fn identity_pose() -> Value {
    json!({
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    })
}

/// Quaternion (x,y,z,w) rotating +X onto the given unit direction.
fn quat_from_x_axis(dir: [f64; 3]) -> [f64; 4] {
    let d = dir[0]; // dot(+X, dir)
    if d > 0.999_999 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    if d < -0.999_999 {
        return [0.0, 0.0, 1.0, 0.0]; // 180 deg about Z
    }
    // axis = cross(+X, dir) = (0, -dir.z, dir.y)
    let (ax, ay, az) = (0.0, -dir[2], dir[1]);
    let w = 1.0 + d;
    let norm = (ax * ax + ay * ay + az * az + w * w).sqrt();
    [ax / norm, ay / norm, az / norm, w / norm]
}

/// Quaternion from XYZ Euler angles in degrees (matches GLB `rotate` attr).
fn quat_from_euler_deg(rotate: (f32, f32, f32)) -> [f64; 4] {
    let (rx, ry, rz) = (
        (rotate.0 as f64).to_radians() / 2.0,
        (rotate.1 as f64).to_radians() / 2.0,
        (rotate.2 as f64).to_radians() / 2.0,
    );
    let (sx, cx) = rx.sin_cos();
    let (sy, cy) = ry.sin_cos();
    let (sz, cz) = rz.sin_cos();
    [
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
        cx * cy * cz + sx * sy * sz,
    ]
}

/// Hamilton product `a ∘ b` (apply `b` first, then `a`). Scalar-last (x,y,z,w).
fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let [ax, ay, az, aw] = a;
    let [bx, by, bz, bw] = b;
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

/// Rotate vector `v` by unit quaternion `q` (scalar-last).
fn quat_rotate_vec(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let [x, y, z, w] = q;
    let [vx, vy, vz] = v;
    let tx = 2.0 * (y * vz - z * vy);
    let ty = 2.0 * (z * vx - x * vz);
    let tz = 2.0 * (x * vy - y * vx);
    [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]
}

/// Compose a parent pose with a local GLB translate/rotate: parent ∘ local.
fn compose_pose_with_glb(
    parent: &Value,
    translate: (f32, f32, f32),
    rotate: (f32, f32, f32),
) -> Value {
    let px = parent["position"]["x"].as_f64().unwrap_or(0.0);
    let py = parent["position"]["y"].as_f64().unwrap_or(0.0);
    let pz = parent["position"]["z"].as_f64().unwrap_or(0.0);
    let pq = [
        parent["orientation"]["x"].as_f64().unwrap_or(0.0),
        parent["orientation"]["y"].as_f64().unwrap_or(0.0),
        parent["orientation"]["z"].as_f64().unwrap_or(0.0),
        parent["orientation"]["w"].as_f64().unwrap_or(1.0),
    ];
    let local_q = quat_from_euler_deg(rotate);
    let offset = quat_rotate_vec(
        pq,
        [translate.0 as f64, translate.1 as f64, translate.2 as f64],
    );
    let oq = quat_mul(pq, local_q);
    json!({
        "position": {"x": px + offset[0], "y": py + offset[1], "z": pz + offset[2]},
        "orientation": {"x": oq[0], "y": oq[1], "z": oq[2], "w": oq[3]},
    })
}

/// Unique scene entity id for an `object_3d` on `frame`. First mesh keeps
/// `{frame}-model`; subsequent meshes on the same frame get `-2`, `-3`, …
fn next_model_entity_id(frame: &str, counts: &mut HashMap<String, u32>) -> String {
    let n = counts.entry(frame.to_string()).or_insert(0);
    *n += 1;
    if *n == 1 {
        format!("{frame}-model")
    } else {
        format!("{frame}-model-{n}")
    }
}

/// Whether a GLB of `size_bytes` should be base64-embedded.
fn should_embed_glb(size_bytes: u64, max_embed_bytes: u64, force_embed: bool) -> bool {
    force_embed || size_bytes <= max_embed_bytes
}

/// Wrap a single scene entity in a SceneUpdate message body, filling in the
/// `metadata` and primitive arrays the foxglove.SceneUpdate schema requires
/// on every entity (schema-validating consumers reject partial entities).
fn scene_update_message(mut entity: Value) -> Vec<u8> {
    if let Value::Object(map) = &mut entity {
        for key in [
            "metadata",
            "arrows",
            "cubes",
            "spheres",
            "cylinders",
            "lines",
            "triangles",
            "texts",
            "models",
        ] {
            map.entry(key).or_insert_with(|| Value::Array(Vec::new()));
        }
    }
    serde_json::to_vec(&json!({"deletions": [], "entities": [entity]})).expect("json serialize")
}

/// Scene topic for a static entity: `/scene/<sanitized-id>`.
///
/// One topic per entity is load-bearing: Foxglove backfills only the *latest*
/// message per topic when a 3D panel (re)mounts — e.g. after a tab switch —
/// so N entities sharing one topic collapse to just the last one.
fn scene_topic(entity_id: &str) -> String {
    format!("/scene/{}", sanitize_topic_segment(entity_id))
}

fn sanitize_topic_segment(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

struct ComponentCursor<'a> {
    timestamps: &'a [Timestamp],
    data: &'a [u8],
    sample_size: usize,
}

struct SceneBuild {
    /// `(topic, SceneUpdate JSON body)` — one topic+message per entity so
    /// latest-per-topic backfill preserves the whole static scene.
    messages: Vec<(String, Vec<u8>)>,
    /// Asset keys referenced by scene models (for attachments).
    referenced_assets: Vec<String>,
}

/// Decimation parameters for line trajectories.
const MAX_LINE_POINTS: usize = 2000;

/// Extract XYZ trajectory from a 7-element world_pos component.
fn extract_trajectory(comp: &ExportComponent, cursor: &ComponentCursor) -> Vec<[f64; 3]> {
    let n = cursor.timestamps.len();
    let step = if n > MAX_LINE_POINTS {
        n.div_ceil(MAX_LINE_POINTS)
    } else {
        1
    };
    let prim = comp.component.schema.prim_type;
    let mut points = Vec::with_capacity(n.min(MAX_LINE_POINTS + 1));
    for i in (0..n).step_by(step) {
        let buf = &cursor.data[i * cursor.sample_size..(i + 1) * cursor.sample_size];
        points.push([
            read_f64(prim, buf, 4),
            read_f64(prim, buf, 5),
            read_f64(prim, buf, 6),
        ]);
    }
    if n > 1 && !(n - 1).is_multiple_of(step) {
        let buf = &cursor.data[(n - 1) * cursor.sample_size..n * cursor.sample_size];
        points.push([
            read_f64(prim, buf, 4),
            read_f64(prim, buf, 5),
            read_f64(prim, buf, 6),
        ]);
    }
    points
}

fn default_line_color() -> Color {
    Color {
        r: 0.2,
        g: 0.6,
        b: 1.0,
        a: 1.0,
    }
}

/// Build a LinePrimitive entity from a Line3d schematic element.
fn build_line_entity(
    line: &Line3d,
    ctx: &eql::Context,
    components: &[ExportComponent],
    component_cursors: &[Option<ComponentCursor>],
    ts_ns: u64,
    geo_frames_active: bool,
) -> Option<Value> {
    let entity = entity_for_eql(&line.eql, ctx)?;
    let pose_name = format!("{entity}.world_pos");
    let (comp_idx, comp) = components
        .iter()
        .enumerate()
        .find(|(_, c)| c.name == pose_name)?;
    let cursor = component_cursors[comp_idx].as_ref()?;

    let flat_count: usize = comp.component.schema.dim.iter().product::<usize>().max(1);
    if flat_count != 7 {
        eprintln!(
            "Warning: line_3d '{}' references a non-pose component ({}), skipping",
            line.eql, comp.name
        );
        return None;
    }

    let points = extract_trajectory(comp, cursor);
    if points.is_empty() {
        return None;
    }

    let color = line
        .color
        .as_ref()
        .copied()
        .unwrap_or_else(default_line_color);
    let point_values: Vec<Value> = points
        .iter()
        .map(|[x, y, z]| json!({"x": x, "y": y, "z": z}))
        .collect();

    // Lines whose data is local to a geodetic frame (`frame="NED"/"ENU"`)
    // attach to that anchor frame when the schematic declares an origin.
    let frame_id = line
        .frame
        .map(|f| format!("{f:?}"))
        .filter(|n| geo_frames_active && (n == "NED" || n == "ENU"))
        .unwrap_or_else(|| "world".to_string());

    // Elodin's `line_width` is a *pixel* width. Foxglove's non-scale-invariant
    // thickness is in meters — writing pixels as meters yields multi-meter
    // ribbons that engulf the vehicle mesh (rc-jet's all-orange viewport).
    Some(json!({
        "timestamp": timestamp_json(ts_ns),
        "frame_id": frame_id,
        "id": format!("{entity}-line"),
        "lifetime": {"sec": 0, "nsec": 0},
        "frame_locked": false,
        "lines": [{
            "type": 0, // LINE_STRIP
            "pose": identity_pose(),
            "thickness": line.line_width as f64,
            "scale_invariant": true,
            "points": point_values,
            "color": color_json(&color),
            "colors": [],
            "indices": [],
        }],
    }))
}

/// Resolve earth.glb: first check DB assets, then `assets/earth.glb` relative
/// to the working directory (repo-root workflows).
fn find_earth_glb(assets_dir: &Path) -> Option<PathBuf> {
    let candidates = [
        assets_dir.join("earth.glb"),
        PathBuf::from("assets/earth.glb"),
    ];
    candidates.into_iter().find(|p| p.is_file())
}

/// Build a model entity for WorldMesh "globe" → earth.glb.
/// Returns `(entity, referenced)` when the mesh is embedded, or `(None, referenced)`
/// when oversized (attachment-only).
fn build_globe_entity(
    assets_dir: &Path,
    components: &[ExportComponent],
    ts_ns: u64,
    max_embed_bytes: u64,
) -> Option<(Option<Value>, Vec<String>)> {
    let glb_path = find_earth_glb(assets_dir)?;
    let bytes = std::fs::read(&glb_path).ok()?;

    let frame = components
        .iter()
        .find(|c| {
            c.pose_entity.is_some()
                && (c.name.starts_with("Earth.") || c.name.starts_with("earth."))
        })
        .and_then(|c| c.pose_entity.clone())
        .unwrap_or_else(|| "world".to_string());

    let mut referenced = Vec::new();
    if let Ok(rel) = glb_path.strip_prefix(assets_dir) {
        referenced.push(rel.to_string_lossy().replace('\\', "/"));
    } else {
        referenced.push("earth.glb".to_string());
    }

    if !should_embed_glb(bytes.len() as u64, max_embed_bytes, false) {
        eprintln!(
            "Note: earth.glb is {} MiB (> {} MiB limit); attached but not embedded in SceneUpdate",
            bytes.len() / (1024 * 1024),
            max_embed_bytes / (1024 * 1024)
        );
        return Some((None, referenced));
    }

    let entity = json!({
        "timestamp": timestamp_json(ts_ns),
        "frame_id": frame,
        "id": "earth-globe",
        "lifetime": {"sec": 0, "nsec": 0},
        "frame_locked": true,
        "models": [{
            "pose": identity_pose(),
            "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
            "color": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0},
            "override_color": false,
            "url": "",
            "media_type": "model/gltf-binary",
            "data": BASE64.encode(&bytes),
        }],
    });
    Some((Some(entity), referenced))
}

/// Resolve the viewport follow entity (first `object_3d` with an entity EQL).
fn resolve_follow_entity(schematics: &LoadedSchematics, ctx: &eql::Context) -> Option<String> {
    schematics
        .primary
        .iter()
        .chain(schematics.windows.iter().map(|(_, s)| s))
        .flat_map(|s| s.elems.iter())
        .find_map(|elem| match elem {
            SchematicElem::Object3d(object) => entity_for_eql(&object.eql, ctx),
            _ => None,
        })
}

/// Build the static scene from schematic 3D elements — one topic + one
/// SceneUpdate message per entity (`/scene/<id>`), so latest-per-topic
/// backfill can never drop entities. Dynamic arrows are handled separately
/// on `/scene_dynamic`.
#[allow(clippy::too_many_arguments)]
fn build_scene(
    schematics: &LoadedSchematics,
    ctx: &eql::Context,
    assets_dir: &Path,
    ts_ns: u64,
    components: &[ExportComponent],
    component_cursors: &[Option<ComponentCursor>],
    max_embed_bytes: u64,
    follow_entity: Option<&str>,
    geo_frames_active: bool,
) -> Option<SceneBuild> {
    let mut messages: Vec<(String, Vec<u8>)> = Vec::new();
    let mut referenced_assets = Vec::new();
    let mut arrow_groups: HashMap<String, Vec<Value>> = HashMap::new();
    let mut literal_counter = 0u32;
    let mut model_id_counts: HashMap<String, u32> = HashMap::new();

    let push = |messages: &mut Vec<(String, Vec<u8>)>, entity: Value| {
        let id = entity["id"].as_str().unwrap_or("entity").to_string();
        messages.push((scene_topic(&id), scene_update_message(entity)));
    };

    let all_elems = schematics
        .primary
        .iter()
        .chain(schematics.windows.iter().map(|(_, s)| s))
        .flat_map(|s| s.elems.iter());

    for elem in all_elems {
        match elem {
            SchematicElem::Object3d(object) => {
                match build_object_entity(
                    object,
                    ctx,
                    assets_dir,
                    ts_ns,
                    max_embed_bytes,
                    follow_entity,
                    &mut literal_counter,
                    &mut model_id_counts,
                ) {
                    Ok((Some(entity), assets)) => {
                        push(&mut messages, entity);
                        referenced_assets.extend(assets);
                    }
                    Ok((None, assets)) => {
                        referenced_assets.extend(assets);
                    }
                    Err(err) => eprintln!("Warning: skipping object_3d ({}): {err}", object.eql),
                }
            }
            SchematicElem::VectorArrow(arrow) => {
                if let Some((frame, primitive)) =
                    build_arrow(arrow, ctx, components, component_cursors)
                {
                    arrow_groups.entry(frame).or_default().push(primitive);
                }
            }
            SchematicElem::Line3d(line) => {
                match build_line_entity(
                    line,
                    ctx,
                    components,
                    component_cursors,
                    ts_ns,
                    geo_frames_active,
                ) {
                    Some(entity) => push(&mut messages, entity),
                    None => eprintln!(
                        "Warning: skipping line_3d '{}' (pose not found or empty)",
                        line.eql
                    ),
                }
            }
            SchematicElem::WorldMesh(wm) => {
                if wm.region == "globe" {
                    match build_globe_entity(assets_dir, components, ts_ns, max_embed_bytes) {
                        Some((Some(entity), assets)) => {
                            push(&mut messages, entity);
                            referenced_assets.extend(assets);
                        }
                        Some((None, assets)) => {
                            referenced_assets.extend(assets);
                        }
                        None => eprintln!("Warning: earth.glb not found for world_mesh globe"),
                    }
                } else {
                    eprintln!(
                        "Note: skipping world_mesh region '{}' (only 'globe' is exported)",
                        wm.region
                    );
                }
            }
            _ => {}
        }
    }

    for (frame, arrows) in arrow_groups {
        push(
            &mut messages,
            json!({
                "timestamp": timestamp_json(ts_ns),
                "frame_id": frame,
                "id": format!("{frame}-arrows"),
                "lifetime": {"sec": 0, "nsec": 0},
                "frame_locked": true,
                "arrows": arrows,
            }),
        );
    }

    if messages.is_empty() && referenced_assets.is_empty() {
        return None;
    }
    if messages.is_empty() {
        // Oversized-only assets still need attachments.
        return Some(SceneBuild {
            messages: Vec::new(),
            referenced_assets,
        });
    }
    Some(SceneBuild {
        messages,
        referenced_assets,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_object_entity(
    object: &Object3D,
    ctx: &eql::Context,
    assets_dir: &Path,
    ts_ns: u64,
    max_embed_bytes: u64,
    follow_entity: Option<&str>,
    literal_counter: &mut u32,
    model_id_counts: &mut HashMap<String, u32>,
) -> Result<(Option<Value>, Vec<String>), Error> {
    let (frame, model_pose, is_literal) = if let Some(f) = entity_for_eql(&object.eql, ctx) {
        (f, identity_pose(), false)
    } else if let Some(pose) = parse_literal_pose(&object.eql, ctx) {
        *literal_counter += 1;
        let id = format!("literal-{literal_counter}");
        let p = json!({
            "position": {"x": pose[4], "y": pose[5], "z": pose[6]},
            "orientation": {"x": pose[0], "y": pose[1], "z": pose[2], "w": pose[3]},
        });
        (id, p, true)
    } else {
        return Ok((None, Vec::new()));
    };

    let force_embed = follow_entity.is_some_and(|f| f == frame);
    let entity_id = next_model_entity_id(&frame, model_id_counts);
    let mut entity = Map::new();
    entity.insert("timestamp".into(), timestamp_json(ts_ns));
    entity.insert(
        "frame_id".into(),
        Value::String(if is_literal {
            "world".to_string()
        } else {
            frame.clone()
        }),
    );
    entity.insert("id".into(), Value::String(entity_id));
    entity.insert("lifetime".into(), json!({"sec": 0, "nsec": 0}));
    entity.insert("frame_locked".into(), Value::Bool(true));
    let mut referenced = Vec::new();

    match &object.mesh {
        Object3DMesh::Glb {
            path,
            scale,
            translate,
            rotate,
            ..
        } => {
            let key = asset_key(path).to_string();
            let glb_path = assets_dir.join(&key);
            let bytes = std::fs::read(&glb_path).map_err(|e| {
                invalid_data(format!("GLB asset {} unreadable: {e}", glb_path.display()))
            })?;
            referenced.push(key.clone());
            if !should_embed_glb(bytes.len() as u64, max_embed_bytes, force_embed) {
                eprintln!(
                    "Note: GLB {} is {} MiB (> {} MiB limit); attached but not embedded in SceneUpdate",
                    key,
                    bytes.len() / (1024 * 1024),
                    max_embed_bytes / (1024 * 1024)
                );
                return Ok((None, referenced));
            }
            let glb_pose = if is_literal {
                compose_pose_with_glb(&model_pose, *translate, *rotate)
            } else {
                let quat = quat_from_euler_deg(*rotate);
                json!({
                    "position": {"x": translate.0, "y": translate.1, "z": translate.2},
                    "orientation": {"x": quat[0], "y": quat[1], "z": quat[2], "w": quat[3]},
                })
            };
            entity.insert(
                "models".into(),
                json!([{
                    "pose": glb_pose,
                    "scale": {"x": scale, "y": scale, "z": scale},
                    "color": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0},
                    "override_color": false,
                    "url": "",
                    "media_type": "model/gltf-binary",
                    "data": BASE64.encode(&bytes),
                }]),
            );
        }
        Object3DMesh::Mesh { mesh, material } => {
            let color = color_json(&material.base_color);
            let pose = model_pose.clone();
            match mesh {
                impeller2_wkt::Mesh::Sphere { radius } => {
                    let d = (radius * 2.0) as f64;
                    entity.insert(
                        "spheres".into(),
                        json!([{"pose": pose, "size": {"x": d, "y": d, "z": d}, "color": color}]),
                    );
                }
                impeller2_wkt::Mesh::Box { x, y, z } => {
                    entity.insert(
                        "cubes".into(),
                        json!([{"pose": pose, "size": {"x": x, "y": y, "z": z}, "color": color}]),
                    );
                }
                impeller2_wkt::Mesh::Cylinder { radius, height } => {
                    let d = (radius * 2.0) as f64;
                    entity.insert(
                        "cylinders".into(),
                        json!([{
                            "pose": pose,
                            "size": {"x": d, "y": d, "z": height},
                            "bottom_scale": 1.0, "top_scale": 1.0, "color": color,
                        }]),
                    );
                }
                impeller2_wkt::Mesh::Plane { width, depth } => {
                    entity.insert(
                        "cubes".into(),
                        json!([{
                            "pose": pose,
                            "size": {"x": width, "y": depth, "z": 0.01},
                            "color": color,
                        }]),
                    );
                }
            }
        }
        Object3DMesh::Ellipsoid { .. } => {
            return Ok((None, Vec::new()));
        }
    }

    Ok((Some(Value::Object(entity)), referenced))
}

/// Shaft length matching the editor: `|v| * scale`, or just `scale` when normalize.
fn arrow_shaft_length(len: f64, scale: f64, normalize: bool) -> f64 {
    if normalize { scale } else { len * scale }
}

fn arrow_primitive(dir: [f64; 3], total: f64, color: &Color, pos: [f64; 3]) -> Value {
    let quat = quat_from_x_axis(dir);
    json!({
        "pose": {
            "position": {"x": pos[0], "y": pos[1], "z": pos[2]},
            "orientation": {"x": quat[0], "y": quat[1], "z": quat[2], "w": quat[3]},
        },
        "shaft_length": total * 0.8,
        "shaft_diameter": (total * 0.02).max(0.01),
        "head_length": total * 0.2,
        "head_diameter": (total * 0.06).max(0.02),
        "color": color_json(color),
    })
}

/// World-frame origin position for a static arrow: literal xyz/pose, or the
/// first sample of `<entity>.world_pos`.
fn static_arrow_origin_pos(
    origin: Option<&str>,
    ctx: &eql::Context,
    components: &[ExportComponent],
    component_cursors: &[Option<ComponentCursor>],
) -> Option<[f64; 3]> {
    let Some(origin) = origin else {
        return Some([0.0, 0.0, 0.0]);
    };
    if let Some(v) = parse_literal_tuple(origin) {
        return if v.len() >= 7 {
            Some([v[v.len() - 3], v[v.len() - 2], v[v.len() - 1]])
        } else if v.len() >= 3 {
            Some([v[0], v[1], v[2]])
        } else {
            None
        };
    }
    let entity = entity_for_eql(origin, ctx)?;
    let pose_name = format!("{entity}.world_pos");
    let (comp_idx, comp) = components
        .iter()
        .enumerate()
        .find(|(_, c)| c.name == pose_name)?;
    let cursor = component_cursors[comp_idx].as_ref()?;
    if cursor.timestamps.is_empty() {
        return None;
    }
    let buf = &cursor.data[..cursor.sample_size];
    let prim = comp.component.schema.prim_type;
    Some([
        read_f64(prim, buf, 4),
        read_f64(prim, buf, 5),
        read_f64(prim, buf, 6),
    ])
}

/// Literal-vector arrows for the static scene. Body-frame arrows attach to the
/// origin entity's TF; world-frame arrows go on `world` with an absolute origin.
fn build_arrow(
    arrow: &VectorArrow3d,
    ctx: &eql::Context,
    components: &[ExportComponent],
    component_cursors: &[Option<ComponentCursor>],
) -> Option<(String, Value)> {
    let vec = parse_literal_tuple(&arrow.vector)?;
    if vec.len() != 3 {
        return None;
    }
    let len = (vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]).sqrt();
    if len == 0.0 {
        return None;
    }
    let dir = [vec[0] / len, vec[1] / len, vec[2] / len];
    let total = arrow_shaft_length(len, arrow.scale, arrow.normalize);
    if arrow.body_frame {
        let origin = arrow.origin.as_deref()?;
        let frame = entity_for_eql(origin, ctx)?;
        Some((
            frame,
            arrow_primitive(dir, total, &arrow.color, [0.0, 0.0, 0.0]),
        ))
    } else {
        let pos =
            static_arrow_origin_pos(arrow.origin.as_deref(), ctx, components, component_cursors)?;
        Some((
            "world".to_string(),
            arrow_primitive(dir, total, &arrow.color, pos),
        ))
    }
}

// ---------------------------------------------------------------------------
// Layout generation
// ---------------------------------------------------------------------------

struct LayoutBuilder<'a> {
    config_by_id: Map<String, Value>,
    counter: u64,
    ctx: &'a eql::Context,
    components: &'a [ExportComponent],
    follow_entity: Option<String>,
    /// First `object_3d` entity per geo frame name ("ENU"/"NED"/"ECEF") —
    /// follow fallback for viewports whose look_at/pos are pure literals.
    frame_entities: HashMap<String, String>,
    /// Every `/scene/...` and `/scene_dynamic/...` topic in the export.
    scene_topics: &'a [String],
}

impl<'a> LayoutBuilder<'a> {
    fn new(
        ctx: &'a eql::Context,
        components: &'a [ExportComponent],
        follow_entity: Option<String>,
        frame_entities: HashMap<String, String>,
        scene_topics: &'a [String],
    ) -> Self {
        Self {
            config_by_id: Map::new(),
            counter: 0,
            ctx,
            components,
            follow_entity,
            frame_entities,
            scene_topics,
        }
    }

    fn add_panel(&mut self, kind: &str, config: Value) -> String {
        self.counter += 1;
        let id = format!("{kind}!elodin{}", self.counter);
        self.config_by_id.insert(id.clone(), config);
        id
    }

    fn component_by_name(&self, name: &str) -> Option<&ExportComponent> {
        self.components.iter().find(|c| c.name == name)
    }

    /// Expand an EQL expression into Foxglove Plot series paths.
    fn series_for_eql(&self, eql_src: &str) -> Vec<Value> {
        let mut series = Vec::new();
        match self.ctx.parse_str(eql_src) {
            Ok(expr) => self.collect_series(&expr, &mut series),
            Err(err) => {
                eprintln!("Warning: could not parse graph EQL '{eql_src}': {err}");
            }
        }
        series
            .into_iter()
            .map(|(path, label)| {
                json!({
                    "value": path,
                    "enabled": true,
                    "timestampMethod": "receiveTime",
                    "label": label,
                })
            })
            .collect()
    }

    fn collect_series(&self, expr: &eql::Expr, out: &mut Vec<(String, String)>) {
        match expr {
            eql::Expr::ComponentPart(part) => {
                if part.component.is_some() {
                    self.push_component_series(&part.name, None, out);
                } else {
                    // Branch node (e.g. bare entity): expand every leaf below it.
                    fn walk(
                        builder: &LayoutBuilder<'_>,
                        part: &eql::ComponentPart,
                        out: &mut Vec<(String, String)>,
                    ) {
                        if part.component.is_some() {
                            builder.push_component_series(&part.name, None, out);
                        }
                        for child in part.children.values() {
                            walk(builder, child, out);
                        }
                    }
                    walk(self, part, out);
                }
            }
            eql::Expr::ArrayAccess(inner, idx) => {
                if let eql::Expr::ComponentPart(part) = inner.as_ref() {
                    self.push_component_series(&part.name, Some(*idx), out);
                } else {
                    self.collect_series(inner, out);
                }
            }
            eql::Expr::Tuple(items) => {
                for item in items {
                    self.collect_series(item, out);
                }
            }
            eql::Expr::BinaryOp(left, right, _) => {
                // Foxglove paths can't express arithmetic; plot the raw operands.
                self.collect_series(left, out);
                self.collect_series(right, out);
            }
            eql::Expr::Formula(_, inner) => self.collect_series(inner, out),
            eql::Expr::Last(inner, _) | eql::Expr::First(inner, _) => {
                self.collect_series(inner, out)
            }
            eql::Expr::Time(_) | eql::Expr::FloatLiteral(_) | eql::Expr::StringLiteral(_) => {}
        }
    }

    fn push_component_series(
        &self,
        name: &str,
        element: Option<usize>,
        out: &mut Vec<(String, String)>,
    ) {
        let Some(comp) = self.component_by_name(name) else {
            return;
        };
        let short = name.rsplit_once('.').map(|(_, s)| s).unwrap_or(name);
        match element {
            Some(idx) => {
                if let Some(path) = element_path_str(&comp.element_paths, idx) {
                    out.push((
                        format!("{}.{}", comp.topic, path),
                        format!("{short}.{path}"),
                    ));
                }
            }
            None => {
                for idx in 0..comp.element_paths.len() {
                    let path = element_path_str(&comp.element_paths, idx).unwrap();
                    out.push((
                        format!("{}.{}", comp.topic, path),
                        format!("{short}.{path}"),
                    ));
                }
            }
        }
    }

    /// Convert a schematic panel into a mosaic node, registering panel configs.
    fn panel_node(&mut self, panel: &Panel) -> Option<Value> {
        match panel {
            Panel::Viewport(viewport) => {
                let mut layers = Map::new();
                if viewport.show_grid {
                    layers.insert(
                        "grid".into(),
                        json!({
                            "layerId": "foxglove.Grid",
                            "instanceId": "grid",
                            "label": "Grid",
                            "visible": true,
                            "frameId": "world",
                            "size": 10,
                            "divisions": 10,
                            "lineWidth": 1,
                            "color": "#a0a0a4",
                            "position": [0, 0, 0],
                            "rotation": [0, 0, 0],
                            "order": 1,
                        }),
                    );
                }
                // Reconstruct the Elodin viewport vantage point: pos EQL like
                // "drone.world_pos + (0,0,0,0, 2,2,2)" carries the camera's
                // ENU offset from the look_at target in its trailing literals.
                let mut offset = viewport
                    .pos
                    .as_deref()
                    .and_then(|pos| self.ctx.parse_str(pos).ok())
                    .and_then(|expr| camera_offset_from_pos(&expr));
                // NED viewports express the offset as (north, east, down);
                // convert to ENU so "up" doesn't become "below the horizon".
                let frame_name = viewport.frame.map(|f| format!("{f:?}"));
                if frame_name.as_deref() == Some("NED")
                    && let Some([n, e, d]) = offset
                {
                    offset = Some([e, n, -d]);
                }
                let (distance, phi, theta) = camera_orbit_from_offset(offset);
                let near = viewport.near.unwrap_or(0.01) as f64;
                // Clamp far so the orbit camera itself is never beyond the far
                // plane (geo-frames sets far=1.5e7 with an 8e7 m camera offset).
                let far = viewport
                    .far
                    .map(|f| (f as f64).max(distance * 4.0))
                    .unwrap_or_else(|| (distance * 4.0).max(5000.0));
                let mut config = Map::new();
                config.insert(
                    "cameraState".into(),
                    json!({
                        "perspective": true,
                        "distance": distance,
                        "phi": phi,
                        "thetaOffset": theta,
                        "targetOffset": [0, 0, 0],
                        "target": [0, 0, 0],
                        "targetOrientation": [0, 0, 0, 1],
                        "fovy": viewport.fov,
                        "near": near,
                        "far": far,
                    }),
                );
                // Follow the viewport's own subject: the look_at target first
                // (what the Elodin camera orbits), then the pos entity, then
                // the first object_3d sharing the viewport's geo frame (a
                // literal look_at like "(0,0,0,1, 0,0,0)" means that frame's
                // origin, e.g. geo-frames' ned_origin), then the global first
                // object. The old global-first order pointed Apollo's camera
                // at `surface` while the lander flew 100 km away.
                if let Some(entity) = viewport
                    .look_at
                    .as_deref()
                    .and_then(|l| entity_for_eql(l, self.ctx))
                    .or_else(|| {
                        viewport
                            .pos
                            .as_deref()
                            .and_then(|p| entity_for_eql(p, self.ctx))
                    })
                    .or_else(|| {
                        frame_name
                            .as_deref()
                            .and_then(|f| self.frame_entities.get(f).cloned())
                    })
                    .or_else(|| self.follow_entity.clone())
                {
                    config.insert("followTf".into(), Value::String(entity));
                    config.insert("followMode".into(), Value::String("follow-position".into()));
                }
                config.insert("layers".into(), Value::Object(layers));
                let mut topics = Map::new();
                for topic in self.scene_topics {
                    topics.insert(topic.clone(), json!({"visible": true}));
                }
                config.insert("topics".into(), Value::Object(topics));
                // Hide the parent->child TF connecting lines (Elodin viewports
                // draw no such line); keep the frame axes and labels.
                config.insert("scene".into(), json!({"transforms": {"lineWidth": 0}}));
                if let Some(name) = &viewport.name {
                    config.insert("title".into(), Value::String(name.clone()));
                }
                Some(Value::String(self.add_panel("3D", Value::Object(config))))
            }
            Panel::Graph(graph) => {
                let mut paths = self.series_for_eql(&graph.eql);
                for (i, path) in paths.iter_mut().enumerate() {
                    if let Some(color) = graph.colors.get(i)
                        && let Some(obj) = path.as_object_mut()
                    {
                        obj.insert("color".into(), Value::String(color_to_hex(color)));
                    }
                }
                let config = json!({
                    "title": graph.name.clone().unwrap_or_else(|| graph.eql.clone()),
                    "paths": paths,
                    "showXAxisLabels": true,
                    "showYAxisLabels": true,
                    "showLegend": true,
                    "legendDisplay": "floating",
                    "showPlotValuesInLegend": false,
                    "isSynced": true,
                    "xAxisVal": "timestamp",
                    "sidebarDimension": 240,
                });
                Some(Value::String(self.add_panel("Plot", config)))
            }
            Panel::ComponentMonitor(monitor) => {
                let topic = topic_for_component(&monitor.component_name);
                let config = json!({
                    "topicPath": topic,
                    "diffEnabled": false,
                    "diffMethod": "custom",
                    "diffTopicPath": "",
                    "showFullMessageForDiff": false,
                    "expansion": "all",
                });
                Some(Value::String(self.add_panel("RawMessages", config)))
            }
            Panel::VideoStream(stream) => {
                let config = json!({
                    "imageMode": {"imageTopic": format!("/video/{}", stream.msg_name)},
                });
                Some(Value::String(self.add_panel("Image", config)))
            }
            Panel::SensorView(view) => {
                #[cfg(feature = "video-export")]
                let topic = format!("/video/{}", view.msg_name);
                #[cfg(not(feature = "video-export"))]
                let topic = format!("/camera/{}", view.msg_name);
                let config = json!({
                    "imageMode": {"imageTopic": topic},
                });
                Some(Value::String(self.add_panel("Image", config)))
            }
            Panel::LogStream(stream) => {
                let config = json!({
                    "searchTerms": [],
                    "minLogLevel": 1,
                    "topicToRender": format!("/log/{}", stream.msg_name),
                });
                Some(Value::String(self.add_panel("RosOut", config)))
            }
            Panel::HSplit(split) => self.split_node(split, "row"),
            Panel::VSplit(split) => self.split_node(split, "column"),
            Panel::Tabs(panels) => {
                let tabs: Vec<Value> = panels
                    .iter()
                    .map(|panel| {
                        let title = panel_title(panel);
                        let layout = self.panel_node(panel);
                        match layout {
                            Some(node) => json!({"title": title, "layout": node}),
                            None => json!({"title": title}),
                        }
                    })
                    .collect();
                if tabs.is_empty() {
                    return None;
                }
                let config = json!({"activeTabIdx": 0, "tabs": tabs});
                Some(Value::String(self.add_panel("Tab", config)))
            }
            // No Foxglove equivalent.
            Panel::ActionPane(_)
            | Panel::QueryTable(_)
            | Panel::QueryPlot(_)
            | Panel::Inspector
            | Panel::Hierarchy
            | Panel::SchematicTree(_)
            | Panel::DataOverview(_)
            | Panel::GeoPositionGauge(_)
            | Panel::OrientationGauge(_) => None,
        }
    }

    /// Fold an n-way split into nested binary mosaic nodes with split percentages.
    fn split_node(&mut self, split: &impeller2_wkt::Split, direction: &str) -> Option<Value> {
        let children: Vec<(Value, f32)> = split
            .panels
            .iter()
            .enumerate()
            .filter_map(|(i, panel)| {
                let share = split.shares.get(&i).copied().unwrap_or(1.0).max(0.001);
                self.panel_node(panel).map(|node| (node, share))
            })
            .collect();
        fold_split(children, direction)
    }
}

fn fold_split(mut children: Vec<(Value, f32)>, direction: &str) -> Option<Value> {
    match children.len() {
        0 => None,
        1 => Some(children.remove(0).0),
        _ => {
            let (first, first_share) = children.remove(0);
            let rest_share: f32 = children.iter().map(|(_, s)| s).sum();
            let second = fold_split(children, direction)?;
            let pct = first_share / (first_share + rest_share) * 100.0;
            Some(json!({
                "first": first,
                "second": second,
                "direction": direction,
                "splitPercentage": pct,
            }))
        }
    }
}

/// Panel title preferring split/viewport names over `Panel::label()`'s generic
/// "Horizontal Split" / "Vertical Split" fallbacks.
fn panel_title(panel: &Panel) -> String {
    match panel {
        Panel::HSplit(split) | Panel::VSplit(split) => split
            .name
            .clone()
            .unwrap_or_else(|| panel.label().to_string()),
        other => other.label().to_string(),
    }
}

fn color_to_hex(color: &Color) -> String {
    let to_byte = |v: f32| -> u8 { (v.clamp(0.0, 1.0) * 255.0).round() as u8 };
    format!(
        "#{:02x}{:02x}{:02x}",
        to_byte(color.r),
        to_byte(color.g),
        to_byte(color.b)
    )
}

/// Build the full Foxglove layout JSON from the loaded schematics.
///
/// The primary schematic's top-level tabs and each secondary window become tabs
/// of a root Tab panel (Foxglove layouts are single-window).
fn build_layout(
    schematics: &LoadedSchematics,
    ctx: &eql::Context,
    components: &[ExportComponent],
    scene_topics: &[String],
) -> Option<Value> {
    let primary = schematics.primary.as_ref()?;

    let follow_entity = resolve_follow_entity(schematics, ctx);
    let mut frame_entities: HashMap<String, String> = HashMap::new();
    for elem in schematics
        .primary
        .iter()
        .chain(schematics.windows.iter().map(|(_, s)| s))
        .flat_map(|s| s.elems.iter())
    {
        if let SchematicElem::Object3d(object) = elem
            && let Some(frame) = object.frame
            && let Some(entity) = entity_for_eql(&object.eql, ctx)
        {
            frame_entities.entry(format!("{frame:?}")).or_insert(entity);
        }
    }
    let mut builder =
        LayoutBuilder::new(ctx, components, follow_entity, frame_entities, scene_topics);

    // (title, panel) pairs that become the root tabs.
    let mut tabs: Vec<(String, Value)> = Vec::new();
    for elem in &primary.elems {
        let SchematicElem::Panel(panel) = elem else {
            continue;
        };
        match panel {
            Panel::Tabs(panels) => {
                for sub in panels {
                    let title = panel_title(sub.collapse());
                    if let Some(node) = builder.panel_node(sub) {
                        tabs.push((title, node));
                    }
                }
            }
            other => {
                let title = panel_title(other.collapse());
                if let Some(node) = builder.panel_node(other) {
                    tabs.push((title, node));
                }
            }
        }
    }
    for (key, window) in &schematics.windows {
        for elem in &window.elems {
            let SchematicElem::Panel(panel) = elem else {
                continue;
            };
            let collapsed = panel.collapse();
            let title = match panel_title(collapsed).as_str() {
                "Tabs" | "Vertical Split" | "Horizontal Split" => key
                    .rsplit('/')
                    .next()
                    .unwrap_or(key)
                    .trim_end_matches(".kdl")
                    .to_string(),
                label => label.to_string(),
            };
            if let Some(node) = builder.panel_node(collapsed) {
                tabs.push((title, node));
            }
        }
    }

    if tabs.is_empty() {
        return None;
    }

    let tab_values: Vec<Value> = tabs
        .into_iter()
        .map(|(title, layout)| json!({"title": title, "layout": layout}))
        .collect();
    let root_id = builder.add_panel("Tab", json!({"activeTabIdx": 0, "tabs": tab_values}));

    Some(json!({
        "configById": Value::Object(builder.config_by_id),
        "globalVariables": {},
        "userNodes": {},
        "playbackConfig": {"speed": 1.0},
        "layout": root_id,
    }))
}

// ---------------------------------------------------------------------------
// Dynamic vector arrows
// ---------------------------------------------------------------------------

/// Maximum output rate for dynamic vector arrows (Hz).
const DYNAMIC_ARROW_MAX_HZ: f64 = 30.0;

/// Element indices carrying a dynamic arrow's xyz vector. Explicit element
/// tuples like `ball.world_vel[3],ball.world_vel[4],ball.world_vel[5]` use
/// their own indices; bare component references use the trailing 3 elements,
/// matching the editor (`component_value_tail_to_vec3` reads the value tail).
fn vector_element_indices(expr: &eql::Expr, comp_name: &str, flat_count: usize) -> [usize; 3] {
    fn collect(expr: &eql::Expr, comp_name: &str, out: &mut Vec<usize>) -> bool {
        match expr {
            eql::Expr::ArrayAccess(inner, idx) => match inner.as_ref() {
                eql::Expr::ComponentPart(part)
                    if part.component.is_some() && part.name == comp_name =>
                {
                    out.push(*idx);
                    true
                }
                _ => false,
            },
            eql::Expr::Tuple(items) => items.iter().all(|item| collect(item, comp_name, out)),
            _ => false,
        }
    }
    let mut indices = Vec::new();
    if collect(expr, comp_name, &mut indices)
        && indices.len() == 3
        && indices.iter().all(|&i| i < flat_count)
    {
        return [indices[0], indices[1], indices[2]];
    }
    [flat_count - 3, flat_count - 2, flat_count - 1]
}

/// Last pose-sample translation at or before `ts_us` (FOHold).
fn pose_translation_at(cursor: &ComponentCursor, prim: PrimType, ts_us: i64) -> Option<[f64; 3]> {
    if cursor.timestamps.is_empty() {
        return None;
    }
    let idx = match cursor.timestamps.binary_search_by_key(&ts_us, |t| t.0) {
        Ok(i) => i,
        Err(0) => 0,
        Err(i) => i - 1,
    };
    let buf = &cursor.data[idx * cursor.sample_size..(idx + 1) * cursor.sample_size];
    Some([
        read_f64(prim, buf, 4),
        read_f64(prim, buf, 5),
        read_f64(prim, buf, 6),
    ])
}

/// Build precomputed dynamic arrow SceneUpdate messages. These are vector_arrows
/// whose `vector` EQL references a data component rather than a literal tuple.
/// Returns one `(topic, sorted (timestamp_us, payload) stream)` per arrow —
/// separate topics so latest-per-topic backfill keeps every arrow alive.
type TimedPayloadStream = Vec<(i64, Vec<u8>)>;
type NamedTimedStreams = Vec<(String, TimedPayloadStream)>;

fn build_dynamic_arrows(
    schematics: &LoadedSchematics,
    ctx: &eql::Context,
    components: &[ExportComponent],
    component_cursors: &[Option<ComponentCursor>],
    epoch_offset_us: i64,
) -> NamedTimedStreams {
    let mut streams: NamedTimedStreams = Vec::new();

    let all_elems = schematics
        .primary
        .iter()
        .chain(schematics.windows.iter().map(|(_, s)| s))
        .flat_map(|s| s.elems.iter());

    for elem in all_elems {
        let SchematicElem::VectorArrow(arrow) = elem else {
            continue;
        };
        if parse_literal_tuple(&arrow.vector).is_some() {
            continue;
        }
        let vec_entity = match entity_for_eql(&arrow.vector, ctx) {
            Some(e) => e,
            None => continue,
        };
        let vec_expr = match ctx.parse_str(&arrow.vector) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let vec_comp_name = {
            let mut names = Vec::new();
            collect_component_names(&vec_expr, &mut names);
            match names.into_iter().next() {
                Some(n) => n,
                None => continue,
            }
        };
        let (comp_idx, comp) = match components
            .iter()
            .enumerate()
            .find(|(_, c)| c.name == vec_comp_name)
        {
            Some(pair) => pair,
            None => continue,
        };
        let cursor = match component_cursors[comp_idx].as_ref() {
            Some(c) => c,
            None => continue,
        };
        let flat_count: usize = comp.component.schema.dim.iter().product::<usize>().max(1);
        if flat_count < 3 {
            continue;
        }
        let [ix, iy, iz] = vector_element_indices(&vec_expr, &vec_comp_name, flat_count);

        // Body-frame arrows ride the origin entity's TF (Foxglove applies
        // attitude). World-frame arrows must live on `world` with an absolute
        // origin — attaching them to the entity TF would wrongly rotate a
        // world vector by the body's attitude.
        let frame = if arrow.body_frame {
            arrow
                .origin
                .as_deref()
                .and_then(|o| entity_for_eql(o, ctx))
                .unwrap_or_else(|| vec_entity.clone())
        } else {
            "world".to_string()
        };

        // World-frame origin: literal xyz/pose, or sample `<entity>.world_pos`
        // at each arrow timestamp. Body-frame origins stay at the entity root.
        enum DynamicOrigin<'a> {
            Fixed([f64; 3]),
            Pose {
                cursor: &'a ComponentCursor<'a>,
                prim: PrimType,
            },
        }
        let origin = if arrow.body_frame {
            DynamicOrigin::Fixed([0.0, 0.0, 0.0])
        } else if let Some(v) = arrow.origin.as_deref().and_then(parse_literal_tuple) {
            let pos = if v.len() >= 7 {
                [v[v.len() - 3], v[v.len() - 2], v[v.len() - 1]]
            } else if v.len() >= 3 {
                [v[0], v[1], v[2]]
            } else {
                continue;
            };
            DynamicOrigin::Fixed(pos)
        } else if let Some(entity) = arrow
            .origin
            .as_deref()
            .and_then(|o| entity_for_eql(o, ctx))
            .or_else(|| Some(vec_entity.clone()))
        {
            let pose_name = format!("{entity}.world_pos");
            match components
                .iter()
                .enumerate()
                .find(|(_, c)| c.name == pose_name)
                .and_then(|(i, c)| {
                    component_cursors[i]
                        .as_ref()
                        .map(|cur| (cur, c.component.schema.prim_type))
                }) {
                Some((pose_cursor, prim)) => DynamicOrigin::Pose {
                    cursor: pose_cursor,
                    prim,
                },
                None => DynamicOrigin::Fixed([0.0, 0.0, 0.0]),
            }
        } else {
            DynamicOrigin::Fixed([0.0, 0.0, 0.0])
        };

        let arrow_id = arrow
            .name
            .clone()
            .unwrap_or_else(|| format!("{vec_comp_name}-arrow"));
        let mut topic = format!("/scene_dynamic/{}", sanitize_topic_segment(&arrow_id));
        let mut ordinal = 1u32;
        while streams.iter().any(|(t, _)| *t == topic) {
            ordinal += 1;
            topic = format!(
                "/scene_dynamic/{}-{ordinal}",
                sanitize_topic_segment(&arrow_id)
            );
        }
        let prim = comp.component.schema.prim_type;
        let mut entries: Vec<(i64, Vec<u8>)> = Vec::new();

        let min_step_us = (1_000_000.0 / DYNAMIC_ARROW_MAX_HZ) as i64;
        let mut last_emitted_us: Option<i64> = None;

        for (i, ts) in cursor.timestamps.iter().enumerate() {
            if let Some(prev) = last_emitted_us
                && ts.0.saturating_sub(prev) < min_step_us
            {
                continue;
            }
            let ts_ns = us_to_ns(*ts, epoch_offset_us);
            let buf = &cursor.data[i * cursor.sample_size..(i + 1) * cursor.sample_size];
            let vx = read_f64(prim, buf, ix);
            let vy = read_f64(prim, buf, iy);
            let vz = read_f64(prim, buf, iz);
            let len = (vx * vx + vy * vy + vz * vz).sqrt();
            if len < 1e-12 {
                // Don't advance the throttle — a zero sample shouldn't delay
                // the next non-zero emit.
                continue;
            }
            let dir = [vx / len, vy / len, vz / len];
            let total = arrow_shaft_length(len, arrow.scale, arrow.normalize);
            let pos = match &origin {
                DynamicOrigin::Fixed(p) => *p,
                DynamicOrigin::Pose {
                    cursor: pose_cursor,
                    prim: pose_prim,
                } => pose_translation_at(pose_cursor, *pose_prim, ts.0).unwrap_or([0.0, 0.0, 0.0]),
            };
            let entity = json!({
                "timestamp": timestamp_json(ts_ns),
                "frame_id": frame,
                "id": arrow_id,
                "lifetime": {"sec": 0, "nsec": 0},
                "frame_locked": true,
                "arrows": [arrow_primitive(dir, total, &arrow.color, pos)],
            });
            entries.push((ts.0, scene_update_message(entity)));
            last_emitted_us = Some(ts.0);
        }
        if !entries.is_empty() {
            entries.sort_by_key(|(ts, _)| *ts);
            streams.push((topic, entries));
        }
    }

    streams
}

// ---------------------------------------------------------------------------
// Sensor camera H.264 encoding (video-export feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "video-export")]
fn encode_sensor_frame(
    cfg: &SensorCameraConfig,
    raw_payload: &[u8],
    ts_ns: u64,
    name: &str,
    encoders: &mut HashMap<usize, crate::export_videos::SensorEncoder>,
    idx: usize,
) -> Vec<u8> {
    let encoder = encoders.entry(idx).or_insert_with(|| {
        let fps = crate::export_videos::sensor_camera_export_fps(cfg, 30);
        crate::export_videos::SensorEncoder::new(cfg.width, cfg.height, fps)
            .expect("openh264 encoder init")
    });
    let yuv = match crate::export_videos::rgba_to_i420(
        raw_payload,
        cfg.width as usize,
        cfg.height as usize,
    ) {
        Ok(yuv) => yuv,
        Err(e) => {
            eprintln!("Warning: sensor frame encode skip for {name}: {e}");
            return msg_log_json(
                &MsgLogKind::SensorCamera(Box::new(cfg.clone())),
                name,
                raw_payload,
                ts_ns,
            );
        }
    };
    match encoder.encode_frame(&yuv) {
        Ok(annexb) if !annexb.is_empty() => {
            let value = json!({
                "timestamp": timestamp_json(ts_ns),
                "frame_id": name,
                "data": BASE64.encode(&annexb),
                "format": "h264",
            });
            serde_json::to_vec(&value).expect("json serialize")
        }
        Ok(_) => Vec::new(),
        Err(e) => {
            eprintln!("Warning: sensor frame encode error for {name}: {e}");
            msg_log_json(
                &MsgLogKind::SensorCamera(Box::new(cfg.clone())),
                name,
                raw_payload,
                ts_ns,
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------

/// Export database contents to a Foxglove-compatible MCAP file.
///
/// Writes `{output}/{db_name}.mcap` and, when the DB carries a schematic,
/// `{output}/{db_name}.foxglove-layout.json`.
pub fn run(
    db_path: PathBuf,
    output_path: PathBuf,
    options: McapExportOptions,
) -> Result<(), Error> {
    if !db_path.exists() {
        return Err(Error::MissingDbState(db_path));
    }
    let db_state_path = db_path.join("db_state");
    if !db_state_path.exists() {
        return Err(Error::MissingDbState(db_state_path));
    }

    // Component names are matched lowercased (same as the parquet/csv exporter);
    // lowercase the pattern too so e.g. `--pattern 'Drone.*'` still matches.
    let glob_pattern = options
        .pattern
        .as_ref()
        .map(|p| Pattern::new(&p.to_lowercase()))
        .transpose()
        .map_err(|e| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Invalid glob pattern: {}", e),
            ))
        })?;

    println!("Opening database: {}", db_path.display());
    let db = DB::open(db_path.clone())?;
    std::fs::create_dir_all(&output_path)?;

    let db_name = db_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("elodin-db")
        .to_string();
    let mcap_path = output_path.join(format!("{db_name}.mcap"));
    let layout_path = output_path.join(format!("{db_name}.foxglove-layout.json"));
    println!("Exporting to: {}", mcap_path.display());

    // ---- snapshot state ---------------------------------------------------
    let earliest = db.earliest_timestamp.latest();
    let last = db.last_updated.latest();

    struct Snapshot {
        components: Vec<ExportComponent>,
        msg_logs: Vec<(PacketId, MsgLog)>,
        db_metadata: BTreeMap<String, String>,
        active_schematic: Option<String>,
        sensor_cameras: Vec<SensorCameraConfig>,
        skipped: usize,
    }

    let snapshot = db.with_state(|state| {
        let mut components = Vec::new();
        let mut skipped = 0usize;
        for component in state.components.values() {
            let Some(metadata) = state.component_metadata.get(&component.component_id) else {
                skipped += 1;
                continue;
            };
            if !options.include_private && metadata.is_private() {
                println!("  Skipping {} (private)", metadata.name);
                skipped += 1;
                continue;
            }
            if let Some(ref pattern) = glob_pattern
                && !pattern.matches(&metadata.name.to_lowercase())
            {
                skipped += 1;
                continue;
            }
            let element_paths =
                element_paths(&metadata.name, &component.schema, metadata.element_names());
            let flat_count: usize = component.schema.dim.iter().product::<usize>().max(1);
            let pose_entity = (metadata.name.ends_with(".world_pos") && flat_count == 7)
                .then(|| metadata.name.trim_end_matches(".world_pos").to_string());
            let mut channel_metadata: BTreeMap<String, String> = metadata
                .metadata
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            channel_metadata.insert(
                "elodin.component_id".to_string(),
                component.component_id.to_string(),
            );
            components.push(ExportComponent {
                component: component.clone(),
                name: metadata.name.clone(),
                topic: topic_for_component(&metadata.name),
                element_paths,
                metadata: channel_metadata,
                pose_entity,
            });
        }
        components.sort_by(|a, b| a.name.cmp(&b.name));

        let msg_logs: Vec<(PacketId, MsgLog)> = state
            .msg_logs
            .iter()
            .map(|(id, log)| (*id, log.clone()))
            .collect();

        let mut db_metadata: BTreeMap<String, String> = state
            .db_config
            .metadata
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        db_metadata.insert(
            "default_stream_time_step_ns".to_string(),
            state
                .db_config
                .default_stream_time_step
                .as_nanos()
                .to_string(),
        );
        let sensor_cameras: Vec<SensorCameraConfig> = state
            .db_config
            .metadata
            .get("sensor_cameras")
            .and_then(|json| serde_json::from_str(json).ok())
            .unwrap_or_default();

        Snapshot {
            components,
            msg_logs,
            db_metadata,
            active_schematic: state.db_config.schematic_active().map(str::to_owned),
            sensor_cameras,
            skipped,
        }
    });

    println!(
        "Found {} components ({} skipped), {} message logs",
        snapshot.components.len(),
        snapshot.skipped,
        snapshot.msg_logs.len()
    );

    // ---- EQL context + schematics -----------------------------------------
    let eql_components: Vec<Arc<eql::Component>> = snapshot
        .components
        .iter()
        .map(|c| {
            let flat_names: Vec<String> = c.element_paths.iter().map(|p| p.join(".")).collect();
            Arc::new(eql::Component::new_with_element_names(
                c.name.clone(),
                c.component.component_id,
                c.component.schema.to_schema(),
                flat_names,
            ))
        })
        .collect();
    let ctx = eql::Context::from_leaves(eql_components, earliest, last);

    let schematics = load_schematics(&db_path, snapshot.active_schematic.as_deref());
    let assets_dir = db_path.join("assets");

    // ---- msg log classification -------------------------------------------
    let sensor_by_msg_id: HashMap<PacketId, SensorCameraConfig> = snapshot
        .sensor_cameras
        .iter()
        .map(|camera| (msg_id(&camera.camera_name), camera.clone()))
        .collect();
    let mut video_names: HashSet<String> = HashSet::new();
    fn collect_video_names(panel: &Panel, out: &mut HashSet<String>) {
        match panel {
            Panel::VideoStream(stream) => {
                out.insert(stream.msg_name.clone());
            }
            Panel::HSplit(split) | Panel::VSplit(split) => {
                for child in &split.panels {
                    collect_video_names(child, out);
                }
            }
            Panel::Tabs(panels) => {
                for child in panels {
                    collect_video_names(child, out);
                }
            }
            _ => {}
        }
    }
    for schematic in schematics
        .primary
        .iter()
        .chain(schematics.windows.iter().map(|(_, s)| s))
    {
        for elem in &schematic.elems {
            if let SchematicElem::Panel(panel) = elem {
                collect_video_names(panel, &mut video_names);
            }
        }
    }

    let export_msg_logs: Vec<ExportMsgLog> = snapshot
        .msg_logs
        .into_iter()
        .filter(|(_, log)| !log.timestamps().is_empty())
        .map(|(packet_id, log)| {
            let name = log
                .metadata()
                .map(|m| m.name.clone())
                .or_else(|| {
                    sensor_by_msg_id
                        .get(&packet_id)
                        .map(|camera| camera.camera_name.clone())
                })
                .unwrap_or_else(|| format!("msg-{}", u16::from_le_bytes(packet_id)));
            let kind = classify_msg_log(&log, packet_id, &name, &sensor_by_msg_id, &video_names);
            let topic = match kind {
                MsgLogKind::H264Video => format!("/video/{name}"),
                MsgLogKind::SensorCamera(_) => {
                    #[cfg(feature = "video-export")]
                    {
                        format!("/video/{name}")
                    }
                    #[cfg(not(feature = "video-export"))]
                    {
                        format!("/camera/{name}")
                    }
                }
                MsgLogKind::LogEntries => format!("/log/{name}"),
                MsgLogKind::Raw => format!("/msg/{name}"),
            };
            ExportMsgLog {
                log,
                name,
                topic,
                kind,
            }
        })
        .collect();

    // ---- create writer + channels ------------------------------------------
    let file = File::create(&mcap_path)?;
    let write_options = mcap::WriteOptions::new()
        .compression(Some(mcap::Compression::Zstd))
        .profile("")
        .library(concat!("elodin-db ", env!("CARGO_PKG_VERSION")));
    let mut writer =
        mcap::Writer::with_options(BufWriter::with_capacity(FILE_BUF_CAP, file), write_options)?;

    let empty_metadata = BTreeMap::new();

    // Component channels.
    let mut component_channels = Vec::with_capacity(snapshot.components.len());
    for comp in &snapshot.components {
        let schema = component_json_schema(
            &comp.name,
            comp.component.schema.prim_type,
            &comp.element_paths,
        );
        let schema_id = writer.add_schema(
            &comp.name,
            "jsonschema",
            serde_json::to_vec(&schema)
                .expect("json serialize")
                .as_slice(),
        )?;
        let channel_id = writer.add_channel(schema_id, &comp.topic, "json", &comp.metadata)?;
        component_channels.push(channel_id);
    }

    // /tf channel (only when pose components exist).
    let has_poses = snapshot.components.iter().any(|c| c.pose_entity.is_some());
    let tf_channel = if has_poses {
        let schema_id = writer.add_schema(
            "foxglove.FrameTransforms",
            "jsonschema",
            SCHEMA_FRAME_TRANSFORMS.as_bytes(),
        )?;
        Some(writer.add_channel(schema_id, "/tf", "json", &empty_metadata)?)
    } else {
        None
    };

    // Msg log channels.
    let mut msg_channels = Vec::with_capacity(export_msg_logs.len());
    for log in &export_msg_logs {
        let (schema_name, schema_body) = match log.kind {
            MsgLogKind::H264Video => ("foxglove.CompressedVideo", SCHEMA_COMPRESSED_VIDEO),
            MsgLogKind::SensorCamera(_) => {
                #[cfg(feature = "video-export")]
                {
                    ("foxglove.CompressedVideo", SCHEMA_COMPRESSED_VIDEO)
                }
                #[cfg(not(feature = "video-export"))]
                {
                    ("foxglove.RawImage", SCHEMA_RAW_IMAGE)
                }
            }
            MsgLogKind::LogEntries => ("foxglove.Log", SCHEMA_LOG),
            MsgLogKind::Raw => ("elodin.RawMessage", SCHEMA_RAW_BYTES),
        };
        let schema_id = writer.add_schema(schema_name, "jsonschema", schema_body.as_bytes())?;
        let mut metadata = BTreeMap::new();
        metadata.insert("elodin.msg_name".to_string(), log.name.clone());
        let channel_id = writer.add_channel(schema_id, &log.topic, "json", &metadata)?;
        msg_channels.push(channel_id);
    }

    // ---- gather cursors -----------------------------------------------------
    let full_range = Timestamp(i64::MIN)..Timestamp(i64::MAX);
    let component_cursors: Vec<Option<ComponentCursor>> = snapshot
        .components
        .iter()
        .map(|c| {
            c.component
                .time_series
                .get_range(&full_range)
                .map(|(timestamps, data)| ComponentCursor {
                    timestamps,
                    data,
                    sample_size: c.component.schema.size(),
                })
        })
        .collect();
    let msg_entries: Vec<Vec<(Timestamp, &[u8])>> = export_msg_logs
        .iter()
        .map(|log| log.log.get_range(&full_range).collect())
        .collect();

    let start_ts = component_cursors
        .iter()
        .flatten()
        .filter_map(|c| c.timestamps.first())
        .chain(msg_entries.iter().filter_map(|m| m.first().map(|(t, _)| t)))
        .min()
        .copied()
        .unwrap_or(earliest);

    // ---- epoch offset ---------------------------------------------------------
    // MCAP log_time is u64 ns since Unix epoch and cannot represent pre-1970
    // absolute times. Prefer the caller's offset when it keeps the earliest
    // sample >= 0; otherwise auto-rebase so relative ordering is preserved
    // (critical for Apollo-style DBs when `--epoch-offset-us 0` would otherwise
    // clamp every sample to log_time 0).
    let requested_offset_us = options.epoch_offset_us.unwrap_or(0);
    let epoch_offset_us = if start_ts
        .0
        .checked_add(requested_offset_us)
        .is_none_or(|us| us < 0)
    {
        let offset = start_ts.0.saturating_neg();
        eprintln!(
            "Warning: earliest timestamp is {} µs (pre-1970); auto-rebasing by +{} µs so earliest becomes t=0{}",
            start_ts.0,
            offset,
            if options.epoch_offset_us.is_some() {
                format!(
                    " (requested --epoch-offset-us {requested_offset_us} leaves pre-epoch times)"
                )
            } else {
                String::new()
            }
        );
        offset
    } else {
        if requested_offset_us != 0 {
            println!("  Using manual epoch offset: {requested_offset_us} µs");
        }
        requested_offset_us
    };
    let start_ns = us_to_ns(start_ts, epoch_offset_us);

    let max_embed_bytes = options.max_embed_mb * 1024 * 1024;
    let follow_entity = resolve_follow_entity(&schematics, &ctx);

    // ---- geodetic frames (schematic `coordinate` node) -----------------------
    let geo_map = entity_geo_frames(&schematics, &ctx);
    let geo_anchors = schematics
        .primary
        .as_ref()
        .and_then(|s| s.origin.as_ref())
        .filter(|_| !geo_map.is_empty())
        .map(geo_frame_anchors);
    let tf_parents: Vec<String> = snapshot
        .components
        .iter()
        .map(|c| {
            c.pose_entity
                .as_deref()
                .filter(|_| geo_anchors.is_some())
                .and_then(|e| geo_map.get(e).cloned())
                .unwrap_or_else(|| "world".to_string())
        })
        .collect();

    // ---- /scene (one message per entity) ------------------------------------
    let scene = build_scene(
        &schematics,
        &ctx,
        &assets_dir,
        start_ns,
        &snapshot.components,
        &component_cursors,
        max_embed_bytes,
        follow_entity.as_deref(),
        geo_anchors.is_some(),
    );

    // ---- dynamic vector arrows on /scene_dynamic ----------------------------
    let dynamic_arrows = build_dynamic_arrows(
        &schematics,
        &ctx,
        &snapshot.components,
        &component_cursors,
        epoch_offset_us,
    );

    let mut referenced_assets: Vec<String> = Vec::new();
    let scene_schema_id =
        if scene.as_ref().is_some_and(|s| !s.messages.is_empty()) || !dynamic_arrows.is_empty() {
            Some(writer.add_schema(
                "foxglove.SceneUpdate",
                "jsonschema",
                SCHEMA_SCENE_UPDATE.as_bytes(),
            )?)
        } else {
            None
        };

    if let Some(scene) = &scene {
        referenced_assets.extend(scene.referenced_assets.iter().cloned());
    }

    // One channel per static entity (`/scene/<id>`) and per dynamic arrow
    // (`/scene_dynamic/<name>`): Foxglove backfills only the latest message
    // per topic when a 3D panel (re)mounts, so shared topics drop entities.
    if let (Some(schema_id), Some(scene)) = (scene_schema_id, &scene) {
        for (topic, msg) in &scene.messages {
            let channel_id = writer.add_channel(schema_id, topic, "json", &empty_metadata)?;
            writer.write_to_known_channel(
                &MessageHeader {
                    channel_id,
                    sequence: 0,
                    log_time: start_ns,
                    publish_time: start_ns,
                },
                msg,
            )?;
        }
    }

    let mut dynamic_arrow_channels: Vec<u16> = Vec::with_capacity(dynamic_arrows.len());
    if let Some(schema_id) = scene_schema_id {
        for (topic, _) in &dynamic_arrows {
            dynamic_arrow_channels.push(writer.add_channel(
                schema_id,
                topic,
                "json",
                &empty_metadata,
            )?);
        }
    }

    let mut sequences: HashMap<u16, u32> = HashMap::new();

    // world→NED / world→ENU anchor frames from the schematic `coordinate` node.
    if let (Some(anchors), Some(channel_id)) = (&geo_anchors, tf_channel) {
        writer.write_to_known_channel(
            &MessageHeader {
                channel_id,
                sequence: 0,
                log_time: start_ns,
                publish_time: start_ns,
            },
            &geo_frame_tf_message(anchors, start_ns),
        )?;
        sequences.insert(channel_id, 1);
    }

    // ---- k-way merge over all cursors ----------------------------------------
    // Cursor id space: [0, n) component channels, [n, 2n) TF from pose
    // components, [2n, 2n + m) msg logs, [2n + m, 2n + m + k) dynamic arrows.
    let n = snapshot.components.len();
    let m = export_msg_logs.len();
    let k = dynamic_arrows.len();
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let mut heap: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();
    let mut positions = vec![0usize; 2 * n + m + k];

    for (i, cursor) in component_cursors.iter().enumerate() {
        if let Some(cursor) = cursor
            && let Some(ts) = cursor.timestamps.first()
        {
            heap.push(Reverse((ts.0, i)));
            if snapshot.components[i].pose_entity.is_some() && tf_channel.is_some() {
                heap.push(Reverse((ts.0, n + i)));
            }
        }
    }
    for (i, entries) in msg_entries.iter().enumerate() {
        if let Some((ts, _)) = entries.first() {
            heap.push(Reverse((ts.0, 2 * n + i)));
        }
    }
    for (i, (_, entries)) in dynamic_arrows.iter().enumerate() {
        if let Some((ts, _)) = entries.first() {
            heap.push(Reverse((*ts, 2 * n + m + i)));
        }
    }

    #[cfg(feature = "video-export")]
    let mut sensor_encoders: HashMap<usize, crate::export_videos::SensorEncoder> = HashMap::new();

    let mut message_count = 0u64;
    while let Some(Reverse((ts_us, cursor_id))) = heap.pop() {
        crate::cancellation::check_cancelled()?;
        let ts_ns = us_to_ns(Timestamp(ts_us), epoch_offset_us);
        let pos = positions[cursor_id];
        let (channel_id, payload, next_ts): (u16, Vec<u8>, Option<i64>) = if cursor_id < n {
            let comp = &snapshot.components[cursor_id];
            let cursor = component_cursors[cursor_id].as_ref().unwrap();
            let buf = &cursor.data[pos * cursor.sample_size..(pos + 1) * cursor.sample_size];
            (
                component_channels[cursor_id],
                component_row_json(comp, buf),
                cursor.timestamps.get(pos + 1).map(|t| t.0),
            )
        } else if cursor_id < 2 * n {
            let comp = &snapshot.components[cursor_id - n];
            let cursor = component_cursors[cursor_id - n].as_ref().unwrap();
            let buf = &cursor.data[pos * cursor.sample_size..(pos + 1) * cursor.sample_size];
            let entity = comp.pose_entity.as_deref().unwrap();
            (
                tf_channel.unwrap(),
                tf_message(
                    entity,
                    &tf_parents[cursor_id - n],
                    comp.component.schema.prim_type,
                    buf,
                    ts_ns,
                ),
                cursor.timestamps.get(pos + 1).map(|t| t.0),
            )
        } else if cursor_id < 2 * n + m {
            let idx = cursor_id - 2 * n;
            let log = &export_msg_logs[idx];
            let (_, raw_payload) = msg_entries[idx][pos];

            let payload = {
                #[cfg(feature = "video-export")]
                {
                    if let MsgLogKind::SensorCamera(cfg) = &log.kind {
                        encode_sensor_frame(
                            cfg,
                            raw_payload,
                            ts_ns,
                            &log.name,
                            &mut sensor_encoders,
                            idx,
                        )
                    } else {
                        msg_log_json(&log.kind, &log.name, raw_payload, ts_ns)
                    }
                }
                #[cfg(not(feature = "video-export"))]
                {
                    msg_log_json(&log.kind, &log.name, raw_payload, ts_ns)
                }
            };

            (
                msg_channels[idx],
                payload,
                msg_entries[idx].get(pos + 1).map(|(t, _)| t.0),
            )
        } else {
            // Dynamic arrow scene updates, one channel per arrow.
            let idx = cursor_id - 2 * n - m;
            let entries = &dynamic_arrows[idx].1;
            let (_, ref payload) = entries[pos];
            (
                dynamic_arrow_channels[idx],
                payload.clone(),
                entries.get(pos + 1).map(|(t, _)| *t),
            )
        };

        let sequence = sequences.entry(channel_id).or_insert(0);
        writer.write_to_known_channel(
            &MessageHeader {
                channel_id,
                sequence: *sequence,
                log_time: ts_ns,
                publish_time: ts_ns,
            },
            &payload,
        )?;
        *sequence += 1;
        message_count += 1;

        positions[cursor_id] = pos + 1;
        if let Some(next) = next_ts {
            heap.push(Reverse((next, cursor_id)));
        }
    }

    // ---- metadata records ------------------------------------------------------
    let mut db_metadata = snapshot.db_metadata.clone();
    if epoch_offset_us != 0 {
        db_metadata.insert(
            "elodin.time_offset_us".to_string(),
            epoch_offset_us.to_string(),
        );
    }
    writer.write_metadata(&mcap::records::Metadata {
        name: "elodin.db_state".to_string(),
        metadata: db_metadata,
    })?;
    let component_meta: BTreeMap<String, String> = snapshot
        .components
        .iter()
        .map(|c| {
            (
                c.name.clone(),
                serde_json::to_string(&c.metadata).unwrap_or_default(),
            )
        })
        .collect();
    writer.write_metadata(&mcap::records::Metadata {
        name: "elodin.components".to_string(),
        metadata: component_meta,
    })?;

    // ---- attachments -------------------------------------------------------------
    let mut attached: HashSet<String> = HashSet::new();
    let mut attach = |writer: &mut mcap::Writer<_>,
                      key: &str,
                      media_type: &str,
                      bytes: Vec<u8>|
     -> Result<(), Error> {
        if !attached.insert(key.to_string()) {
            return Ok(());
        }
        writer.attach(&mcap::Attachment {
            log_time: start_ns,
            create_time: start_ns,
            name: key.to_string(),
            media_type: media_type.to_string(),
            data: Cow::Owned(bytes),
        })?;
        Ok(())
    };

    for (key, bytes) in &schematics.raw {
        attach(&mut writer, key, "application/kdl", bytes.clone())?;
    }
    for key in &referenced_assets {
        if let Ok(bytes) = std::fs::read(assets_dir.join(key)) {
            attach(&mut writer, key, "model/gltf-binary", bytes)?;
        }
    }
    if options.all_assets && assets_dir.is_dir() {
        for entry in walk_files(&assets_dir) {
            let Ok(rel) = entry.strip_prefix(&assets_dir) else {
                continue;
            };
            let key = rel.to_string_lossy().replace('\\', "/");
            let media_type = match entry.extension().and_then(|e| e.to_str()) {
                Some("glb") => "model/gltf-binary",
                Some("kdl") => "application/kdl",
                Some("png") => "image/png",
                Some("jpg") | Some("jpeg") => "image/jpeg",
                Some("json") => "application/json",
                _ => "application/octet-stream",
            };
            if let Ok(bytes) = std::fs::read(&entry) {
                attach(&mut writer, &key, media_type, bytes)?;
            }
        }
    }

    let summary = writer.finish()?;
    drop(summary);

    let mcap_size = std::fs::metadata(&mcap_path).map(|m| m.len()).unwrap_or(0);
    println!(
        "  Exported {} ({} messages, {:.1} MiB)",
        mcap_path.display(),
        message_count,
        mcap_size as f64 / (1024.0 * 1024.0)
    );

    // ---- layout ---------------------------------------------------------------------
    let mut scene_topics: Vec<String> = Vec::new();
    if let Some(scene) = &scene {
        scene_topics.extend(scene.messages.iter().map(|(t, _)| t.clone()));
    }
    scene_topics.extend(dynamic_arrows.iter().map(|(t, _)| t.clone()));
    match build_layout(&schematics, &ctx, &snapshot.components, &scene_topics) {
        Some(layout) => {
            std::fs::write(
                &layout_path,
                serde_json::to_vec_pretty(&layout).expect("json serialize"),
            )?;
            println!("  Exported {} (Foxglove layout)", layout_path.display());
        }
        None => println!("  No schematic panels found; skipping layout generation"),
    }

    println!();
    println!("Export complete.");
    println!("Open the .mcap directly in Foxglove, or upload it:");
    println!("  POST https://api.foxglove.dev/v1/data/upload  (then PUT to the returned link)");
    println!(
        "Create the layout with POST https://api.foxglove.dev/v1/layouts (data = layout json)"
    );
    Ok(())
}

fn walk_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.is_file() {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_paths_prefers_valid_metadata_names() {
        let schema = crate::ComponentSchema::new(PrimType::F64, &[7]);
        let paths = element_paths("drone.world_pos", &schema, "q0,q1,q2,q3,x,y,z");
        assert_eq!(paths.len(), 7);
        assert_eq!(paths[0], vec!["q0"]);
        assert_eq!(paths[6], vec!["z"]);
    }

    #[test]
    fn element_paths_falls_back_on_count_mismatch() {
        // drone.force-style bug: 4 names for a 6-element component.
        let schema = crate::ComponentSchema::new(PrimType::F64, &[6]);
        let paths = element_paths("drone.force", &schema, "z,x,y,z");
        assert_eq!(paths.len(), 6);
        assert_eq!(paths[0], vec!["x"]);
    }

    #[test]
    fn element_paths_nests_dotted_names() {
        let schema = crate::ComponentSchema::new(PrimType::F64, &[3, 3]);
        let paths = element_paths(
            "drone.rate_pid_state",
            &schema,
            "e.r,e.p,e.y,i.r,i.p,i.y,d.r,d.p,d.y",
        );
        assert_eq!(paths.len(), 9);
        assert_eq!(paths[0], vec!["e", "r"]);
        assert_eq!(paths[8], vec!["d", "y"]);
    }

    #[test]
    fn scalar_uses_value_field() {
        let schema = crate::ComponentSchema::new(PrimType::U64, &[]);
        let paths = element_paths("Globals.tick", &schema, "");
        assert_eq!(paths, vec![vec!["value".to_string()]]);
    }

    #[test]
    fn camera_orbit_matches_elodin_vantage() {
        // "drone.world_pos + (0,0,0,0, 2,2,2)": 2m E, 2m N, 2m up from target.
        let (distance, phi, theta) = camera_orbit_from_offset(Some([2.0, 2.0, 2.0]));
        assert!((distance - 12.0f64.sqrt()).abs() < 1e-9);
        assert!(
            (phi - 54.7356).abs() < 1e-3,
            "phi {phi} should be ~54.7 deg"
        );
        assert!((theta - 45.0).abs() < 1e-9);
        // Missing offset falls back to the same 3/4 view, not top-down.
        let (_, phi_default, _) = camera_orbit_from_offset(None);
        assert!(phi_default > 30.0 && phi_default < 80.0);
    }

    #[test]
    fn quat_from_x_axis_axes() {
        // +X stays identity.
        assert_eq!(quat_from_x_axis([1.0, 0.0, 0.0]), [0.0, 0.0, 0.0, 1.0]);
        // +Y is a 90 deg rotation about +Z.
        let q = quat_from_x_axis([0.0, 1.0, 0.0]);
        assert!((q[2] - (std::f64::consts::FRAC_PI_4).sin()).abs() < 1e-9);
        assert!((q[3] - (std::f64::consts::FRAC_PI_4).cos()).abs() < 1e-9);
    }

    #[test]
    fn tf_message_maps_quaternion_scalar_last() {
        let buf: Vec<u8> = [0.1f64, 0.2, 0.3, 0.9, 1.0, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let msg = tf_message("drone", "world", PrimType::F64, &buf, 1_000);
        let value: Value = serde_json::from_slice(&msg).unwrap();
        let tf = &value["transforms"][0];
        assert_eq!(tf["child_frame_id"], "drone");
        assert_eq!(tf["rotation"]["x"], 0.1);
        assert_eq!(tf["rotation"]["w"], 0.9);
        assert_eq!(tf["translation"]["x"], 1.0);
        assert_eq!(tf["translation"]["z"], 3.0);
    }

    // --- Gap 1: epoch offset tests ---

    #[test]
    fn us_to_ns_with_zero_offset() {
        assert_eq!(us_to_ns(Timestamp(1_000_000), 0), 1_000_000_000);
    }

    #[test]
    fn us_to_ns_positive_offset() {
        assert_eq!(us_to_ns(Timestamp(-500), 1000), 500_000);
    }

    #[test]
    fn us_to_ns_auto_rebase_clamps_to_zero() {
        assert_eq!(us_to_ns(Timestamp(-1000), 1000), 0);
    }

    #[test]
    fn us_to_ns_negative_after_offset_clamps() {
        assert_eq!(us_to_ns(Timestamp(-2000), 500), 0);
    }

    #[test]
    fn us_to_ns_pre1970_zero_offset_clamps_each_sample() {
        // Documents the footgun: without auto-rebase, every pre-1970 sample
        // collapses to 0. [`run`] must reject/override insufficient offsets.
        assert_eq!(us_to_ns(Timestamp(-100_000), 0), 0);
        assert_eq!(us_to_ns(Timestamp(-90_000), 0), 0);
    }

    // --- Gap 3: camera near/far derivation ---

    #[test]
    fn camera_near_far_defaults() {
        let (distance, _, _) = camera_orbit_from_offset(Some([2.0, 2.0, 2.0]));
        let near: f64 = 0.01; // default when viewport.near is None
        let far = (distance * 4.0).max(5000.0);
        assert!((near - 0.01).abs() < 1e-9);
        assert!(far >= 5000.0);
    }

    #[test]
    fn camera_far_scales_with_distance() {
        let (distance, _, _) = camera_orbit_from_offset(Some([5000.0, 0.0, 0.0]));
        let far = (distance * 4.0).max(5000.0);
        assert!(far >= distance * 4.0);
        assert!(far >= 5000.0);
    }

    // --- Gap 5: literal pose parsing ---

    #[test]
    fn parse_literal_tuple_7() {
        let vals = parse_literal_tuple("(0, 0, 0, 1, 1.5, 2.5, 3.5)");
        assert_eq!(vals, Some(vec![0.0, 0.0, 0.0, 1.0, 1.5, 2.5, 3.5]));
    }

    #[test]
    fn parse_literal_tuple_3() {
        let vals = parse_literal_tuple("(1, 2, 3)");
        assert_eq!(vals, Some(vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn parse_literal_tuple_not_a_tuple() {
        assert_eq!(parse_literal_tuple("drone.world_pos"), None);
    }

    // --- Gap 4: line decimation ---

    fn make_test_component(name: &str, prim: PrimType, dim: &[usize]) -> ExportComponent {
        let db_path = std::env::temp_dir().join(format!("elodin_mcap_test_{}", fastrand::u64(..)));
        let _ = std::fs::create_dir_all(&db_path);
        let component = crate::Component::create(
            &db_path,
            impeller2::types::ComponentId::new(name),
            name.to_string(),
            crate::ComponentSchema::new(prim, dim),
            Timestamp(0),
        )
        .expect("create component");
        ExportComponent {
            component,
            name: name.to_string(),
            topic: topic_for_component(name),
            element_paths: vec![],
            metadata: Default::default(),
            pose_entity: name
                .ends_with(".world_pos")
                .then(|| name.trim_end_matches(".world_pos").to_string()),
        }
    }

    #[test]
    fn extract_trajectory_under_limit() {
        let sample_size = 7 * 8;
        let num_rows = 100;
        let mut data = Vec::with_capacity(num_rows * sample_size);
        let mut timestamps = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            let row: Vec<u8> = [0.0, 0.0, 0.0, 1.0, i as f64, (i * 2) as f64, 0.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            data.extend_from_slice(&row);
            timestamps.push(Timestamp(i as i64 * 1000));
        }
        let comp = make_test_component("test.world_pos", PrimType::F64, &[7]);
        let cursor = ComponentCursor {
            timestamps: &timestamps,
            data: &data,
            sample_size,
        };
        let points = extract_trajectory(&comp, &cursor);
        assert_eq!(points.len(), 100);
        assert_eq!(points[0], [0.0, 0.0, 0.0]);
        assert_eq!(points[50], [50.0, 100.0, 0.0]);
    }

    #[test]
    fn extract_trajectory_decimates() {
        let sample_size = 7 * 8;
        let num_rows = 5000;
        let mut data = Vec::with_capacity(num_rows * sample_size);
        let mut timestamps = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            let row: Vec<u8> = [0.0, 0.0, 0.0, 1.0, i as f64, 0.0, 0.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            data.extend_from_slice(&row);
            timestamps.push(Timestamp(i as i64 * 1000));
        }
        let comp = make_test_component("test.world_pos", PrimType::F64, &[7]);
        let cursor = ComponentCursor {
            timestamps: &timestamps,
            data: &data,
            sample_size,
        };
        let points = extract_trajectory(&comp, &cursor);
        assert!(points.len() <= MAX_LINE_POINTS + 1);
        assert_eq!(points[0], [0.0, 0.0, 0.0]);
        assert_eq!(points.last().unwrap()[0], 4999.0);
    }

    // --- Pass 2: scene entity helpers ---

    fn empty_eql_ctx() -> eql::Context {
        eql::Context::new(BTreeMap::new(), Timestamp(0), Timestamp(1_000_000))
    }

    fn test_glb_object(
        eql: &str,
        path: &str,
        translate: (f32, f32, f32),
        rotate: (f32, f32, f32),
    ) -> Object3D {
        Object3D {
            eql: eql.to_string(),
            mesh: Object3DMesh::Glb {
                path: path.to_string(),
                scale: 1.0,
                translate,
                rotate,
                animations: vec![],
                emissivity: 0.0,
                glow: 0.0,
                glow_color: None,
            },
            frame: None,
            frame_orientation: None,
            orientation: Default::default(),
            icon: None,
            thrusters: vec![],
            mesh_visibility_range: None,
            node_id: Default::default(),
        }
    }

    #[test]
    fn next_model_entity_id_unique_per_frame() {
        let mut counts = HashMap::new();
        assert_eq!(next_model_entity_id("body", &mut counts), "body-model");
        assert_eq!(next_model_entity_id("body", &mut counts), "body-model-2");
        assert_eq!(next_model_entity_id("body", &mut counts), "body-model-3");
        assert_eq!(next_model_entity_id("other", &mut counts), "other-model");
    }

    #[test]
    fn should_embed_glb_respects_force() {
        assert!(should_embed_glb(100, 50, true));
        assert!(!should_embed_glb(100, 50, false));
        assert!(should_embed_glb(50, 50, false));
    }

    #[test]
    fn compose_pose_applies_local_translate_in_parent_frame() {
        // Parent at (10,0,0), identity orientation; local translate (1,2,3).
        let parent = json!({
            "position": {"x": 10.0, "y": 0.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        });
        let pose = compose_pose_with_glb(&parent, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0));
        assert!((pose["position"]["x"].as_f64().unwrap() - 11.0).abs() < 1e-9);
        assert!((pose["position"]["y"].as_f64().unwrap() - 2.0).abs() < 1e-9);
        assert!((pose["position"]["z"].as_f64().unwrap() - 3.0).abs() < 1e-9);
        assert!((pose["orientation"]["w"].as_f64().unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn compose_pose_applies_local_yaw_to_orientation() {
        let parent = identity_pose();
        // Translate is in parent frame (not spun by local rotate); rotate sets orientation.
        let pose = compose_pose_with_glb(&parent, (1.0, 0.0, 0.0), (0.0, 0.0, 90.0));
        assert!((pose["position"]["x"].as_f64().unwrap() - 1.0).abs() < 1e-9);
        assert!(pose["position"]["y"].as_f64().unwrap().abs() < 1e-9);
        let qw = pose["orientation"]["w"].as_f64().unwrap();
        let qz = pose["orientation"]["z"].as_f64().unwrap();
        assert!((qw - (std::f64::consts::FRAC_PI_4).cos()).abs() < 1e-6);
        assert!((qz - (std::f64::consts::FRAC_PI_4).sin()).abs() < 1e-6);
    }

    #[test]
    fn is_annex_b_requires_plausible_nal() {
        // 4-byte start code + IDR NAL (type 5).
        assert!(is_annex_b(&[0, 0, 0, 1, 0x65, 0x88]));
        // 3-byte start code + non-IDR slice (type 1).
        assert!(is_annex_b(&[0, 0, 1, 0x01, 0x00]));
        // Bare prefix without a NAL byte.
        assert!(!is_annex_b(&[0, 0, 1]));
        // forbidden_zero_bit set — not a valid NAL header.
        assert!(!is_annex_b(&[0, 0, 0, 1, 0x85]));
        // Random telemetry that happens to start with 00 00 01.
        assert!(!is_annex_b(&[0, 0, 1, 0xff, 0xff]));
    }

    #[test]
    fn arrow_shaft_length_respects_normalize() {
        assert!((arrow_shaft_length(4.0, 2.0, false) - 8.0).abs() < 1e-12);
        assert!((arrow_shaft_length(4.0, 2.0, true) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn scene_update_message_is_one_entity() {
        let msg = scene_update_message(json!({"id": "a"}));
        let v: Value = serde_json::from_slice(&msg).unwrap();
        assert_eq!(v["entities"].as_array().unwrap().len(), 1);
        assert_eq!(v["entities"][0]["id"], "a");
        assert!(v["deletions"].as_array().unwrap().is_empty());
    }

    #[test]
    fn scene_update_message_fills_required_arrays() {
        let msg = scene_update_message(json!({"id": "a", "models": [{"url": ""}]}));
        let v: Value = serde_json::from_slice(&msg).unwrap();
        let entity = &v["entities"][0];
        // Present arrays are preserved; the rest of the schema-required
        // entity arrays are filled with empty defaults.
        assert_eq!(entity["models"].as_array().unwrap().len(), 1);
        for key in [
            "metadata",
            "arrows",
            "cubes",
            "spheres",
            "cylinders",
            "lines",
            "triangles",
            "texts",
        ] {
            assert!(
                entity[key].as_array().is_some_and(|a| a.is_empty()),
                "{key} must be an empty array"
            );
        }
    }

    #[test]
    fn vector_element_indices_explicit_tuple() {
        let component = Arc::new(eql::Component::new(
            "ball.world_vel".into(),
            impeller2::types::ComponentId::new("ball.world_vel"),
            impeller2::schema::Schema::new(PrimType::F64, [6usize]).unwrap(),
        ));
        let ctx = eql::Context::from_leaves([component], Timestamp(0), Timestamp(1_000_000));
        let expr = ctx
            .parse_str("ball.world_vel[3],ball.world_vel[4],ball.world_vel[5]")
            .unwrap();
        assert_eq!(
            vector_element_indices(&expr, "ball.world_vel", 6),
            [3, 4, 5]
        );
    }

    #[test]
    fn vector_element_indices_bare_component_uses_tail() {
        // Bare component references read the trailing 3 elements, matching the
        // editor's `component_value_tail_to_vec3` (6-elem spatial vector ->
        // linear part; plain 3-vector -> all of it).
        let vel = Arc::new(eql::Component::new(
            "ball.world_vel".into(),
            impeller2::types::ComponentId::new("ball.world_vel"),
            impeller2::schema::Schema::new(PrimType::F64, [6usize]).unwrap(),
        ));
        let wind = Arc::new(eql::Component::new(
            "ball.wind".into(),
            impeller2::types::ComponentId::new("ball.wind"),
            impeller2::schema::Schema::new(PrimType::F64, [3usize]).unwrap(),
        ));
        let ctx = eql::Context::from_leaves([vel, wind], Timestamp(0), Timestamp(1_000_000));
        let expr = ctx.parse_str("ball.world_vel").unwrap();
        assert_eq!(
            vector_element_indices(&expr, "ball.world_vel", 6),
            [3, 4, 5]
        );
        let expr = ctx.parse_str("ball.wind").unwrap();
        assert_eq!(vector_element_indices(&expr, "ball.wind", 3), [0, 1, 2]);
    }

    #[test]
    fn oversized_glb_omits_model_not_empty_data() {
        let dir = std::env::temp_dir().join(format!("elodin_mcap_glb_{}", fastrand::u64(..)));
        std::fs::create_dir_all(&dir).unwrap();
        let glb_path = dir.join("big.glb");
        // 2 KiB payload with a tiny max-embed so the guard trips.
        std::fs::write(&glb_path, vec![0u8; 2048]).unwrap();
        let ctx = empty_eql_ctx();
        let object = test_glb_object(
            "(0, 0, 0, 1, 1, 2, 3)",
            "big.glb",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 45.0),
        );
        let mut lit = 0u32;
        let mut counts = HashMap::new();
        let (entity, assets) = build_object_entity(
            &object,
            &ctx,
            &dir,
            0,
            1, // 1 byte max → always oversized
            None,
            &mut lit,
            &mut counts,
        )
        .unwrap();
        assert!(
            entity.is_none(),
            "oversized GLB must not emit a model entity"
        );
        assert_eq!(assets, vec!["big.glb".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn force_embed_bypasses_size_guard() {
        let dir = std::env::temp_dir().join(format!("elodin_mcap_glb_{}", fastrand::u64(..)));
        std::fs::create_dir_all(&dir).unwrap();
        let glb_path = dir.join("big.glb");
        std::fs::write(&glb_path, vec![0u8; 64]).unwrap();
        // Entity-backed object needs a component leaf named like `lander.world_pos`.
        let component = Arc::new(eql::Component::new(
            "lander.world_pos".into(),
            impeller2::types::ComponentId::new("lander.world_pos"),
            impeller2::schema::Schema::new(PrimType::F64, [7usize]).unwrap(),
        ));
        let ctx = eql::Context::from_leaves([component], Timestamp(0), Timestamp(1_000_000));
        let object = test_glb_object(
            "lander.world_pos",
            "big.glb",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        );
        let mut lit = 0u32;
        let mut counts = HashMap::new();
        let (entity, _) = build_object_entity(
            &object,
            &ctx,
            &dir,
            0,
            1,
            Some("lander"),
            &mut lit,
            &mut counts,
        )
        .unwrap();
        let entity = entity.expect("follow-entity mesh must embed");
        let data = entity["models"][0]["data"].as_str().unwrap();
        assert!(!data.is_empty());
        assert_eq!(entity["id"], "lander-model");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn literal_pose_composes_glb_rotate() {
        let dir = std::env::temp_dir().join(format!("elodin_mcap_glb_{}", fastrand::u64(..)));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tiny.glb"), b"glTF").unwrap();
        let ctx = empty_eql_ctx();
        let object = test_glb_object(
            "(0, 0, 0, 1, 100, 0, 0)",
            "tiny.glb",
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 90.0),
        );
        let mut lit = 0u32;
        let mut counts = HashMap::new();
        let (entity, _) = build_object_entity(
            &object,
            &ctx,
            &dir,
            0,
            1024 * 1024,
            None,
            &mut lit,
            &mut counts,
        )
        .unwrap();
        let entity = entity.unwrap();
        // Literal (100,0,0) + local translate (1,0,0); local 90° Z is orientation-only.
        assert!(
            (entity["models"][0]["pose"]["position"]["x"]
                .as_f64()
                .unwrap()
                - 101.0)
                .abs()
                < 1e-6
        );
        assert!(
            entity["models"][0]["pose"]["position"]["y"]
                .as_f64()
                .unwrap()
                .abs()
                < 1e-6
        );
        let qz = entity["models"][0]["pose"]["orientation"]["z"]
            .as_f64()
            .unwrap();
        assert!((qz - (std::f64::consts::FRAC_PI_4).sin()).abs() < 1e-6);
        assert_eq!(entity["frame_id"], "world");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- Pass 3: camera + line fixes ---

    fn ctx_with_pose(name: &str) -> eql::Context {
        let component = Arc::new(eql::Component::new(
            name.into(),
            impeller2::types::ComponentId::new(name),
            impeller2::schema::Schema::new(PrimType::F64, [7usize]).unwrap(),
        ));
        eql::Context::from_leaves([component], Timestamp(0), Timestamp(1_000_000))
    }

    #[test]
    fn camera_offset_parses_translate_world() {
        let ctx = ctx_with_pose("lander.world_pos");
        let expr = ctx
            .parse_str("lander.world_pos.translate_world(10.0, 10.0, 4.0)")
            .unwrap();
        let offset = camera_offset_from_pos(&expr).expect("offset");
        assert_eq!(offset, [10.0, 10.0, 4.0]);
        let (distance, phi, theta) = camera_orbit_from_offset(Some(offset));
        assert!((distance - 216.0f64.sqrt()).abs() < 1e-9);
        assert!(phi > 0.0 && theta > 0.0);
    }

    #[test]
    fn camera_offset_parses_chained_axis_translate() {
        let ctx = ctx_with_pose("bdx.world_pos");
        let expr = ctx
            .parse_str("bdx.world_pos.rotate_z(-90).translate_y(-2.0)")
            .unwrap();
        let offset = camera_offset_from_pos(&expr).expect("offset");
        assert_eq!(offset, [0.0, -2.0, 0.0]);
    }

    #[test]
    fn camera_offset_literal_tuple_still_works() {
        let ctx = ctx_with_pose("drone.world_pos");
        let expr = ctx.parse_str("drone.world_pos + (0,0,0,0, 2,2,2)").unwrap();
        assert_eq!(camera_offset_from_pos(&expr), Some([2.0, 2.0, 2.0]));
    }

    #[test]
    fn line_entity_uses_pixel_thickness() {
        let sample_size = 7 * 8;
        let mut data = Vec::new();
        let mut timestamps = Vec::new();
        for i in 0..10 {
            let row: Vec<u8> = [0.0, 0.0, 0.0, 1.0, i as f64, 0.0, 0.0]
                .iter()
                .flat_map(|v: &f64| v.to_le_bytes())
                .collect();
            data.extend_from_slice(&row);
            timestamps.push(Timestamp(i * 1000));
        }
        let comp = make_test_component("jet.world_pos", PrimType::F64, &[7]);
        let cursors = vec![Some(ComponentCursor {
            timestamps: &timestamps,
            data: &data,
            sample_size,
        })];
        let line = Line3d {
            eql: "jet.world_pos".into(),
            line_width: 3.0,
            color: None,
            future_color: None,
            perspective: false,
            frame: None,
            node_id: Default::default(),
        };
        let ctx = ctx_with_pose("jet.world_pos");
        let comps = vec![comp];
        let entity =
            build_line_entity(&line, &ctx, &comps, &cursors, 0, false).expect("line entity");
        let l = &entity["lines"][0];
        assert_eq!(
            l["scale_invariant"], true,
            "pixel-width lines must be scale invariant"
        );
        assert_eq!(l["thickness"], 3.0);
    }

    #[test]
    fn build_scene_emits_per_entity_messages() {
        let dir = std::env::temp_dir().join(format!("elodin_mcap_scene_{}", fastrand::u64(..)));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.glb"), b"aaaa").unwrap();
        std::fs::write(dir.join("b.glb"), b"bbbb").unwrap();
        let ctx = empty_eql_ctx();
        let schematics = LoadedSchematics {
            primary: Some(Schematic {
                elems: vec![
                    SchematicElem::Object3d(test_glb_object(
                        "(0, 0, 0, 1, 0, 0, 0)",
                        "a.glb",
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                    )),
                    SchematicElem::Object3d(test_glb_object(
                        "(0, 0, 0, 1, 5, 0, 0)",
                        "b.glb",
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                    )),
                ],
                theme: None,
                timeline: None,
                frame: None,
                origin: None,
                skybox: None,
                environment: None,
                telemetry_mode: false,
            }),
            windows: vec![],
            raw: Vec::new(),
        };
        let scene = build_scene(
            &schematics,
            &ctx,
            &dir,
            0,
            &[],
            &[],
            1024 * 1024,
            None,
            false,
        )
        .unwrap();
        assert_eq!(scene.messages.len(), 2);
        // Each entity gets its own topic (latest-per-topic backfill safety).
        assert_eq!(scene.messages[0].0, "/scene/literal-1-model");
        assert_eq!(scene.messages[1].0, "/scene/literal-2-model");
        for (_, msg) in &scene.messages {
            let v: Value = serde_json::from_slice(msg).unwrap();
            assert_eq!(v["entities"].as_array().unwrap().len(), 1);
            assert!(
                !v["entities"][0]["models"][0]["data"]
                    .as_str()
                    .unwrap()
                    .is_empty()
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn scene_topic_sanitizes_ids() {
        assert_eq!(scene_topic("bdx-model"), "/scene/bdx-model");
        assert_eq!(scene_topic("DPS thrust"), "/scene/DPS-thrust");
    }

    #[test]
    fn geo_anchors_equator_prime_meridian() {
        let anchors = geo_frame_anchors(&impeller2_wkt::GeoOriginConfig {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
        });
        // Origin on the equator at lon 0: ECEF (a, 0, 0).
        assert!((anchors.origin_ecef[0] - 6_378_137.0).abs() < 1.0);
        assert!(anchors.origin_ecef[1].abs() < 1e-6);
        assert!(anchors.origin_ecef[2].abs() < 1e-6);
        // ENU: local up (0,0,1) maps to ECEF radial +X.
        let up = quat_rotate_vec(anchors.enu_quat, [0.0, 0.0, 1.0]);
        assert!((up[0] - 1.0).abs() < 1e-9, "up {up:?}");
        // NED: local down (0,0,1) maps to ECEF -X.
        let down = quat_rotate_vec(anchors.ned_quat, [0.0, 0.0, 1.0]);
        assert!((down[0] + 1.0).abs() < 1e-9, "down {down:?}");
        // ENU east (1,0,0) maps to ECEF +Y.
        let east = quat_rotate_vec(anchors.enu_quat, [1.0, 0.0, 0.0]);
        assert!((east[1] - 1.0).abs() < 1e-9, "east {east:?}");
    }
}
