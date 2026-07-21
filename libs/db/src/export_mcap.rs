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
//! - schematic `object_3d` / `vector_arrow` -> `/scene` (`foxglove.SceneUpdate`,
//!   GLBs embedded as base64 glTF binary)
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
    Color, LogEntry, Object3D, Object3DMesh, Panel, Schematic, SchematicElem, SensorCameraConfig,
    VectorArrow3d, log_entry_msg_schema,
};
use mcap::records::MessageHeader;
use serde_json::{Map, Value, json};

use crate::msg_log::MsgLog;
use crate::{Component, DB, Error};

/// 1 MiB output buffer, matching the other exporters.
const FILE_BUF_CAP: usize = 1 << 20;

/// Options for the MCAP export.
#[derive(Clone, Debug, Default)]
pub struct McapExportOptions {
    /// Glob pattern over component names; non-matching components are skipped.
    pub pattern: Option<String>,
    /// Include components whose metadata contains `"private": "true"`.
    pub include_private: bool,
    /// Attach every file under `{db}/assets/` instead of only schematic-referenced ones.
    pub all_assets: bool,
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

fn us_to_ns(ts: Timestamp) -> u64 {
    ts.0.max(0) as u64 * 1000
}

/// FrameTransforms message for one pose sample (`[qx,qy,qz,qw, x,y,z]`).
fn tf_message(entity: &str, prim: PrimType, buf: &[u8], ts_ns: u64) -> Vec<u8> {
    let q = |i| read_f64(prim, buf, i);
    let msg = json!({
        "transforms": [{
            "timestamp": timestamp_json(ts_ns),
            "parent_frame_id": "world",
            "child_frame_id": entity,
            "translation": {"x": q(4), "y": q(5), "z": q(6)},
            "rotation": {"x": q(0), "y": q(1), "z": q(2), "w": q(3)},
        }]
    });
    serde_json::to_vec(&msg).expect("json serialize")
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

fn is_annex_b(payload: &[u8]) -> bool {
    payload.starts_with(&[0, 0, 0, 1]) || payload.starts_with(&[0, 0, 1])
}

fn classify_msg_log(
    log: &MsgLog,
    name: &str,
    sensor_by_msg_id: &HashMap<PacketId, SensorCameraConfig>,
    video_names: &HashSet<String>,
) -> MsgLogKind {
    if let Some(cfg) = sensor_by_msg_id.get(&msg_id(name)) {
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
    let vals = find_literal_tuple(expr)?;
    if vals.len() < 3 {
        return None;
    }
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

struct SceneBuild {
    /// SceneUpdate JSON message body.
    message: Vec<u8>,
    /// Asset keys referenced by embedded models (for attachments).
    referenced_assets: Vec<String>,
}

/// Build the one-shot `/scene` SceneUpdate from schematic 3D elements.
fn build_scene(
    schematics: &LoadedSchematics,
    ctx: &eql::Context,
    assets_dir: &Path,
    ts_ns: u64,
) -> Option<SceneBuild> {
    let mut entities: Vec<Value> = Vec::new();
    let mut referenced_assets = Vec::new();
    let mut arrow_groups: HashMap<String, Vec<Value>> = HashMap::new();

    let all_elems = schematics
        .primary
        .iter()
        .chain(schematics.windows.iter().map(|(_, s)| s))
        .flat_map(|s| s.elems.iter());

    for elem in all_elems {
        match elem {
            SchematicElem::Object3d(object) => {
                match build_object_entity(object, ctx, assets_dir, ts_ns) {
                    Ok(Some((entity, assets))) => {
                        entities.push(entity);
                        referenced_assets.extend(assets);
                    }
                    Ok(None) => {}
                    Err(err) => eprintln!("Warning: skipping object_3d ({}): {err}", object.eql),
                }
            }
            SchematicElem::VectorArrow(arrow) => match build_arrow(arrow, ctx) {
                Some((frame, primitive)) => arrow_groups.entry(frame).or_default().push(primitive),
                None => eprintln!(
                    "Warning: skipping vector_arrow '{}' (only constant body-frame arrows are exported)",
                    arrow.vector
                ),
            },
            _ => {}
        }
    }

    for (frame, arrows) in arrow_groups {
        entities.push(json!({
            "timestamp": timestamp_json(ts_ns),
            "frame_id": frame,
            "id": format!("{frame}-arrows"),
            "lifetime": {"sec": 0, "nsec": 0},
            "frame_locked": true,
            "arrows": arrows,
        }));
    }

    if entities.is_empty() {
        return None;
    }
    let msg = json!({"deletions": [], "entities": entities});
    Some(SceneBuild {
        message: serde_json::to_vec(&msg).expect("json serialize"),
        referenced_assets,
    })
}

fn build_object_entity(
    object: &Object3D,
    ctx: &eql::Context,
    assets_dir: &Path,
    ts_ns: u64,
) -> Result<Option<(Value, Vec<String>)>, Error> {
    let Some(frame) = entity_for_eql(&object.eql, ctx) else {
        return Ok(None);
    };
    let mut entity = Map::new();
    entity.insert("timestamp".into(), timestamp_json(ts_ns));
    entity.insert("frame_id".into(), Value::String(frame.clone()));
    entity.insert("id".into(), Value::String(format!("{frame}-model")));
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
            let quat = quat_from_euler_deg(*rotate);
            entity.insert(
                "models".into(),
                json!([{
                    "pose": {
                        "position": {"x": translate.0, "y": translate.1, "z": translate.2},
                        "orientation": {"x": quat[0], "y": quat[1], "z": quat[2], "w": quat[3]},
                    },
                    "scale": {"x": scale, "y": scale, "z": scale},
                    "color": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0},
                    "override_color": false,
                    "url": "",
                    "media_type": "model/gltf-binary",
                    "data": BASE64.encode(&bytes),
                }]),
            );
            referenced.push(key);
        }
        Object3DMesh::Mesh { mesh, material } => {
            let color = color_json(&material.base_color);
            match mesh {
                impeller2_wkt::Mesh::Sphere { radius } => {
                    let d = (radius * 2.0) as f64;
                    entity.insert(
                        "spheres".into(),
                        json!([{"pose": identity_pose(), "size": {"x": d, "y": d, "z": d}, "color": color}]),
                    );
                }
                impeller2_wkt::Mesh::Box { x, y, z } => {
                    entity.insert(
                        "cubes".into(),
                        json!([{"pose": identity_pose(), "size": {"x": x, "y": y, "z": z}, "color": color}]),
                    );
                }
                impeller2_wkt::Mesh::Cylinder { radius, height } => {
                    let d = (radius * 2.0) as f64;
                    entity.insert(
                        "cylinders".into(),
                        json!([{
                            "pose": identity_pose(),
                            "size": {"x": d, "y": d, "z": height},
                            "bottom_scale": 1.0, "top_scale": 1.0, "color": color,
                        }]),
                    );
                }
                impeller2_wkt::Mesh::Plane { width, depth } => {
                    entity.insert(
                        "cubes".into(),
                        json!([{
                            "pose": identity_pose(),
                            "size": {"x": width, "y": depth, "z": 0.01},
                            "color": color,
                        }]),
                    );
                }
            }
        }
        Object3DMesh::Ellipsoid { .. } => {
            // Data-driven scale; no static scene equivalent.
            return Ok(None);
        }
    }

    Ok(Some((Value::Object(entity), referenced)))
}

/// Body-frame constant-vector arrows attach to the origin entity's TF frame.
fn build_arrow(arrow: &VectorArrow3d, ctx: &eql::Context) -> Option<(String, Value)> {
    if !arrow.body_frame {
        return None;
    }
    let origin = arrow.origin.as_deref()?;
    let frame = entity_for_eql(origin, ctx)?;
    let vec = parse_literal_tuple(&arrow.vector)?;
    if vec.len() != 3 {
        return None;
    }
    let len = (vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]).sqrt();
    if len == 0.0 {
        return None;
    }
    let dir = [vec[0] / len, vec[1] / len, vec[2] / len];
    let quat = quat_from_x_axis(dir);
    let total = len * arrow.scale;
    let primitive = json!({
        "pose": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"x": quat[0], "y": quat[1], "z": quat[2], "w": quat[3]},
        },
        "shaft_length": total * 0.8,
        "shaft_diameter": total * 0.02f64.max(0.01),
        "head_length": total * 0.2,
        "head_diameter": total * 0.06f64.max(0.02),
        "color": color_json(&arrow.color),
    });
    Some((frame, primitive))
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
}

impl<'a> LayoutBuilder<'a> {
    fn new(
        ctx: &'a eql::Context,
        components: &'a [ExportComponent],
        follow_entity: Option<String>,
    ) -> Self {
        Self {
            config_by_id: Map::new(),
            counter: 0,
            ctx,
            components,
            follow_entity,
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
                let offset = viewport
                    .pos
                    .as_deref()
                    .and_then(|pos| self.ctx.parse_str(pos).ok())
                    .and_then(|expr| camera_offset_from_pos(&expr));
                let (distance, phi, theta) = camera_orbit_from_offset(offset);
                let mut config = Map::new();
                // cameraState angles (phi/thetaOffset/fovy) are in degrees.
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
                        "near": 0.01,
                        "far": 5000,
                    }),
                );
                if let Some(entity) = self.follow_entity.clone().or_else(|| {
                    viewport
                        .pos
                        .as_deref()
                        .and_then(|p| entity_for_eql(p, self.ctx))
                }) {
                    config.insert("followTf".into(), Value::String(entity));
                    config.insert("followMode".into(), Value::String("follow-position".into()));
                }
                config.insert("layers".into(), Value::Object(layers));
                config.insert("topics".into(), json!({"/scene": {"visible": true}}));
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
                let config = json!({
                    "imageMode": {"imageTopic": format!("/camera/{}", view.msg_name)},
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
            | Panel::DataOverview(_) => None,
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
) -> Option<Value> {
    let primary = schematics.primary.as_ref()?;

    // Follow entity: first object_3d in any schematic.
    let follow_entity = schematics
        .primary
        .iter()
        .chain(schematics.windows.iter().map(|(_, s)| s))
        .flat_map(|s| s.elems.iter())
        .find_map(|elem| match elem {
            SchematicElem::Object3d(object) => entity_for_eql(&object.eql, ctx),
            _ => None,
        });

    let mut builder = LayoutBuilder::new(ctx, components, follow_entity);

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

    let glob_pattern = options
        .pattern
        .as_ref()
        .map(|p| Pattern::new(p))
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
            let kind = classify_msg_log(&log, &name, &sensor_by_msg_id, &video_names);
            let topic = match kind {
                MsgLogKind::H264Video => format!("/video/{name}"),
                MsgLogKind::SensorCamera(_) => format!("/camera/{name}"),
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
            MsgLogKind::SensorCamera(_) => ("foxglove.RawImage", SCHEMA_RAW_IMAGE),
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
    struct ComponentCursor<'a> {
        timestamps: &'a [Timestamp],
        data: &'a [u8],
        sample_size: usize,
    }
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
    let start_ns = us_to_ns(start_ts);

    // ---- /scene one-shot -----------------------------------------------------
    let scene = build_scene(&schematics, &ctx, &assets_dir, start_ns);
    let mut referenced_assets: Vec<String> = Vec::new();
    if let Some(scene) = &scene {
        let schema_id = writer.add_schema(
            "foxglove.SceneUpdate",
            "jsonschema",
            SCHEMA_SCENE_UPDATE.as_bytes(),
        )?;
        let channel_id = writer.add_channel(schema_id, "/scene", "json", &empty_metadata)?;
        writer.write_to_known_channel(
            &MessageHeader {
                channel_id,
                sequence: 0,
                log_time: start_ns,
                publish_time: start_ns,
            },
            &scene.message,
        )?;
        referenced_assets.extend(scene.referenced_assets.iter().cloned());
    }

    // ---- k-way merge over all cursors ----------------------------------------
    // Cursor id space: [0, n) component channels, [n, 2n) TF from pose
    // components, [2n, 2n + m) msg logs.
    let n = snapshot.components.len();
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let mut heap: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();
    let mut positions = vec![0usize; n * 2 + export_msg_logs.len()];

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

    let mut sequences: HashMap<u16, u32> = HashMap::new();
    let mut message_count = 0u64;
    while let Some(Reverse((ts_us, cursor_id))) = heap.pop() {
        crate::cancellation::check_cancelled()?;
        let ts_ns = us_to_ns(Timestamp(ts_us));
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
                tf_message(entity, comp.component.schema.prim_type, buf, ts_ns),
                cursor.timestamps.get(pos + 1).map(|t| t.0),
            )
        } else {
            let idx = cursor_id - 2 * n;
            let log = &export_msg_logs[idx];
            let (_, payload) = msg_entries[idx][pos];
            (
                msg_channels[idx],
                msg_log_json(&log.kind, &log.name, payload, ts_ns),
                msg_entries[idx].get(pos + 1).map(|(t, _)| t.0),
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
    writer.write_metadata(&mcap::records::Metadata {
        name: "elodin.db_state".to_string(),
        metadata: snapshot.db_metadata.clone(),
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
    match build_layout(&schematics, &ctx, &snapshot.components) {
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
        let msg = tf_message("drone", PrimType::F64, &buf, 1_000);
        let value: Value = serde_json::from_slice(&msg).unwrap();
        let tf = &value["transforms"][0];
        assert_eq!(tf["child_frame_id"], "drone");
        assert_eq!(tf["rotation"]["x"], 0.1);
        assert_eq!(tf["rotation"]["w"], 0.9);
        assert_eq!(tf["translation"]["x"], 1.0);
        assert_eq!(tf["translation"]["z"], 3.0);
    }
}
