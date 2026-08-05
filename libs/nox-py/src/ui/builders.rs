//! Panel / object builders for `elodin.ui` (Phase 1: EQL as strings).

use std::collections::HashMap;
use std::ops::Range;
use std::str::FromStr;
use std::time::Duration;

use bevy_geo_frames::{GeoFrame, RotationKind};
use impeller2_kdl::color_from_name;
use impeller2_wkt::*;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

#[pyclass(name = "Panel", module = "elodin.ui")]
#[derive(Clone, Debug)]
pub struct PyPanel {
    pub(crate) inner: Panel,
    pub(crate) share: Option<f32>,
}

#[pyclass(name = "Object3D", module = "elodin.ui")]
#[derive(Clone, Debug)]
pub struct PyObject3D {
    pub(crate) inner: Object3D,
}

#[pyclass(name = "Line3d", module = "elodin.ui")]
#[derive(Clone, Debug)]
pub struct PyLine3d {
    pub(crate) inner: Line3d,
}

#[pyclass(name = "VectorArrow", module = "elodin.ui")]
#[derive(Clone, Debug)]
pub struct PyVectorArrow {
    pub(crate) inner: VectorArrow3d,
}

#[pyclass(name = "WorldMesh", module = "elodin.ui")]
#[derive(Clone, Debug)]
pub struct PyWorldMesh {
    pub(crate) inner: WorldMesh,
}

#[pyclass(name = "Window", module = "elodin.ui")]
#[derive(Clone, Debug)]
pub struct PyWindow {
    pub(crate) inner: WindowSchematic,
}

#[pyclass(name = "Mesh", module = "elodin.ui")]
#[derive(Clone, Debug)]
pub struct PyMesh {
    pub(crate) inner: Object3DMesh,
}

#[pyclass(name = "Joint", module = "elodin.ui")]
#[derive(Clone, Debug)]
pub struct PyJoint {
    pub(crate) inner: JointAnimation,
}

#[pyclass(name = "Coordinate", module = "elodin.ui")]
#[derive(Clone, Debug)]
pub struct PyCoordinate {
    pub(crate) inner: CoordinateConfig,
}

#[pyclass(name = "Theme", module = "elodin.ui")]
#[derive(Clone, Debug)]
pub struct PyTheme {
    pub(crate) inner: ThemeConfig,
}

#[pyclass(name = "Timeline", module = "elodin.ui")]
#[derive(Clone, Debug)]
pub struct PyTimeline {
    pub(crate) inner: TimelineConfig,
}

pub(crate) fn extract_coordinate(obj: &Bound<'_, PyAny>) -> PyResult<CoordinateConfig> {
    Ok(obj.extract::<PyRef<'_, PyCoordinate>>()?.inner)
}

pub(crate) fn extract_theme(obj: &Bound<'_, PyAny>) -> PyResult<ThemeConfig> {
    Ok(obj.extract::<PyRef<'_, PyTheme>>()?.inner.clone())
}

pub(crate) fn extract_timeline(obj: &Bound<'_, PyAny>) -> PyResult<TimelineConfig> {
    Ok(obj.extract::<PyRef<'_, PyTimeline>>()?.inner.clone())
}

fn extract_panel(obj: &Bound<'_, PyAny>) -> PyResult<(Panel, Option<f32>)> {
    let panel = obj.extract::<PyRef<'_, PyPanel>>()?;
    Ok((panel.inner.clone(), panel.share))
}

fn collect_split_children(
    children: Vec<Bound<'_, PyAny>>,
) -> PyResult<(Vec<Panel>, HashMap<usize, f32>)> {
    let mut panels = Vec::with_capacity(children.len());
    let mut shares = HashMap::new();
    for (i, child) in children.into_iter().enumerate() {
        let (panel, share) = extract_panel(&child)?;
        if let Some(share) = share {
            shares.insert(i, share);
        }
        panels.push(panel);
    }
    Ok((panels, shares))
}

fn parse_color(name: &str) -> PyResult<Color> {
    color_from_name(name)
        .ok_or_else(|| PyValueError::new_err(format!("unknown color name {name:?}")))
}

fn parse_frame(frame: &str) -> PyResult<GeoFrame> {
    GeoFrame::from_str(frame)
        .map_err(|_| PyValueError::new_err(format!("unknown coordinate frame {frame:?}")))
}

#[pyfunction]
#[pyo3(signature = (frame, lat=None, lon=None, alt=None, body=None))]
fn coordinate(
    frame: &str,
    lat: Option<f64>,
    lon: Option<f64>,
    alt: Option<f64>,
    body: Option<&str>,
) -> PyResult<PyCoordinate> {
    let origin = match (lat, lon) {
        (Some(latitude), Some(longitude)) => Some(GeoOriginConfig {
            latitude,
            longitude,
            altitude: alt.unwrap_or(0.0),
        }),
        (None, None) if alt.is_none() => None,
        _ => {
            return Err(PyValueError::new_err(
                "coordinate origin requires both lat and lon",
            ));
        }
    };
    let body = match body {
        None => None,
        Some(s) => Some(
            CelestialBody::from_str_ci(s)
                .ok_or_else(|| PyValueError::new_err(format!("unknown body {s:?}")))?,
        ),
    };
    Ok(PyCoordinate {
        inner: CoordinateConfig {
            frame: parse_frame(frame)?,
            origin,
            body,
        },
    })
}

#[pyfunction]
#[pyo3(signature = (mode=None, scheme=None))]
fn theme(mode: Option<String>, scheme: Option<String>) -> PyTheme {
    PyTheme {
        inner: ThemeConfig { mode, scheme },
    }
}

#[pyfunction]
#[pyo3(signature = (played_color=None, future_color=None, follow_latest=false, range=None))]
fn timeline(
    played_color: Option<&str>,
    future_color: Option<&str>,
    follow_latest: bool,
    range: Option<String>,
) -> PyResult<PyTimeline> {
    let mut cfg = TimelineConfig {
        follow_latest,
        range,
        ..TimelineConfig::default()
    };
    if let Some(name) = played_color {
        cfg.played_color = parse_color(name)?;
    }
    if let Some(name) = future_color {
        cfg.future_color = parse_color(name)?;
    }
    Ok(PyTimeline { inner: cfg })
}

#[pyfunction]
#[pyo3(signature = (*children))]
fn tabs(children: Vec<Bound<'_, PyAny>>) -> PyResult<PyPanel> {
    let panels = children
        .into_iter()
        .map(|c| extract_panel(&c).map(|(p, _)| p))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyPanel {
        inner: Panel::Tabs(panels),
        share: None,
    })
}

#[pyfunction]
#[pyo3(signature = (*children, name=None, active=false, share=None))]
fn hsplit(
    children: Vec<Bound<'_, PyAny>>,
    name: Option<String>,
    active: bool,
    share: Option<f32>,
) -> PyResult<PyPanel> {
    let (panels, shares) = collect_split_children(children)?;
    Ok(PyPanel {
        inner: Panel::HSplit(Split {
            panels,
            shares,
            active,
            name,
        }),
        share,
    })
}

#[pyfunction]
#[pyo3(signature = (*children, name=None, active=false, share=None))]
fn vsplit(
    children: Vec<Bound<'_, PyAny>>,
    name: Option<String>,
    active: bool,
    share: Option<f32>,
) -> PyResult<PyPanel> {
    let (panels, shares) = collect_split_children(children)?;
    Ok(PyPanel {
        inner: Panel::VSplit(Split {
            panels,
            shares,
            active,
            name,
        }),
        share,
    })
}

#[pyfunction]
#[pyo3(signature = (
    eql,
    name=None,
    graph_type=None,
    locked=false,
    auto_y_range=true,
    y_min=None,
    y_max=None,
    colors=None,
    share=None,
))]
#[allow(clippy::too_many_arguments)]
fn graph(
    eql: &str,
    name: Option<String>,
    graph_type: Option<&str>,
    locked: bool,
    auto_y_range: bool,
    y_min: Option<f64>,
    y_max: Option<f64>,
    colors: Option<Vec<String>>,
    share: Option<f32>,
) -> PyResult<PyPanel> {
    let graph_type = match graph_type {
        None | Some("line") => GraphType::Line,
        Some("point") => GraphType::Point,
        Some("bar") => GraphType::Bar,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "unknown graph type {other:?}"
            )));
        }
    };
    let y_range: Range<f64> = match (y_min, y_max) {
        (Some(min), Some(max)) => min..max,
        (None, None) => 0.0..1.0,
        _ => {
            return Err(PyValueError::new_err(
                "graph y_min and y_max must both be set or both omitted",
            ));
        }
    };
    let colors = colors
        .unwrap_or_default()
        .into_iter()
        .map(|c| parse_color(&c))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyPanel {
        inner: Panel::Graph(Graph {
            eql: eql.to_string(),
            name,
            graph_type,
            locked,
            auto_y_range,
            y_range,
            node_id: NodeId::default(),
            colors,
        }),
        share,
    })
}

#[pyfunction]
#[pyo3(signature = (
    name=None,
    fov=45.0,
    near=None,
    far=None,
    aspect=None,
    active=false,
    show_grid=false,
    show_arrows=true,
    create_frustum=false,
    show_frustums=false,
    show_view_cube=true,
    effects=true,
    hdr=false,
    pos=None,
    look_at=None,
    up=None,
    frame=None,
    share=None,
))]
#[allow(clippy::too_many_arguments)]
fn viewport(
    name: Option<String>,
    fov: f32,
    near: Option<f32>,
    far: Option<f32>,
    aspect: Option<f32>,
    active: bool,
    show_grid: bool,
    show_arrows: bool,
    create_frustum: bool,
    show_frustums: bool,
    show_view_cube: bool,
    effects: bool,
    hdr: bool,
    pos: Option<String>,
    look_at: Option<String>,
    up: Option<String>,
    frame: Option<&str>,
    share: Option<f32>,
) -> PyResult<PyPanel> {
    Ok(PyPanel {
        inner: Panel::Viewport(Viewport {
            fov,
            near,
            far,
            aspect,
            active,
            show_grid,
            show_arrows,
            create_frustum,
            show_frustums,
            show_view_cube,
            effects,
            hdr,
            name,
            pos,
            look_at,
            up,
            frame: frame.map(parse_frame).transpose()?,
            ..Viewport::default()
        }),
        share,
    })
}

#[pyfunction]
#[pyo3(signature = (component_name, name=None, share=None))]
fn component_monitor(component_name: &str, name: Option<String>, share: Option<f32>) -> PyPanel {
    PyPanel {
        inner: Panel::ComponentMonitor(ComponentMonitor {
            component_name: component_name.to_string(),
            name,
        }),
        share,
    }
}

#[pyfunction]
fn inspector() -> PyPanel {
    PyPanel {
        inner: Panel::Inspector,
        share: None,
    }
}

#[pyfunction]
fn hierarchy() -> PyPanel {
    PyPanel {
        inner: Panel::Hierarchy,
        share: None,
    }
}

#[pyfunction]
#[pyo3(signature = (name=None))]
fn schematic_tree(name: Option<String>) -> PyPanel {
    PyPanel {
        inner: Panel::SchematicTree(name),
        share: None,
    }
}

#[pyfunction]
#[pyo3(signature = (name=None))]
fn data_overview(name: Option<String>) -> PyPanel {
    PyPanel {
        inner: Panel::DataOverview(name),
        share: None,
    }
}

#[pyfunction]
#[pyo3(signature = (msg_name, name=None, share=None))]
fn video_stream(msg_name: &str, name: Option<String>, share: Option<f32>) -> PyPanel {
    PyPanel {
        inner: Panel::VideoStream(VideoStream {
            msg_name: msg_name.to_string(),
            name,
        }),
        share,
    }
}

#[pyfunction]
#[pyo3(signature = (msg_name, name=None, share=None))]
fn sensor_view(msg_name: &str, name: Option<String>, share: Option<f32>) -> PyPanel {
    PyPanel {
        inner: Panel::SensorView(SensorView {
            msg_name: msg_name.to_string(),
            name,
        }),
        share,
    }
}

#[pyfunction]
#[pyo3(signature = (msg_name, name=None, share=None))]
fn log_stream(msg_name: &str, name: Option<String>, share: Option<f32>) -> PyPanel {
    PyPanel {
        inner: Panel::LogStream(LogStream {
            msg_name: msg_name.to_string(),
            name,
        }),
        share,
    }
}

#[pyfunction]
#[pyo3(signature = (name, lua, share=None))]
fn action_pane(name: &str, lua: &str, share: Option<f32>) -> PyPanel {
    PyPanel {
        inner: Panel::ActionPane(ActionPane {
            name: name.to_string(),
            lua: lua.to_string(),
        }),
        share,
    }
}

#[pyfunction]
#[pyo3(signature = (query, name=None, query_type=None, share=None))]
fn query_table(
    query: &str,
    name: Option<String>,
    query_type: Option<&str>,
    share: Option<f32>,
) -> PyResult<PyPanel> {
    let query_type = match query_type {
        None | Some("eql") | Some("EQL") => QueryType::EQL,
        Some("sql") | Some("SQL") => QueryType::SQL,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "unknown query_type {other:?}"
            )));
        }
    };
    Ok(PyPanel {
        inner: Panel::QueryTable(QueryTable {
            name,
            query: query.to_string(),
            query_type,
        }),
        share,
    })
}

#[pyfunction]
#[pyo3(signature = (
    name,
    query,
    refresh_interval_ms=1000.0,
    auto_refresh=true,
    color=None,
    query_type=None,
    plot_mode=None,
    x_label=None,
    y_label=None,
    share=None,
))]
#[allow(clippy::too_many_arguments)]
fn query_plot(
    name: &str,
    query: &str,
    refresh_interval_ms: f64,
    auto_refresh: bool,
    color: Option<&str>,
    query_type: Option<&str>,
    plot_mode: Option<&str>,
    x_label: Option<String>,
    y_label: Option<String>,
    share: Option<f32>,
) -> PyResult<PyPanel> {
    let query_type = match query_type {
        None | Some("eql") | Some("EQL") => QueryType::EQL,
        Some("sql") | Some("SQL") => QueryType::SQL,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "unknown query_type {other:?}"
            )));
        }
    };
    let plot_mode = match plot_mode {
        None | Some("time_series") | Some("TimeSeries") => PlotMode::TimeSeries,
        Some("xy") | Some("XY") => PlotMode::XY,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "unknown plot_mode {other:?}"
            )));
        }
    };
    Ok(PyPanel {
        inner: Panel::QueryPlot(QueryPlot {
            name: name.to_string(),
            query: query.to_string(),
            refresh_interval: Duration::from_secs_f64(refresh_interval_ms / 1000.0),
            auto_refresh,
            color: match color {
                Some(c) => parse_color(c)?,
                None => Color::YALK,
            },
            query_type,
            plot_mode,
            x_label,
            y_label,
            node_id: NodeId::default(),
        }),
        share,
    })
}

#[pyfunction]
#[pyo3(signature = (path, scale=None, translate=None, rotate=None, emissivity=0.0, glow=0.0))]
fn glb(
    path: &str,
    scale: Option<f32>,
    translate: Option<(f32, f32, f32)>,
    rotate: Option<(f32, f32, f32)>,
    emissivity: f32,
    glow: f32,
) -> PyMesh {
    PyMesh {
        inner: Object3DMesh::Glb {
            path: path.to_string(),
            scale: scale.unwrap_or(default_glb_scale()),
            translate: translate.unwrap_or_else(default_glb_translate),
            rotate: rotate.unwrap_or_else(default_glb_rotate),
            animations: Vec::new(),
            emissivity,
            glow,
            glow_color: None,
        },
    }
}

#[pyfunction]
#[pyo3(signature = (joint, rotation_vector))]
fn joint(joint: &str, rotation_vector: &str) -> PyJoint {
    PyJoint {
        inner: JointAnimation {
            joint_name: joint.to_string(),
            eql_expr: rotation_vector.to_string(),
        },
    }
}

#[pyfunction]
#[pyo3(signature = (eql, mesh, frame=None, orientation=None, animate=None))]
fn object_3d(
    eql: &str,
    mesh: Bound<'_, PyAny>,
    frame: Option<&str>,
    orientation: Option<&str>,
    animate: Option<Vec<Bound<'_, PyAny>>>,
) -> PyResult<PyObject3D> {
    let mut mesh = mesh.extract::<PyRef<'_, PyMesh>>()?.inner.clone();
    if let Some(anims) = animate {
        let animations = anims
            .into_iter()
            .map(|a| Ok(a.extract::<PyRef<'_, PyJoint>>()?.inner.clone()))
            .collect::<PyResult<Vec<_>>>()?;
        if let Object3DMesh::Glb {
            path,
            scale,
            translate,
            rotate,
            emissivity,
            glow,
            glow_color,
            ..
        } = mesh
        {
            mesh = Object3DMesh::Glb {
                path,
                scale,
                translate,
                rotate,
                animations,
                emissivity,
                glow,
                glow_color,
            };
        } else if !animations.is_empty() {
            return Err(PyTypeError::new_err(
                "animate= is only supported on glb meshes",
            ));
        }
    }
    let orientation = match orientation {
        None => RotationKind::default(),
        Some("relative") => RotationKind::Relative,
        Some("absolute") => RotationKind::Absolute,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "unknown orientation {other:?}"
            )));
        }
    };
    Ok(PyObject3D {
        inner: Object3D {
            eql: eql.to_string(),
            mesh,
            frame: frame.map(parse_frame).transpose()?,
            frame_orientation: None,
            orientation,
            icon: None,
            thrusters: Vec::new(),
            mesh_visibility_range: None,
            node_id: NodeId::default(),
        },
    })
}

#[pyfunction]
#[pyo3(signature = (eql, line_width=1.0, color=None, future_color=None, perspective=true, frame=None))]
fn line_3d(
    eql: &str,
    line_width: f32,
    color: Option<&str>,
    future_color: Option<&str>,
    perspective: bool,
    frame: Option<&str>,
) -> PyResult<PyLine3d> {
    Ok(PyLine3d {
        inner: Line3d {
            eql: eql.to_string(),
            line_width,
            color: color.map(parse_color).transpose()?,
            future_color: future_color.map(parse_color).transpose()?,
            perspective,
            frame: frame.map(parse_frame).transpose()?,
            node_id: NodeId::default(),
        },
    })
}

#[pyfunction]
#[pyo3(signature = (vector, origin=None, name=None, color=None, scale=None, body_frame=false, normalize=false, frame=None))]
#[allow(clippy::too_many_arguments)]
fn vector_arrow(
    vector: &str,
    origin: Option<String>,
    name: Option<String>,
    color: Option<&str>,
    scale: Option<f64>,
    body_frame: bool,
    normalize: bool,
    frame: Option<&str>,
) -> PyResult<PyVectorArrow> {
    Ok(PyVectorArrow {
        inner: VectorArrow3d {
            vector: vector.to_string(),
            origin,
            scale: scale.unwrap_or(1.0),
            name,
            color: match color {
                Some(c) => parse_color(c)?,
                None => Color::WHITE,
            },
            body_frame,
            normalize,
            show_name: true,
            thickness: ArrowThickness::default(),
            label_position: LabelPosition::default(),
            frame: frame.map(parse_frame).transpose()?,
            node_id: NodeId::default(),
        },
    })
}

#[pyfunction]
#[pyo3(signature = (region, lod_count=None, translate=None, frame=None, visible=true))]
fn world_mesh(
    region: &str,
    lod_count: Option<u32>,
    translate: Option<(f64, f64, f64)>,
    frame: Option<&str>,
    visible: bool,
) -> PyResult<PyWorldMesh> {
    Ok(PyWorldMesh {
        inner: WorldMesh {
            region: region.to_string(),
            lod_count,
            translate,
            frame: frame.map(parse_frame).transpose()?,
            visible,
            node_id: NodeId::default(),
        },
    })
}

#[pyfunction]
#[pyo3(signature = (path=None, title=None, screen=None))]
fn window(path: Option<String>, title: Option<String>, screen: Option<u32>) -> PyWindow {
    PyWindow {
        inner: WindowSchematic {
            title,
            path,
            screen,
            screen_rect: None,
        },
    }
}

pub(super) fn register_builders(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(coordinate, module)?)?;
    module.add_function(wrap_pyfunction!(theme, module)?)?;
    module.add_function(wrap_pyfunction!(timeline, module)?)?;
    module.add_function(wrap_pyfunction!(tabs, module)?)?;
    module.add_function(wrap_pyfunction!(hsplit, module)?)?;
    module.add_function(wrap_pyfunction!(vsplit, module)?)?;
    module.add_function(wrap_pyfunction!(graph, module)?)?;
    module.add_function(wrap_pyfunction!(viewport, module)?)?;
    module.add_function(wrap_pyfunction!(component_monitor, module)?)?;
    module.add_function(wrap_pyfunction!(inspector, module)?)?;
    module.add_function(wrap_pyfunction!(hierarchy, module)?)?;
    module.add_function(wrap_pyfunction!(schematic_tree, module)?)?;
    module.add_function(wrap_pyfunction!(data_overview, module)?)?;
    module.add_function(wrap_pyfunction!(video_stream, module)?)?;
    module.add_function(wrap_pyfunction!(sensor_view, module)?)?;
    module.add_function(wrap_pyfunction!(log_stream, module)?)?;
    module.add_function(wrap_pyfunction!(action_pane, module)?)?;
    module.add_function(wrap_pyfunction!(query_table, module)?)?;
    module.add_function(wrap_pyfunction!(query_plot, module)?)?;
    module.add_function(wrap_pyfunction!(glb, module)?)?;
    module.add_function(wrap_pyfunction!(joint, module)?)?;
    module.add_function(wrap_pyfunction!(object_3d, module)?)?;
    module.add_function(wrap_pyfunction!(line_3d, module)?)?;
    module.add_function(wrap_pyfunction!(vector_arrow, module)?)?;
    module.add_function(wrap_pyfunction!(world_mesh, module)?)?;
    module.add_function(wrap_pyfunction!(window, module)?)?;
    Ok(())
}
