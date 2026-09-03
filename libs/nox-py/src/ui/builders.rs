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

#[pyclass(name = "Color", module = "elodin.ui")]
#[derive(Clone, Copy, Debug)]
pub struct PyColor {
    pub(crate) inner: Color,
}

#[pyclass(name = "Environment", module = "elodin.ui")]
#[derive(Clone, Debug)]
pub struct PyEnvironment {
    pub(crate) inner: EnvironmentConfig,
}

#[pyclass(name = "Sun", module = "elodin.ui")]
#[derive(Clone, Copy, Debug)]
pub struct PySun {
    pub(crate) inner: SunConfig,
}

#[pyclass(name = "Atmosphere", module = "elodin.ui")]
#[derive(Clone, Copy, Debug)]
pub struct PyAtmosphere {
    pub(crate) inner: AtmosphereConfig,
}

#[pyclass(name = "Earth", module = "elodin.ui")]
#[derive(Clone, Copy, Debug)]
pub struct PyEarth {
    pub(crate) inner: EarthConfig,
}

#[pyclass(name = "Bloom", module = "elodin.ui")]
#[derive(Clone, Debug)]
pub struct PyBloom {
    pub(crate) inner: BloomConfig,
}

#[pyclass(name = "Icon", module = "elodin.ui")]
#[derive(Clone, Debug)]
pub struct PyIcon {
    pub(crate) inner: Object3DIcon,
}

#[pyclass(name = "VisibilityRange", module = "elodin.ui")]
#[derive(Clone, Debug)]
pub struct PyVisibilityRange {
    pub(crate) inner: VisRange,
}

#[pyclass(name = "Thruster", module = "elodin.ui")]
#[derive(Clone, Debug)]
pub struct PyThruster {
    pub(crate) inner: Thruster,
}

#[pyclass(name = "ThrusterLight", module = "elodin.ui")]
#[derive(Clone, Debug)]
pub struct PyThrusterLight {
    pub(crate) inner: ThrusterLight,
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

pub(crate) fn extract_environment(obj: &Bound<'_, PyAny>) -> PyResult<EnvironmentConfig> {
    Ok(obj.extract::<PyRef<'_, PyEnvironment>>()?.inner.clone())
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

fn parse_named_color(name: &str) -> PyResult<Color> {
    color_from_name(name)
        .ok_or_else(|| PyValueError::new_err(format!("unknown color name {name:?}")))
}

fn extract_color(obj: &Bound<'_, PyAny>) -> PyResult<Color> {
    if let Ok(color) = obj.extract::<PyRef<'_, PyColor>>() {
        return Ok(color.inner);
    }
    if let Ok(name) = obj.extract::<String>() {
        return parse_named_color(&name);
    }
    Err(PyTypeError::new_err(
        "color must be an elodin.ui.Color or named color string",
    ))
}

fn extract_optional_color(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Color>> {
    match obj {
        None => Ok(None),
        Some(obj) if obj.is_none() => Ok(None),
        Some(obj) => extract_color(obj).map(Some),
    }
}

fn parse_frame(frame: &str) -> PyResult<GeoFrame> {
    GeoFrame::from_str(frame)
        .map_err(|_| PyValueError::new_err(format!("unknown coordinate frame {frame:?}")))
}

/// Accept `str`, an object with `__str__` (e.g. `ui.Expr`), or a list of those.
pub(crate) fn extract_eql(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(s) = obj.extract::<String>() {
        return Ok(s);
    }
    if let Ok(list) = obj.downcast::<pyo3::types::PyList>() {
        let mut parts = Vec::with_capacity(list.len());
        for item in list.iter() {
            parts.push(extract_eql(&item)?);
        }
        return Ok(parts.join(", "));
    }
    Ok(obj.str()?.to_string())
}

fn extract_optional_eql(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<String>> {
    match obj {
        None => Ok(None),
        Some(o) if o.is_none() => Ok(None),
        Some(o) => Ok(Some(extract_eql(o)?)),
    }
}

#[pyfunction]
#[pyo3(signature = (r, g, b, a=255))]
fn color(r: u8, g: u8, b: u8, a: u8) -> PyColor {
    PyColor {
        inner: Color::rgba(
            f32::from(r) / 255.0,
            f32::from(g) / 255.0,
            f32::from(b) / 255.0,
            f32::from(a) / 255.0,
        ),
    }
}

#[pyfunction]
#[pyo3(signature = (
    azimuth=None,
    elevation=None,
    illuminance=100_000.0,
    shadows=true,
    direction=None,
))]
fn sun(
    azimuth: Option<f32>,
    elevation: Option<f32>,
    illuminance: f32,
    shadows: bool,
    direction: Option<(f32, f32, f32)>,
) -> PySun {
    PySun {
        inner: SunConfig {
            azimuth_deg: azimuth,
            elevation_deg: elevation,
            illuminance,
            shadows,
            direction,
        },
    }
}

#[pyfunction]
#[pyo3(signature = (
    origin=(0.0, 0.0, 0.0),
    inner_radius=6_360_000.0,
    outer_radius=6_460_000.0,
    ground_albedo=(0.3, 0.3, 0.3),
    raymarched=false,
))]
fn atmosphere(
    origin: (f64, f64, f64),
    inner_radius: f32,
    outer_radius: f32,
    ground_albedo: (f32, f32, f32),
    raymarched: bool,
) -> PyResult<PyAtmosphere> {
    if outer_radius <= inner_radius {
        return Err(PyValueError::new_err(
            "atmosphere outer_radius must be greater than inner_radius",
        ));
    }
    Ok(PyAtmosphere {
        inner: AtmosphereConfig {
            origin,
            inner_radius,
            outer_radius,
            ground_albedo,
            raymarched,
        },
    })
}

#[pyfunction]
#[pyo3(signature = (
    stars_density=0.05,
    stars_size=0.40,
    stars_brightness=1.88,
    city_lights_density=0.05,
    city_lights_size=1.0,
    city_lights_height=0.0,
    city_lights_brightness=0.05,
    airglow_density=0.55,
    airglow_size=1.05,
    airglow_brightness=1.45,
    night_map_brightness=0.05,
))]
#[allow(clippy::too_many_arguments)]
fn earth(
    stars_density: f32,
    stars_size: f32,
    stars_brightness: f32,
    city_lights_density: f32,
    city_lights_size: f32,
    city_lights_height: f32,
    city_lights_brightness: f32,
    airglow_density: f32,
    airglow_size: f32,
    airglow_brightness: f32,
    night_map_brightness: f32,
) -> PyEarth {
    PyEarth {
        inner: EarthConfig {
            stars: EarthStarsConfig {
                density: stars_density,
                size: stars_size,
                brightness: stars_brightness,
            },
            city_lights: EarthCityLightsConfig {
                density: city_lights_density,
                size: city_lights_size,
                height: city_lights_height,
                brightness: city_lights_brightness,
            },
            airglow: EarthAirglowConfig {
                density: airglow_density,
                size: airglow_size,
                brightness: airglow_brightness,
            },
            night_map: EarthNightMapConfig {
                brightness: night_map_brightness,
            },
        }
        .clamp(),
    }
}

#[pyfunction]
#[pyo3(signature = (sun=None, ambient=1.0, sky=None, atmosphere=None, earth=None))]
fn environment(
    sun: Option<Bound<'_, PyAny>>,
    ambient: f32,
    sky: Option<Bound<'_, PyAny>>,
    atmosphere: Option<Bound<'_, PyAny>>,
    earth: Option<Bound<'_, PyAny>>,
) -> PyResult<PyEnvironment> {
    Ok(PyEnvironment {
        inner: EnvironmentConfig {
            sun: sun
                .map(|obj| -> PyResult<_> { Ok(obj.extract::<PyRef<'_, PySun>>()?.inner) })
                .transpose()?,
            ambient_scale: ambient,
            sky_color: extract_optional_color(sky.as_ref())?,
            atmosphere: atmosphere
                .map(|obj| -> PyResult<_> { Ok(obj.extract::<PyRef<'_, PyAtmosphere>>()?.inner) })
                .transpose()?,
            earth: earth
                .map(|obj| -> PyResult<_> { Ok(obj.extract::<PyRef<'_, PyEarth>>()?.inner) })
                .transpose()?,
        },
    })
}

#[pyfunction]
#[pyo3(signature = (preset="natural", intensity=None, threshold=None, threshold_softness=None))]
fn bloom(
    preset: &str,
    intensity: Option<f32>,
    threshold: Option<f32>,
    threshold_softness: Option<f32>,
) -> PyResult<PyBloom> {
    let preset = match preset {
        "natural" => BloomPreset::Natural,
        "old_school" => BloomPreset::OldSchool,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown bloom preset {other:?}"
            )));
        }
    };
    if [intensity, threshold, threshold_softness]
        .into_iter()
        .flatten()
        .any(|value| value < 0.0)
    {
        return Err(PyValueError::new_err("bloom values must be non-negative"));
    }
    Ok(PyBloom {
        inner: BloomConfig {
            preset,
            intensity,
            threshold,
            threshold_softness,
        },
    })
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
    played_color: Option<&Bound<'_, PyAny>>,
    future_color: Option<&Bound<'_, PyAny>>,
    follow_latest: bool,
    range: Option<String>,
) -> PyResult<PyTimeline> {
    let mut cfg = TimelineConfig {
        follow_latest,
        range,
        ..TimelineConfig::default()
    };
    if let Some(value) = played_color {
        cfg.played_color = extract_color(value)?;
    }
    if let Some(value) = future_color {
        cfg.future_color = extract_color(value)?;
    }
    Ok(PyTimeline { inner: cfg })
}

#[pyfunction]
#[pyo3(signature = (*children, share=None))]
fn tabs(children: Vec<Bound<'_, PyAny>>, share: Option<f32>) -> PyResult<PyPanel> {
    let panels = children
        .into_iter()
        .map(|c| extract_panel(&c).map(|(p, _)| p))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyPanel {
        inner: Panel::Tabs(panels),
        share,
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
    eql: &Bound<'_, PyAny>,
    name: Option<String>,
    graph_type: Option<&str>,
    locked: bool,
    auto_y_range: bool,
    y_min: Option<f64>,
    y_max: Option<f64>,
    colors: Option<Vec<Bound<'_, PyAny>>>,
    share: Option<f32>,
) -> PyResult<PyPanel> {
    let eql = extract_eql(eql)?;
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
        .map(|c| extract_color(&c))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyPanel {
        inner: Panel::Graph(Graph {
            eql,
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
    frustums_color=None,
    projection_color=None,
    frustums_thickness=0.006,
    show_view_cube=true,
    view_cube_frame=None,
    effects=true,
    hdr=false,
    cinematic=false,
    bloom=None,
    ev100=None,
    pos=None,
    look_at=None,
    up=None,
    smoothing=0.0,
    frame=None,
    arrows=None,
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
    frustums_color: Option<&Bound<'_, PyAny>>,
    projection_color: Option<&Bound<'_, PyAny>>,
    frustums_thickness: f32,
    show_view_cube: bool,
    view_cube_frame: Option<&str>,
    effects: bool,
    hdr: bool,
    cinematic: bool,
    bloom: Option<Bound<'_, PyAny>>,
    ev100: Option<f32>,
    pos: Option<&Bound<'_, PyAny>>,
    look_at: Option<&Bound<'_, PyAny>>,
    up: Option<&Bound<'_, PyAny>>,
    smoothing: f32,
    frame: Option<&str>,
    arrows: Option<Vec<Bound<'_, PyAny>>>,
    share: Option<f32>,
) -> PyResult<PyPanel> {
    if frustums_thickness <= 0.0 {
        return Err(PyValueError::new_err(
            "frustums_thickness must be greater than zero",
        ));
    }
    if !smoothing.is_finite() || smoothing < 0.0 {
        return Err(PyValueError::new_err(
            "smoothing must be a finite non-negative number",
        ));
    }
    let local_arrows = arrows
        .unwrap_or_default()
        .into_iter()
        .map(|arrow| Ok(arrow.extract::<PyRef<'_, PyVectorArrow>>()?.inner.clone()))
        .collect::<PyResult<Vec<_>>>()?;
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
            frustums_color: extract_optional_color(frustums_color)?
                .unwrap_or_else(default_viewport_frustums_color),
            projection_color: extract_optional_color(projection_color)?
                .unwrap_or_else(default_viewport_projection_color),
            frustums_thickness,
            show_view_cube,
            view_cube_frame: view_cube_frame.map(parse_frame).transpose()?,
            effects,
            hdr,
            cinematic,
            bloom: bloom
                .map(|obj| -> PyResult<_> {
                    Ok(obj.extract::<PyRef<'_, PyBloom>>()?.inner.clone())
                })
                .transpose()?,
            ev100,
            name,
            pos: extract_optional_eql(pos)?,
            look_at: extract_optional_eql(look_at)?,
            up: extract_optional_eql(up)?,
            smoothing,
            frame: frame.map(parse_frame).transpose()?,
            local_arrows,
            node_id: NodeId::default(),
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
#[pyo3(signature = (eql, source=None, display="NED", name=None, share=None))]
fn geo_position_gauge(
    eql: &Bound<'_, PyAny>,
    source: Option<&str>,
    display: &str,
    name: Option<String>,
    share: Option<f32>,
) -> PyResult<PyPanel> {
    let display = DisplayFrame::from_str_ci(display)
        .ok_or_else(|| PyValueError::new_err("display must be ECEF, NED, ENU, or LLA"))?;
    Ok(PyPanel {
        inner: Panel::GeoPositionGauge(GeoPositionGauge {
            eql: extract_eql(eql)?,
            source: source.map(parse_frame).transpose()?,
            display,
            name,
            node_id: NodeId::default(),
        }),
        share,
    })
}

#[pyfunction]
#[pyo3(signature = (eql, source=None, display=None, reference=None, name=None, share=None))]
fn orientation_gauge(
    eql: &Bound<'_, PyAny>,
    source: Option<&str>,
    display: Option<&str>,
    reference: Option<(f64, f64, f64, f64)>,
    name: Option<String>,
    share: Option<f32>,
) -> PyResult<PyPanel> {
    Ok(PyPanel {
        inner: Panel::OrientationGauge(OrientationGauge {
            eql: extract_eql(eql)?,
            source: source.map(parse_frame).transpose()?,
            display: display.map(parse_frame).transpose()?,
            reference: reference.map(|(x, y, z, w)| [x, y, z, w]),
            name,
            node_id: NodeId::default(),
        }),
        share,
    })
}

#[pyfunction]
#[pyo3(signature = (eql, source=None, reference=None, name=None, share=None))]
fn horizon_gauge(
    eql: &Bound<'_, PyAny>,
    source: Option<&str>,
    reference: Option<(f64, f64, f64, f64)>,
    name: Option<String>,
    share: Option<f32>,
) -> PyResult<PyPanel> {
    Ok(PyPanel {
        inner: Panel::HorizonGauge(HorizonGauge {
            eql: extract_eql(eql)?,
            source: source.map(parse_frame).transpose()?,
            reference: reference.map(|(x, y, z, w)| [x, y, z, w]),
            name,
            node_id: NodeId::default(),
        }),
        share,
    })
}

#[pyfunction]
#[pyo3(signature = (share=None))]
fn inspector(share: Option<f32>) -> PyPanel {
    PyPanel {
        inner: Panel::Inspector,
        share,
    }
}

#[pyfunction]
#[pyo3(signature = (share=None))]
fn hierarchy(share: Option<f32>) -> PyPanel {
    PyPanel {
        inner: Panel::Hierarchy,
        share,
    }
}

#[pyfunction]
#[pyo3(signature = (name=None, share=None))]
fn schematic_tree(name: Option<String>, share: Option<f32>) -> PyPanel {
    PyPanel {
        inner: Panel::SchematicTree(name),
        share,
    }
}

#[pyfunction]
#[pyo3(signature = (name=None, share=None))]
fn data_overview(name: Option<String>, share: Option<f32>) -> PyPanel {
    PyPanel {
        inner: Panel::DataOverview(name),
        share,
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
    color: Option<&Bound<'_, PyAny>>,
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
            color: extract_optional_color(color)?.unwrap_or(Color::YALK),
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
#[pyo3(signature = (
    path,
    scale=None,
    translate=None,
    rotate=None,
    emissivity=0.0,
    glow=0.0,
    glow_color=None,
))]
fn glb(
    path: &str,
    scale: Option<f32>,
    translate: Option<(f32, f32, f32)>,
    rotate: Option<(f32, f32, f32)>,
    emissivity: f32,
    glow: f32,
    glow_color: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyMesh> {
    Ok(PyMesh {
        inner: Object3DMesh::Glb {
            path: path.to_string(),
            scale: scale.unwrap_or(default_glb_scale()),
            translate: translate.unwrap_or_else(default_glb_translate),
            rotate: rotate.unwrap_or_else(default_glb_rotate),
            animations: Vec::new(),
            emissivity,
            glow,
            glow_color: extract_optional_color(glow_color)?,
        },
    })
}

fn material(color: Option<&Bound<'_, PyAny>>, emissivity: f32) -> PyResult<Material> {
    Ok(Material {
        base_color: extract_optional_color(color)?.unwrap_or(Color::WHITE),
        emissivity,
    })
}

#[pyfunction]
#[pyo3(signature = (radius=1.0, color=None, emissivity=0.0))]
fn sphere(radius: f32, color: Option<&Bound<'_, PyAny>>, emissivity: f32) -> PyResult<PyMesh> {
    Ok(PyMesh {
        inner: Object3DMesh::Mesh {
            mesh: impeller2_wkt::Mesh::Sphere { radius },
            material: material(color, emissivity)?,
        },
    })
}

#[pyfunction]
#[pyo3(name = "box", signature = (x=1.0, y=1.0, z=1.0, color=None, emissivity=0.0))]
fn box_mesh(
    x: f32,
    y: f32,
    z: f32,
    color: Option<&Bound<'_, PyAny>>,
    emissivity: f32,
) -> PyResult<PyMesh> {
    Ok(PyMesh {
        inner: Object3DMesh::Mesh {
            mesh: impeller2_wkt::Mesh::Box { x, y, z },
            material: material(color, emissivity)?,
        },
    })
}

#[pyfunction]
#[pyo3(signature = (radius=1.0, height=1.0, color=None, emissivity=0.0))]
fn cylinder(
    radius: f32,
    height: f32,
    color: Option<&Bound<'_, PyAny>>,
    emissivity: f32,
) -> PyResult<PyMesh> {
    Ok(PyMesh {
        inner: Object3DMesh::Mesh {
            mesh: impeller2_wkt::Mesh::Cylinder { radius, height },
            material: material(color, emissivity)?,
        },
    })
}

#[pyfunction]
#[pyo3(signature = (width=1.0, depth=1.0, color=None, emissivity=0.0))]
fn plane(
    width: f32,
    depth: f32,
    color: Option<&Bound<'_, PyAny>>,
    emissivity: f32,
) -> PyResult<PyMesh> {
    Ok(PyMesh {
        inner: Object3DMesh::Mesh {
            mesh: impeller2_wkt::Mesh::Plane { width, depth },
            material: material(color, emissivity)?,
        },
    })
}

#[pyfunction]
#[pyo3(signature = (
    scale=None,
    color=None,
    error_covariance_cholesky=None,
    error_covariance=None,
    error_confidence_interval=70.0,
    show_grid=false,
    grid_color=None,
))]
#[allow(clippy::too_many_arguments)]
fn ellipsoid(
    scale: Option<&Bound<'_, PyAny>>,
    color: Option<&Bound<'_, PyAny>>,
    error_covariance_cholesky: Option<&Bound<'_, PyAny>>,
    error_covariance: Option<&Bound<'_, PyAny>>,
    error_confidence_interval: f32,
    show_grid: bool,
    grid_color: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyMesh> {
    if error_covariance_cholesky.is_some() && error_covariance.is_some() {
        return Err(PyValueError::new_err(
            "set only one of error_covariance_cholesky or error_covariance",
        ));
    }
    Ok(PyMesh {
        inner: Object3DMesh::Ellipsoid {
            scale: extract_optional_eql(scale)?.unwrap_or_else(default_ellipsoid_scale_expr),
            color: extract_optional_color(color)?.unwrap_or_else(default_ellipsoid_color),
            error_covariance_cholesky: extract_optional_eql(error_covariance_cholesky)?,
            error_covariance: extract_optional_eql(error_covariance)?,
            error_confidence_interval,
            show_grid,
            grid_color: extract_optional_color(grid_color)?
                .unwrap_or_else(default_ellipsoid_grid_color),
        },
    })
}

#[pyfunction]
#[pyo3(signature = (joint, rotation_vector))]
fn joint(joint: &str, rotation_vector: &Bound<'_, PyAny>) -> PyResult<PyJoint> {
    Ok(PyJoint {
        inner: JointAnimation {
            joint_name: joint.to_string(),
            eql_expr: extract_eql(rotation_vector)?,
        },
    })
}

#[pyfunction]
#[pyo3(signature = (min=0.0, max=f32::MAX, fade_distance=0.0))]
fn visibility_range(min: f32, max: f32, fade_distance: f32) -> PyVisibilityRange {
    PyVisibilityRange {
        inner: VisRange {
            min,
            max,
            fade_distance,
        },
    }
}

#[pyfunction]
#[pyo3(signature = (path=None, builtin=None, color=None, size=32.0, visibility=None))]
fn icon(
    path: Option<String>,
    builtin: Option<String>,
    color: Option<&Bound<'_, PyAny>>,
    size: f32,
    visibility: Option<Bound<'_, PyAny>>,
) -> PyResult<PyIcon> {
    let source = match (path, builtin) {
        (Some(path), None) => Object3DIconSource::Path(path),
        (None, Some(name)) => Object3DIconSource::Builtin(name),
        _ => {
            return Err(PyValueError::new_err(
                "icon requires exactly one of path or builtin",
            ));
        }
    };
    Ok(PyIcon {
        inner: Object3DIcon {
            source,
            color: extract_optional_color(color)?.unwrap_or_else(default_icon_color),
            size,
            visibility_range: visibility
                .map(|obj| -> PyResult<_> {
                    Ok(obj.extract::<PyRef<'_, PyVisibilityRange>>()?.inner.clone())
                })
                .transpose()?,
        },
    })
}

#[pyfunction]
#[pyo3(signature = (
    color,
    intensity,
    range=30.0,
    offset=0.0,
    spot_angle=None,
    shadows=false,
))]
fn thruster_light(
    color: (f32, f32, f32),
    intensity: f32,
    range: f32,
    offset: f32,
    spot_angle: Option<f32>,
    shadows: bool,
) -> PyThrusterLight {
    PyThrusterLight {
        inner: ThrusterLight {
            color,
            intensity,
            range,
            offset,
            spot_angle,
            shadows,
        },
    }
}

#[pyfunction]
#[pyo3(signature = (
    intensity,
    position,
    name=None,
    direction=None,
    body_frame=false,
    effect="plume",
    extra_effects=None,
    emission_rate=None,
    cutoff=0.02,
    scale=1.0,
    light=None,
))]
#[allow(clippy::too_many_arguments)]
fn thruster(
    intensity: &Bound<'_, PyAny>,
    position: (f32, f32, f32),
    name: Option<String>,
    direction: Option<(f32, f32, f32)>,
    body_frame: bool,
    effect: &str,
    extra_effects: Option<Vec<String>>,
    emission_rate: Option<f32>,
    cutoff: f32,
    scale: f32,
    light: Option<Bound<'_, PyAny>>,
) -> PyResult<PyThruster> {
    Ok(PyThruster {
        inner: Thruster {
            name,
            body_frame,
            position,
            direction,
            intensity: extract_eql(intensity)?,
            effect: effect.to_string(),
            extra_effects: extra_effects.unwrap_or_default(),
            emission_rate,
            cutoff,
            scale,
            light: light
                .map(|obj| -> PyResult<_> {
                    Ok(obj.extract::<PyRef<'_, PyThrusterLight>>()?.inner.clone())
                })
                .transpose()?,
        },
    })
}

#[pyfunction]
#[pyo3(signature = (
    eql,
    mesh,
    frame=None,
    frame_orientation=None,
    orientation=None,
    animate=None,
    icon=None,
    thrusters=None,
    visibility=None,
))]
#[allow(clippy::too_many_arguments)]
fn object_3d(
    eql: &Bound<'_, PyAny>,
    mesh: Bound<'_, PyAny>,
    frame: Option<&str>,
    frame_orientation: Option<&str>,
    orientation: Option<&str>,
    animate: Option<Vec<Bound<'_, PyAny>>>,
    icon: Option<Bound<'_, PyAny>>,
    thrusters: Option<Vec<Bound<'_, PyAny>>>,
    visibility: Option<Bound<'_, PyAny>>,
) -> PyResult<PyObject3D> {
    let eql = extract_eql(eql)?;
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
            eql,
            mesh,
            frame: frame.map(parse_frame).transpose()?,
            frame_orientation: frame_orientation.map(parse_frame).transpose()?,
            orientation,
            icon: icon
                .map(|obj| -> PyResult<_> { Ok(obj.extract::<PyRef<'_, PyIcon>>()?.inner.clone()) })
                .transpose()?,
            thrusters: thrusters
                .unwrap_or_default()
                .into_iter()
                .map(|obj| Ok(obj.extract::<PyRef<'_, PyThruster>>()?.inner.clone()))
                .collect::<PyResult<Vec<_>>>()?,
            mesh_visibility_range: visibility
                .map(|obj| -> PyResult<_> {
                    Ok(obj.extract::<PyRef<'_, PyVisibilityRange>>()?.inner.clone())
                })
                .transpose()?,
            node_id: NodeId::default(),
        },
    })
}

#[pyfunction]
#[pyo3(signature = (eql, line_width=1.0, color=None, future_color=None, perspective=true, frame=None))]
fn line_3d(
    eql: &Bound<'_, PyAny>,
    line_width: f32,
    color: Option<&Bound<'_, PyAny>>,
    future_color: Option<&Bound<'_, PyAny>>,
    perspective: bool,
    frame: Option<&str>,
) -> PyResult<PyLine3d> {
    Ok(PyLine3d {
        inner: Line3d {
            eql: extract_eql(eql)?,
            line_width,
            color: extract_optional_color(color)?,
            future_color: extract_optional_color(future_color)?,
            perspective,
            frame: frame.map(parse_frame).transpose()?,
            node_id: NodeId::default(),
        },
    })
}

#[pyfunction]
#[pyo3(signature = (
    vector,
    origin=None,
    name=None,
    color=None,
    scale=None,
    body_frame=false,
    normalize=false,
    show_name=true,
    thickness=0.1,
    label_position=None,
    frame=None,
))]
#[allow(clippy::too_many_arguments)]
fn vector_arrow(
    vector: &Bound<'_, PyAny>,
    origin: Option<&Bound<'_, PyAny>>,
    name: Option<String>,
    color: Option<&Bound<'_, PyAny>>,
    scale: Option<f64>,
    body_frame: bool,
    normalize: bool,
    show_name: bool,
    thickness: f32,
    label_position: Option<&str>,
    frame: Option<&str>,
) -> PyResult<PyVectorArrow> {
    let label_position = match label_position {
        None => LabelPosition::None,
        Some(value) if value.ends_with('m') => {
            let magnitude = value[..value.len() - 1].parse::<f32>().map_err(|_| {
                PyValueError::new_err("absolute label_position must look like '2.5m'")
            })?;
            LabelPosition::Absolute(magnitude)
        }
        Some(value) => LabelPosition::Proportionate(value.parse::<f32>().map_err(|_| {
            PyValueError::new_err("label_position must be a proportion or length like '2.5m'")
        })?),
    };
    Ok(PyVectorArrow {
        inner: VectorArrow3d {
            vector: extract_eql(vector)?,
            origin: extract_optional_eql(origin)?,
            scale: scale.unwrap_or(1.0),
            name,
            color: extract_optional_color(color)?.unwrap_or(Color::WHITE),
            body_frame,
            normalize,
            show_name,
            thickness: ArrowThickness::new(thickness),
            label_position,
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
#[pyo3(signature = (path=None, title=None, screen=None, rect=None))]
fn window(
    path: Option<String>,
    title: Option<String>,
    screen: Option<u32>,
    rect: Option<(u32, u32, u32, u32)>,
) -> PyWindow {
    PyWindow {
        inner: WindowSchematic {
            title,
            path,
            screen,
            screen_rect: rect.map(|(x, y, width, height)| WindowRect {
                x,
                y,
                width,
                height,
            }),
        },
    }
}

pub(super) fn register_builders(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(color, module)?)?;
    module.add_function(wrap_pyfunction!(sun, module)?)?;
    module.add_function(wrap_pyfunction!(atmosphere, module)?)?;
    module.add_function(wrap_pyfunction!(earth, module)?)?;
    module.add_function(wrap_pyfunction!(environment, module)?)?;
    module.add_function(wrap_pyfunction!(bloom, module)?)?;
    module.add_function(wrap_pyfunction!(coordinate, module)?)?;
    module.add_function(wrap_pyfunction!(theme, module)?)?;
    module.add_function(wrap_pyfunction!(timeline, module)?)?;
    module.add_function(wrap_pyfunction!(tabs, module)?)?;
    module.add_function(wrap_pyfunction!(hsplit, module)?)?;
    module.add_function(wrap_pyfunction!(vsplit, module)?)?;
    module.add_function(wrap_pyfunction!(graph, module)?)?;
    module.add_function(wrap_pyfunction!(viewport, module)?)?;
    module.add_function(wrap_pyfunction!(component_monitor, module)?)?;
    module.add_function(wrap_pyfunction!(geo_position_gauge, module)?)?;
    module.add_function(wrap_pyfunction!(orientation_gauge, module)?)?;
    module.add_function(wrap_pyfunction!(horizon_gauge, module)?)?;
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
    module.add_function(wrap_pyfunction!(sphere, module)?)?;
    module.add_function(wrap_pyfunction!(box_mesh, module)?)?;
    module.add_function(wrap_pyfunction!(cylinder, module)?)?;
    module.add_function(wrap_pyfunction!(plane, module)?)?;
    module.add_function(wrap_pyfunction!(ellipsoid, module)?)?;
    module.add_function(wrap_pyfunction!(joint, module)?)?;
    module.add_function(wrap_pyfunction!(visibility_range, module)?)?;
    module.add_function(wrap_pyfunction!(icon, module)?)?;
    module.add_function(wrap_pyfunction!(thruster_light, module)?)?;
    module.add_function(wrap_pyfunction!(thruster, module)?)?;
    module.add_function(wrap_pyfunction!(object_3d, module)?)?;
    module.add_function(wrap_pyfunction!(line_3d, module)?)?;
    module.add_function(wrap_pyfunction!(vector_arrow, module)?)?;
    module.add_function(wrap_pyfunction!(world_mesh, module)?)?;
    module.add_function(wrap_pyfunction!(window, module)?)?;
    Ok(())
}
