use crate::Color;
use impeller2::component::Asset;
use impeller2::types::EntityId;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use std::collections::HashMap;
use std::fmt;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct NodeId(pub u64);

static NODE_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

impl NodeId {
    pub fn next() -> Self {
        Self(NODE_ID_COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

fn default_true() -> bool {
    true
}

pub fn default_timeline_played_color() -> Color {
    Color::YELLOW
}

pub fn default_timeline_future_color() -> Color {
    Color::WHITE
}

pub fn default_viewport_frustums_color() -> Color {
    Color::YELLOW
}

pub fn default_viewport_frustums_thickness() -> f32 {
    0.006
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::TypePath,))]
#[cfg_attr(feature = "bevy", type_path = "impeller2::wkt::gui::Schematic")]
pub struct Schematic {
    pub elems: Vec<SchematicElem>,
    #[serde(default)]
    pub theme: Option<ThemeConfig>,
    #[serde(default)]
    pub timeline: Option<TimelineConfig>,
}

#[cfg(feature = "bevy")]
impl bevy::prelude::Asset for Schematic {}

#[cfg(feature = "bevy")]
impl bevy::asset::VisitAssetDependencies for Schematic {
    fn visit_dependencies(&self, _visit: &mut impl FnMut(bevy::asset::UntypedAssetId)) {}
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SchematicElem {
    Panel(Panel),
    Object3d(Object3D),
    Line3d(Line3d),
    VectorArrow(VectorArrow3d),
    Window(WindowSchematic),
    Theme(ThemeConfig),
    Timeline(TimelineConfig),
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::TypePath))]
pub struct WindowRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct WindowSchematic {
    pub title: Option<String>,
    pub path: Option<String>,
    pub screen: Option<u32>,
    #[serde(default)]
    pub screen_rect: Option<WindowRect>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct ThemeConfig {
    pub mode: Option<String>,
    pub scheme: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TimelineConfig {
    #[serde(default = "default_timeline_played_color")]
    pub played_color: Color,
    #[serde(default = "default_timeline_future_color")]
    pub future_color: Color,
    #[serde(default)]
    pub follow_latest: bool,
}

impl Default for TimelineConfig {
    fn default() -> Self {
        Self {
            played_color: default_timeline_played_color(),
            future_color: default_timeline_future_color(),
            follow_latest: false,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub enum Panel {
    Viewport(Viewport),
    VSplit(Split),
    HSplit(Split),
    Graph(Graph),
    ComponentMonitor(ComponentMonitor),
    ActionPane(ActionPane),
    QueryTable(QueryTable),
    QueryPlot(QueryPlot),
    Tabs(Vec<Panel>),
    Inspector,
    Hierarchy,
    SchematicTree(Option<String>),
    DataOverview(Option<String>),
    VideoStream(VideoStream),
    SensorView(SensorView),
    LogStream(LogStream),
}

impl Panel {
    pub fn label(&self) -> &str {
        match self {
            Panel::Viewport(viewport) => viewport.name.as_deref().unwrap_or("Viewport"),
            Panel::VSplit(_) => "Vertical Split",
            Panel::HSplit(_) => "Horizontal Split",
            Panel::Graph(graph) => graph.name.as_deref().unwrap_or("Graph"),
            Panel::ComponentMonitor(monitor) => {
                monitor.name.as_deref().unwrap_or(&monitor.component_name)
            }
            Panel::ActionPane(action_pane) => action_pane.name.as_str(),
            Panel::QueryTable(query_table) => query_table.name.as_deref().unwrap_or("Query Table"),
            Panel::QueryPlot(query_plot) => &query_plot.name,
            Panel::Tabs(_) => "Tabs",
            Panel::Inspector => "Inspector",
            Panel::Hierarchy => "Hierarchy",
            Panel::SchematicTree(name) => name.as_deref().unwrap_or("Tree"),
            Panel::DataOverview(name) => name.as_deref().unwrap_or("Data Overview"),
            Panel::VideoStream(v) => v.name.as_deref().unwrap_or("Video Stream"),
            Panel::SensorView(v) => v.name.as_deref().unwrap_or("Sensor View"),
            Panel::LogStream(l) => l.name.as_deref().unwrap_or("Log Stream"),
        }
    }

    pub fn collapse(&self) -> &Panel {
        match self {
            Panel::Tabs(panels) if panels.len() == 1 => panels[0].collapse(),
            this => this,
        }
    }

    pub fn children(&self) -> &[Panel] {
        match self {
            Panel::HSplit(split) | Panel::VSplit(split) => &split.panels,
            Panel::Tabs(panels) => panels,
            _ => &[],
        }
    }

    pub fn children_mut(&mut self) -> &mut [Panel] {
        match self {
            Panel::HSplit(split) | Panel::VSplit(split) => &mut split.panels,
            Panel::Tabs(panels) => panels,
            _ => &mut [],
        }
    }

    pub fn node_id(&self) -> Option<NodeId> {
        match self {
            Panel::Graph(graph) => Some(graph.node_id),
            Panel::QueryPlot(query_plot) => Some(query_plot.node_id),
            Panel::Viewport(v) => Some(v.node_id),
            _ => None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Split {
    pub panels: Vec<Panel>,
    pub shares: HashMap<usize, f32>,
    pub active: bool,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Viewport {
    pub fov: f32,
    #[serde(default)]
    pub near: Option<f32>,
    #[serde(default)]
    pub far: Option<f32>,
    #[serde(default, alias = "aspect_ratio")]
    pub aspect: Option<f32>,
    pub active: bool,
    pub show_grid: bool,
    pub show_arrows: bool,
    #[serde(default)]
    pub create_frustum: bool,
    #[serde(default, alias = "show_frustum")]
    pub show_frustums: bool,
    #[serde(default = "default_viewport_frustums_color")]
    pub frustums_color: Color,
    #[serde(default = "default_viewport_frustums_thickness")]
    pub frustums_thickness: f32,
    #[serde(default = "default_true")]
    pub show_view_cube: bool,
    pub hdr: bool,
    pub name: Option<String>,
    pub pos: Option<String>,
    pub look_at: Option<String>,
    /// Optional camera up vector in world frame. EQL that evaluates to a 3-vector (e.g. "(0,0,1)" or "pose.direction(0,1,1)" for body-frame direction).
    pub up: Option<String>,
    #[serde(default)]
    pub local_arrows: Vec<VectorArrow3d>,
    #[serde(default)]
    pub node_id: NodeId,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            fov: 45.0,
            near: None,
            far: None,
            aspect: None,
            active: false,
            show_grid: false,
            show_arrows: true,
            create_frustum: false,
            show_frustums: false,
            frustums_color: default_viewport_frustums_color(),
            frustums_thickness: default_viewport_frustums_thickness(),
            show_view_cube: true,
            hdr: false,
            name: None,
            pos: None,
            look_at: None,
            up: None,
            local_arrows: Vec::new(),
            node_id: NodeId::default(),
        }
    }
}

impl Asset for Panel {
    const NAME: &'static str = "panel";
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Graph {
    pub eql: String,
    pub name: Option<String>,
    #[serde(default)]
    pub graph_type: GraphType,
    #[serde(default)]
    pub locked: bool,
    pub auto_y_range: bool,
    pub y_range: Range<f64>,
    #[serde(default)]
    pub node_id: NodeId,
    pub colors: Vec<crate::Color>,
}

#[derive(Serialize, Deserialize, Clone, Copy, Hash, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub enum GraphType {
    #[default]
    Line,
    Point,
    Bar,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Line3d {
    pub eql: String,
    pub line_width: f32,
    pub color: Color,
    pub perspective: bool,
    #[serde(default)]
    pub node_id: NodeId,
}

impl Asset for Line3d {
    const NAME: &'static str = "line_3d";
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct VectorArrow3d {
    pub vector: String,
    pub origin: Option<String>,
    #[serde(default = "VectorArrow3d::default_scale")]
    pub scale: f64,
    pub name: Option<String>,
    #[serde(default = "VectorArrow3d::default_color")]
    pub color: Color,
    #[serde(default)]
    #[serde(alias = "in_body_frame")]
    pub body_frame: bool,
    #[serde(default)]
    pub normalize: bool,
    #[serde(default = "VectorArrow3d::default_show_name")]
    pub show_name: bool,
    #[serde(default = "VectorArrow3d::default_thickness")]
    pub thickness: ArrowThickness,
    #[serde(default = "VectorArrow3d::default_label_position")]
    pub label_position: LabelPosition,
    #[serde(default)]
    pub node_id: NodeId,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
/// The position of a label.
pub enum LabelPosition {
    /// No label position.
    #[default]
    None,
    /// A value from [0, 1] meant to represent some proportion of a whole.
    Proportionate(f32),
    /// An absolute magnitude in meters.
    Absolute(f32),
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct ArrowThickness(pub f32);

impl ArrowThickness {
    pub const DEFAULT: f32 = 0.1;
    const MIN: f32 = 0.001;

    pub fn new(raw: f32) -> Self {
        if !raw.is_finite() {
            return Self(Self::DEFAULT);
        }

        Self(Self::round_to_precision(raw.max(Self::MIN)))
    }

    pub fn value(self) -> f32 {
        if !self.0.is_finite() {
            return Self::DEFAULT;
        }

        Self::round_to_precision(self.0.max(Self::MIN))
    }

    pub fn round_to_precision(value: f32) -> f32 {
        (value * 1000.0).round() / 1000.0
    }
}

impl Serialize for ArrowThickness {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_f32(self.value())
    }
}

impl<'de> Deserialize<'de> for ArrowThickness {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ThicknessVisitor;

        impl<'de> de::Visitor<'de> for ThicknessVisitor {
            type Value = ArrowThickness;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a numeric arrow thickness")
            }

            fn visit_f32<E>(self, value: f32) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(ArrowThickness::new(value))
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(ArrowThickness::new(value as f32))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(ArrowThickness::new(value as f32))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(ArrowThickness::new(value as f32))
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                v.parse::<f32>()
                    .map(ArrowThickness::new)
                    .map_err(|_| E::custom(format!("invalid arrow thickness '{v}'")))
            }

            fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                self.visit_str(&v)
            }
        }

        deserializer.deserialize_any(ThicknessVisitor)
    }
}

impl Default for ArrowThickness {
    fn default() -> Self {
        Self(Self::DEFAULT)
    }
}

impl VectorArrow3d {
    fn default_scale() -> f64 {
        1.0
    }

    fn default_color() -> Color {
        Color::WHITE
    }

    fn default_show_name() -> bool {
        true
    }

    fn default_thickness() -> ArrowThickness {
        ArrowThickness::default()
    }

    fn default_label_position() -> LabelPosition {
        LabelPosition::default()
    }
}

impl Asset for VectorArrow3d {
    const NAME: &'static str = "vector_arrow";
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Camera;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct BodyAxes {
    pub entity_id: EntityId,
    pub scale: f32,
}

impl Asset for BodyAxes {
    const NAME: &'static str = "body_axes";
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub enum Mesh {
    Sphere { radius: f32 },
    Box { x: f32, y: f32, z: f32 },
    Cylinder { radius: f32, height: f32 },
    Plane { width: f32, depth: f32 },
}

impl Mesh {
    pub fn cuboid(x: f32, y: f32, z: f32) -> Self {
        Self::Box { x, y, z }
    }

    pub fn sphere(radius: f32) -> Self {
        Self::Sphere { radius }
    }

    pub fn plane(width: f32, depth: f32) -> Self {
        Self::Plane { width, depth }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Glb(pub String);

impl Asset for Mesh {
    const NAME: &'static str = "mesh";
}

impl Asset for Glb {
    const NAME: &'static str = "glb";
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Material {
    pub base_color: Color,
    #[serde(default)]
    pub emissivity: f32,
}

impl Material {
    pub fn color(r: f32, g: f32, b: f32) -> Self {
        Material {
            base_color: Color::rgb(r, g, b),
            emissivity: 0.0,
        }
    }

    pub fn color_with_alpha(r: f32, g: f32, b: f32, a: f32) -> Self {
        Material {
            base_color: Color::rgba(r, g, b, a),
            emissivity: 0.0,
        }
    }

    pub fn color_with_emissivity(r: f32, g: f32, b: f32, emissivity: f32) -> Self {
        Material {
            base_color: Color::rgb(r, g, b),
            emissivity,
        }
    }

    pub fn with_color(color: Color) -> Self {
        Material {
            base_color: color,
            emissivity: 0.0,
        }
    }
}

impl Asset for Material {
    const NAME: &'static str = "material";
}

pub fn default_ellipsoid_scale_expr() -> String {
    "(1, 1, 1)".to_string()
}

pub fn default_ellipsoid_color() -> Color {
    Color::WHITE
}

pub fn default_ellipsoid_confidence_interval() -> f32 {
    70.0
}

pub fn default_ellipsoid_show_grid() -> bool {
    false
}

pub fn default_ellipsoid_grid_color() -> Color {
    Color::BLACK
}

pub fn default_glb_scale() -> f32 {
    1.0
}

pub fn default_glb_translate() -> (f32, f32, f32) {
    (0.0, 0.0, 0.0)
}

pub fn default_glb_rotate() -> (f32, f32, f32) {
    (0.0, 0.0, 0.0)
}

fn default_glb_animations() -> Vec<JointAnimation> {
    Vec::new()
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct JointAnimation {
    pub joint_name: String,
    pub eql_expr: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub enum Object3DMesh {
    Glb {
        path: String,
        #[serde(default = "default_glb_scale")]
        scale: f32,
        #[serde(default = "default_glb_translate")]
        translate: (f32, f32, f32),
        #[serde(default = "default_glb_rotate")]
        rotate: (f32, f32, f32),
        #[serde(default = "default_glb_animations")]
        animations: Vec<JointAnimation>,
    },
    Mesh {
        mesh: Mesh,
        material: Material,
    },
    Ellipsoid {
        #[serde(default = "default_ellipsoid_scale_expr")]
        scale: String,
        #[serde(default = "default_ellipsoid_color")]
        color: Color,
        #[serde(default)]
        error_covariance_cholesky: Option<String>,
        #[serde(default = "default_ellipsoid_confidence_interval")]
        error_confidence_interval: f32,
        #[serde(default = "default_ellipsoid_show_grid")]
        show_grid: bool,
        #[serde(default = "default_ellipsoid_grid_color")]
        grid_color: Color,
    },
}

impl Object3DMesh {
    /// Create a GLB mesh with default scale (1.0), translate (0,0,0), and rotate (0,0,0)
    pub fn glb(path: impl Into<String>) -> Self {
        Self::Glb {
            path: path.into(),
            scale: default_glb_scale(),
            translate: default_glb_translate(),
            rotate: default_glb_rotate(),
            animations: default_glb_animations(),
        }
    }

    pub fn path(&self) -> Option<&str> {
        match self {
            Self::Glb { path, .. } => Some(path),
            _ => None,
        }
    }
}

impl fmt::Display for Object3DMesh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Glb { path, .. } => write!(f, "GLB '{}'", path),
            Self::Mesh { mesh, .. } => match mesh {
                Mesh::Sphere { radius } => write!(f, "Sphere(radius={})", radius),
                Mesh::Box { x, y, z } => write!(f, "Box({}x{}x{})", x, y, z),
                Mesh::Cylinder { radius, height } => {
                    write!(f, "Cylinder(radius={}, height={})", radius, height)
                }
                Mesh::Plane { width, depth } => write!(f, "Plane({}x{})", width, depth),
            },
            Self::Ellipsoid { .. } => write!(f, "Ellipsoid"),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Object3DIconSource {
    Path(String),
    Builtin(String),
}

pub fn default_icon_color() -> Color {
    Color {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    }
}

pub fn default_icon_size() -> f32 {
    32.0
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VisRange {
    #[serde(default)]
    pub min: f32,
    #[serde(default = "vis_range_default_max")]
    pub max: f32,
    #[serde(default)]
    pub fade_distance: f32,
}

fn vis_range_default_max() -> f32 {
    f32::MAX
}

impl Default for VisRange {
    fn default() -> Self {
        Self {
            min: 0.0,
            max: f32::MAX,
            fade_distance: 0.0,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Object3DIcon {
    pub source: Object3DIconSource,
    #[serde(default = "default_icon_color")]
    pub color: Color,
    #[serde(default = "default_icon_size")]
    pub size: f32,
    #[serde(default)]
    pub visibility_range: Option<VisRange>,
}

/// Maps a built-in icon name (snake_case) to its Material Icons Unicode codepoint.
pub fn builtin_icon_char(name: &str) -> Option<char> {
    let cp: u32 = match name {
        "satellite_alt" => 0xeb3a,
        "satellite" => 0xe562,
        "rocket_launch" => 0xeb9b,
        "rocket" => 0xeba5,
        "flight" => 0xe539,
        "flight_takeoff" => 0xe53d,
        "public" => 0xe80b,
        "language" => 0xe894,
        "circle" => 0xef4a,
        "fiber_manual_record" => 0xe061,
        "star" => 0xe838,
        "star_outline" => 0xe83a,
        "location_on" => 0xe0c8,
        "place" => 0xe55f,
        "adjust" => 0xe39e,
        "gps_fixed" => 0xe1b3,
        "my_location" => 0xe55c,
        "explore" => 0xe87a,
        "navigation" => 0xe55d,
        "near_me" => 0xe569,
        "diamond" => 0xead5,
        "hexagon" => 0xeb39,
        "change_history" => 0xe86b,
        "lens" => 0xe3fa,
        "panorama_fish_eye" => 0xe40c,
        "radio_button_unchecked" => 0xe836,
        "brightness_1" => 0xe3a6,
        "flare" => 0xef4e,
        "wb_sunny" => 0xe430,
        "bolt" => 0xea0b,
        _ => return None,
    };
    char::from_u32(cp)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Object3D {
    pub eql: String,
    pub mesh: Object3DMesh,
    #[serde(default)]
    pub icon: Option<Object3DIcon>,
    #[serde(default)]
    pub mesh_visibility_range: Option<VisRange>,
    #[serde(default)]
    pub node_id: NodeId,
}

impl Asset for Object3D {
    const NAME: &'static str = "object3d";
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct ComponentMonitor {
    /// The component name that we are monitoring.
    ///
    /// NOTE: It may be nice to allow this to be an EQL expression that we
    /// monitor, which can be a simple component_name.
    pub component_name: String,
    pub name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VideoStream {
    /// Message name containing H.264 video frames
    pub msg_name: String,
    /// Display name for the tile
    pub name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SensorView {
    /// Message name for the sensor camera frame data (e.g. "drone.scene_cam")
    pub msg_name: String,
    /// Display name for the tile
    pub name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LogStream {
    /// Message name for the log entry stream (e.g. "fsw.log")
    pub msg_name: String,
    /// Display name for the tile
    pub name: Option<String>,
}

fn default_format() -> String {
    "rgba".to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorCameraConfig {
    pub entity_name: String,
    pub camera_name: String,
    pub width: u32,
    pub height: u32,
    pub fov_degrees: f32,
    pub near: f32,
    pub far: f32,
    pub pos_offset: [f64; 3],
    pub look_at_offset: [f64; 3],
    #[serde(default = "default_format")]
    pub format: String,
    #[serde(default)]
    pub effect: String,
    #[serde(default)]
    pub effect_params: HashMap<String, f64>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct QueryTable {
    pub name: Option<String>,
    pub query: String,
    pub query_type: QueryType,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct ActionPane {
    pub name: String,
    pub lua: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct QueryPlot {
    pub name: String,
    pub query: String,
    pub refresh_interval: Duration,
    pub auto_refresh: bool,
    pub color: Color,
    pub query_type: QueryType,
    /// Plot mode: TimeSeries (default) or XY for arbitrary X/Y plotting
    #[serde(default)]
    pub plot_mode: PlotMode,
    /// Optional X-axis label (only used in XY mode)
    #[serde(default)]
    pub x_label: Option<String>,
    /// Optional Y-axis label
    #[serde(default)]
    pub y_label: Option<String>,
    #[serde(default)]
    pub node_id: NodeId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Deserialize, Serialize)]
pub enum QueryType {
    #[default]
    EQL,
    SQL,
}

/// Plot mode for query plots - determines how the X-axis is interpreted and displayed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Deserialize, Serialize)]
pub enum PlotMode {
    /// Time-series mode: X-axis represents time, labels formatted as durations
    #[default]
    TimeSeries,
    /// XY mode: X-axis represents arbitrary numeric values, labels formatted as numbers
    XY,
}
