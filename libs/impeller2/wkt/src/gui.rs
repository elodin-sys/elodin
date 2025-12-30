use crate::Color;
use impeller2::component::Asset;
use impeller2::types::EntityId;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de, de::DeserializeOwned};
use std::collections::HashMap;
use std::fmt;
use std::ops::Range;
use std::time::Duration;
use strum::{EnumString, IntoStaticStr, VariantNames};

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(bound = "T: Serialize + DeserializeOwned")]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::TypePath,))]
#[cfg_attr(feature = "bevy", type_path = "impeller2::wkt::gui::Schematic")]
pub struct Schematic<T = ()> {
    pub elems: Vec<SchematicElem<T>>,
    #[serde(default)]
    pub theme: Option<ThemeConfig>,
}

#[cfg(feature = "bevy")]
impl bevy::prelude::Asset for Schematic {}

#[cfg(feature = "bevy")]
impl bevy::asset::VisitAssetDependencies for Schematic {
    fn visit_dependencies(&self, _visit: &mut impl FnMut(bevy::asset::UntypedAssetId)) {}
}

impl<T> Default for Schematic<T> {
    fn default() -> Self {
        Self {
            elems: Default::default(),
            theme: None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(bound = "T: Serialize + DeserializeOwned")]
pub enum SchematicElem<T = ()> {
    Panel(Panel<T>),
    Object3d(Object3D<T>),
    Line3d(Line3d<T>),
    VectorArrow(VectorArrow3d<T>),
    Window(WindowSchematic),
    Theme(ThemeConfig),
}

impl<T> SchematicElem<T> {
    pub fn clear_aux(self) -> SchematicElem<()> {
        match self {
            SchematicElem::Panel(panel) => SchematicElem::Panel(panel.map_aux(|_| ())),
            SchematicElem::Object3d(obj) => SchematicElem::Object3d(obj.map_aux(|_| ())),
            SchematicElem::Line3d(line) => SchematicElem::Line3d(line.map_aux(|_| ())),
            SchematicElem::VectorArrow(arrow) => SchematicElem::VectorArrow(arrow.map_aux(|_| ())),
            SchematicElem::Window(window) => SchematicElem::Window(window),
            SchematicElem::Theme(theme) => SchematicElem::Theme(theme.clone()),
        }
    }
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

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(bound = "T: Serialize + DeserializeOwned")]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub enum Panel<T = ()> {
    Viewport(Viewport<T>),
    VSplit(Split<T>),
    HSplit(Split<T>),
    Graph(Graph<T>),
    ComponentMonitor(ComponentMonitor),
    ActionPane(ActionPane),
    QueryTable(QueryTable),
    QueryPlot(QueryPlot<T>),
    Tabs(Vec<Panel<T>>),
    Inspector,
    Hierarchy,
    SchematicTree(Option<String>),
    DataOverview(Option<String>),
    Dashboard(Box<Dashboard<T>>),
}

impl<T> Panel<T> {
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
            Panel::Dashboard(d) => d.root.name.as_deref().unwrap_or("Dashboard"),
        }
    }

    pub fn collapse(&self) -> &Panel<T> {
        match self {
            Panel::Tabs(panels) if panels.len() == 1 => panels[0].collapse(),
            this => this,
        }
    }

    pub fn children(&self) -> &[Panel<T>] {
        match self {
            Panel::HSplit(split) | Panel::VSplit(split) => &split.panels,
            Panel::Tabs(panels) => panels,
            _ => &[],
        }
    }

    pub fn children_mut(&mut self) -> &mut [Panel<T>] {
        match self {
            Panel::HSplit(split) | Panel::VSplit(split) => &mut split.panels,
            Panel::Tabs(panels) => panels,
            _ => &mut [],
        }
    }

    pub fn map_aux<U>(&self, f: impl Fn(&T) -> U) -> Panel<U> {
        match self {
            Panel::HSplit(split) => Panel::HSplit(split.map_aux(f)),
            Panel::VSplit(split) => Panel::VSplit(split.map_aux(f)),
            Panel::Tabs(panels) => Panel::Tabs(panels.iter().map(|p| p.map_aux(&f)).collect()),
            Panel::Graph(graph) => Panel::Graph(graph.map_aux(f)),
            Panel::ComponentMonitor(component_monitor) => {
                Panel::ComponentMonitor(component_monitor.clone())
            }
            Panel::ActionPane(action_pane) => Panel::ActionPane(action_pane.clone()),
            Panel::QueryTable(query_table) => Panel::QueryTable(query_table.clone()),
            Panel::QueryPlot(query_plot) => Panel::QueryPlot(query_plot.map_aux(f)),
            Panel::Hierarchy => Panel::Hierarchy,
            Panel::SchematicTree(name) => Panel::SchematicTree(name.clone()),
            Panel::Inspector => Panel::Inspector,
            Panel::DataOverview(name) => Panel::DataOverview(name.clone()),
            Panel::Viewport(v) => Panel::Viewport(v.map_aux(f)),
            Panel::Dashboard(d) => Panel::Dashboard(Box::new(d.map_aux(f))),
        }
    }

    pub fn aux(&self) -> Option<&T> {
        match self {
            Panel::Graph(graph) => Some(&graph.aux),
            Panel::QueryPlot(query_plot) => Some(&query_plot.aux),
            Panel::Viewport(v) => Some(&v.aux),
            Panel::Dashboard(d) => Some(&d.aux),
            _ => None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(bound = "T: Serialize + DeserializeOwned")]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Split<T = ()> {
    pub panels: Vec<Panel<T>>,
    pub shares: HashMap<usize, f32>,
    pub active: bool,
    #[serde(default)]
    pub name: Option<String>,
}

impl<T> Split<T> {
    pub fn map_aux<U>(&self, f: impl Fn(&T) -> U) -> Split<U> {
        Split {
            panels: self.panels.iter().map(|p| p.map_aux(&f)).collect(),
            shares: self.shares.clone(),
            active: self.active,
            name: self.name.clone(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(bound = "T: Serialize + DeserializeOwned")]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Viewport<T = ()> {
    pub fov: f32,
    pub active: bool,
    pub show_grid: bool,
    pub show_arrows: bool,
    pub hdr: bool,
    pub name: Option<String>,
    pub pos: Option<String>,
    pub look_at: Option<String>,
    #[serde(default)]
    pub local_arrows: Vec<VectorArrow3d>,
    pub aux: T,
}

impl<T> Viewport<T> {
    pub fn map_aux<U>(&self, f: impl Fn(&T) -> U) -> Viewport<U> {
        Viewport {
            fov: self.fov,
            active: self.active,
            show_grid: self.show_grid,
            show_arrows: self.show_arrows,
            hdr: self.hdr,
            name: self.name.clone(),
            pos: self.pos.clone(),
            look_at: self.look_at.clone(),
            local_arrows: self.local_arrows.clone(),
            aux: f(&self.aux),
        }
    }
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            fov: 45.0,
            active: false,
            show_grid: false,
            show_arrows: true,
            hdr: false,
            name: None,
            pos: None,
            look_at: None,
            local_arrows: Vec::new(),
            aux: (),
        }
    }
}

impl Asset for Panel {
    const NAME: &'static str = "panel";
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(bound = "T: Serialize + DeserializeOwned")]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Graph<T = ()> {
    pub eql: String,
    pub name: Option<String>,
    #[serde(default)]
    pub graph_type: GraphType,
    #[serde(default)]
    pub locked: bool,
    pub auto_y_range: bool,
    pub y_range: Range<f64>,
    pub aux: T,
    pub colors: Vec<crate::Color>,
}

impl<T> Graph<T> {
    pub fn map_aux<U>(&self, f: impl Fn(&T) -> U) -> Graph<U> {
        Graph {
            eql: self.eql.clone(),
            name: self.name.clone(),
            graph_type: self.graph_type,
            locked: self.locked,
            auto_y_range: self.auto_y_range,
            y_range: self.y_range.clone(),
            aux: f(&self.aux),
            colors: self.colors.clone(),
        }
    }
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
#[serde(bound = "T: Serialize + DeserializeOwned")]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Line3d<T = ()> {
    pub eql: String,
    pub line_width: f32,
    pub color: Color,
    pub perspective: bool,
    pub aux: T,
}

impl<T> Line3d<T> {
    pub fn map_aux<U>(&self, f: impl Fn(&T) -> U) -> Line3d<U> {
        Line3d {
            eql: self.eql.clone(),
            line_width: self.line_width,
            color: self.color,
            perspective: self.perspective,
            aux: f(&self.aux),
        }
    }
}

impl<T: Serialize + DeserializeOwned> Asset for Line3d<T> {
    const NAME: &'static str = "line_3d";
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "T: Serialize + DeserializeOwned")]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct VectorArrow3d<T = ()> {
    pub vector: String,
    pub origin: Option<String>,
    #[serde(default = "VectorArrow3d::<T>::default_scale")]
    pub scale: f64,
    pub name: Option<String>,
    #[serde(default = "VectorArrow3d::<T>::default_color")]
    pub color: Color,
    #[serde(default)]
    #[serde(alias = "in_body_frame")]
    pub body_frame: bool,
    #[serde(default)]
    pub normalize: bool,
    #[serde(default = "VectorArrow3d::<T>::default_show_name")]
    pub show_name: bool,
    #[serde(default = "VectorArrow3d::<T>::default_thickness")]
    pub thickness: ArrowThickness,
    #[serde(default = "VectorArrow3d::<T>::default_label_position")]
    pub label_position: LabelPosition,
    pub aux: T,
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

impl<T> VectorArrow3d<T> {
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

    pub fn map_aux<U>(&self, f: impl Fn(&T) -> U) -> VectorArrow3d<U> {
        VectorArrow3d {
            vector: self.vector.clone(),
            origin: self.origin.clone(),
            scale: self.scale,
            name: self.name.clone(),
            color: self.color,
            body_frame: self.body_frame,
            normalize: self.normalize,
            show_name: self.show_name,
            thickness: self.thickness,
            label_position: self.label_position.clone(),
            aux: f(&self.aux),
        }
    }
}

impl<T: Serialize + DeserializeOwned> Asset for VectorArrow3d<T> {
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

pub fn default_glb_scale() -> f32 {
    1.0
}

pub fn default_glb_translate() -> (f32, f32, f32) {
    (0.0, 0.0, 0.0)
}

pub fn default_glb_rotate() -> (f32, f32, f32) {
    (0.0, 0.0, 0.0)
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
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "T: Serialize + DeserializeOwned")]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Object3D<T = ()> {
    pub eql: String,
    pub mesh: Object3DMesh,
    pub aux: T,
}

impl<T> Object3D<T> {
    pub fn map_aux<U>(&self, f: impl FnOnce(&T) -> U) -> Object3D<U> {
        Object3D {
            eql: self.eql.clone(),
            mesh: self.mesh.clone(),
            aux: f(&self.aux),
        }
    }
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
#[serde(bound = "T: Serialize + DeserializeOwned")]
pub struct QueryPlot<T = ()> {
    pub name: String,
    pub query: String,
    pub refresh_interval: Duration,
    pub auto_refresh: bool,
    pub color: Color,
    pub query_type: QueryType,
    pub aux: T,
}

impl<T> QueryPlot<T> {
    pub fn map_aux<U>(&self, f: impl Fn(&T) -> U) -> QueryPlot<U> {
        QueryPlot {
            name: self.name.clone(),
            query: self.query.clone(),
            refresh_interval: self.refresh_interval,
            auto_refresh: self.auto_refresh,
            color: self.color,
            query_type: self.query_type,
            aux: f(&self.aux),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Deserialize, Serialize)]
pub enum QueryType {
    #[default]
    EQL,
    SQL,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Dashboard<T = ()> {
    pub root: DashboardNode<T>,
    pub aux: T,
}

impl<T> Dashboard<T> {
    pub fn map_aux<U>(&self, f: impl Fn(&T) -> U) -> Dashboard<U> {
        Dashboard {
            root: self.root.map_aux(&f),
            aux: f(&self.aux),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct DashboardNode<T> {
    pub name: Option<String>,
    pub display: Display,
    pub box_sizing: BoxSizing,
    pub position_type: PositionType,
    pub overflow: Overflow,
    pub overflow_clip_margin: OverflowClipMargin,
    pub left: Val,
    pub right: Val,
    pub top: Val,
    pub bottom: Val,
    pub width: Val,
    pub height: Val,
    pub min_width: Val,
    pub min_height: Val,
    pub max_width: Val,
    pub max_height: Val,
    pub aspect_ratio: Option<f32>,
    pub align_items: AlignItems,
    pub justify_items: JustifyItems,
    pub align_self: AlignSelf,
    pub justify_self: JustifySelf,
    pub align_content: AlignContent,
    pub justify_content: JustifyContent,
    pub margin: UiRect,
    pub padding: UiRect,
    pub border: UiRect,
    pub flex_direction: FlexDirection,
    pub flex_wrap: FlexWrap,
    pub flex_grow: f32,
    pub flex_shrink: f32,
    pub flex_basis: Val,
    pub row_gap: Val,
    pub column_gap: Val,
    pub children: Vec<DashboardNode<T>>,
    pub color: Color,
    pub text: Option<String>,
    pub font_size: f32,
    pub text_color: Color,
    pub aux: T,
}

impl<T> DashboardNode<T> {
    pub fn map_aux<U>(&self, f: impl Fn(&T) -> U) -> DashboardNode<U> {
        DashboardNode {
            name: self.name.clone(),
            display: self.display,
            box_sizing: self.box_sizing,
            position_type: self.position_type,
            overflow: self.overflow,
            overflow_clip_margin: self.overflow_clip_margin.clone(),
            left: self.left.clone(),
            right: self.right.clone(),
            top: self.top.clone(),
            bottom: self.bottom.clone(),
            width: self.width.clone(),
            height: self.height.clone(),
            min_width: self.min_width.clone(),
            min_height: self.min_height.clone(),
            max_width: self.max_width.clone(),
            max_height: self.max_height.clone(),
            aspect_ratio: self.aspect_ratio,
            align_items: self.align_items,
            justify_items: self.justify_items,
            align_self: self.align_self,
            justify_self: self.justify_self,
            align_content: self.align_content,
            justify_content: self.justify_content,
            margin: self.margin.clone(),
            padding: self.padding.clone(),
            border: self.border.clone(),
            flex_direction: self.flex_direction,
            flex_wrap: self.flex_wrap,
            flex_grow: self.flex_grow,
            flex_shrink: self.flex_shrink,
            flex_basis: self.flex_basis.clone(),
            row_gap: self.row_gap.clone(),
            column_gap: self.column_gap.clone(),
            children: self.children.iter().map(|c| c.map_aux(&f)).collect(),
            color: self.color,
            text: self.text.clone(),
            font_size: self.font_size,
            text_color: self.text_color,
            aux: f(&self.aux),
        }
    }
}

#[derive(
    Debug, Clone, Copy, Default, Deserialize, Serialize, EnumString, IntoStaticStr, VariantNames,
)]
#[strum(serialize_all = "kebab-case")]
pub enum PositionType {
    #[default]
    Relative,
    Absolute,
}

#[derive(
    Debug, Clone, Copy, Default, Deserialize, Serialize, EnumString, IntoStaticStr, VariantNames,
)]
#[strum(serialize_all = "kebab-case")]
pub enum AlignItems {
    #[default]
    Default,
    Start,
    End,
    FlexStart,
    FlexEnd,
    Center,
    Baseline,
    Stretch,
}

#[derive(
    Debug, Clone, Copy, Default, Deserialize, Serialize, EnumString, IntoStaticStr, VariantNames,
)]
#[strum(serialize_all = "kebab-case")]
pub enum JustifyItems {
    #[default]
    Default,
    Start,
    End,
    Center,
    Baseline,
    Stretch,
}

#[derive(
    Debug, Clone, Copy, Default, Deserialize, Serialize, EnumString, IntoStaticStr, VariantNames,
)]
#[strum(serialize_all = "kebab-case")]
pub enum Display {
    #[default]
    Flex,
    Grid,
    Block,
    None,
}

#[derive(
    Debug, Clone, Copy, Default, Deserialize, Serialize, EnumString, IntoStaticStr, VariantNames,
)]
#[strum(serialize_all = "kebab-case")]
pub enum BoxSizing {
    #[default]
    BorderBox,
    ContentBox,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
pub struct Overflow {
    pub x: OverflowAxis,
    pub y: OverflowAxis,
}

#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct OverflowClipMargin {
    pub visual_box: OverflowClipBox,
    pub margin: f32,
}

#[derive(
    Debug, Clone, Copy, Default, Deserialize, Serialize, EnumString, IntoStaticStr, VariantNames,
)]
#[strum(serialize_all = "kebab-case")]
pub enum OverflowClipBox {
    #[default]
    ContentBox,
    PaddingBox,
    BorderBox,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UiRect {
    pub left: Val,
    pub right: Val,
    pub top: Val,
    pub bottom: Val,
}

impl Default for UiRect {
    fn default() -> Self {
        Self {
            left: Val::Px("0.0".to_string()),
            right: Val::Px("0.0".to_string()),
            top: Val::Px("0.0".to_string()),
            bottom: Val::Px("0.0".to_string()),
        }
    }
}

#[derive(
    Debug, Default, Copy, Clone, Deserialize, Serialize, EnumString, IntoStaticStr, VariantNames,
)]
#[strum(serialize_all = "kebab-case")]
pub enum OverflowAxis {
    #[default]
    Visible,
    Clip,
    Hidden,
    Scroll,
}

#[derive(
    Debug, Default, Copy, Clone, Deserialize, Serialize, EnumString, IntoStaticStr, VariantNames,
)]
#[strum(serialize_all = "kebab-case")]
pub enum FlexDirection {
    #[default]
    Row,
    Column,
    RowReverse,
    ColumnReverse,
}

#[derive(
    Debug, Default, Copy, Clone, Deserialize, Serialize, EnumString, IntoStaticStr, VariantNames,
)]
#[strum(serialize_all = "kebab-case")]
pub enum FlexWrap {
    #[default]
    NoWrap,
    Wrap,
    WrapReverse,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub enum Val {
    #[default]
    Auto,
    Px(String),
    Percent(String),
    Vw(String),
    Vh(String),
    VMin(String),
    VMax(String),
}

impl Val {
    pub fn eql(&self) -> &str {
        match self {
            Val::Auto => "",
            Val::Px(v) => v,
            Val::Percent(v) => v,
            Val::Vw(v) => v,
            Val::Vh(v) => v,
            Val::VMin(v) => v,
            Val::VMax(v) => v,
        }
    }
}

#[derive(
    Debug, Default, Copy, Clone, Deserialize, Serialize, EnumString, IntoStaticStr, VariantNames,
)]
#[strum(serialize_all = "kebab-case")]
pub enum AlignSelf {
    #[default]
    Auto,
    Start,
    End,
    FlexStart,
    FlexEnd,
    Center,
    Baseline,
    Stretch,
}

#[derive(
    Debug, Default, Copy, Clone, Deserialize, Serialize, EnumString, IntoStaticStr, VariantNames,
)]
#[strum(serialize_all = "kebab-case")]
pub enum JustifySelf {
    #[default]
    Auto,
    Start,
    End,
    Center,
    Baseline,
    Stretch,
}

#[derive(
    Debug, Default, Copy, Clone, Deserialize, Serialize, EnumString, IntoStaticStr, VariantNames,
)]
#[strum(serialize_all = "kebab-case")]
pub enum AlignContent {
    #[default]
    Default,
    Start,
    End,
    FlexStart,
    FlexEnd,
    Center,
    Stretch,
    SpaceBetween,
    SpaceEvenly,
    SpaceAround,
}

#[derive(
    Debug, Default, Copy, Clone, Deserialize, Serialize, EnumString, IntoStaticStr, VariantNames,
)]
#[strum(serialize_all = "kebab-case")]
pub enum JustifyContent {
    #[default]
    Default,
    Start,
    End,
    FlexStart,
    FlexEnd,
    Center,
    Stretch,
    SpaceBetween,
    SpaceEvenly,
    SpaceAround,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct DashboardText<T = ()> {
    pub text: String,
    pub aux: T,
}

impl<T> DashboardText<T> {
    pub fn map_aux<U>(&self, f: impl FnOnce(&T) -> U) -> DashboardText<U> {
        DashboardText {
            text: self.text.clone(),
            aux: f(&self.aux),
        }
    }
}
