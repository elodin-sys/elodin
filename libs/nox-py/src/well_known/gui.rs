use crate::*;

use impeller2::component::Asset;
use impeller2_wkt::{Graph, GraphComponent, GraphType, Split};
use nox_ecs::nox::Vector3;
use numpy::PyArrayLike1;
use pyo3::exceptions::PyValueError;

use crate::EntityId;

#[pyclass]
#[derive(Clone)]
pub struct Panel {
    inner: impeller2_wkt::Panel,
}

#[pymethods]
impl Panel {
    #[staticmethod]
    pub fn sidebars(inner: Panel) -> Self {
        let inner = impeller2_wkt::Panel::HSplit(Split {
            panels: vec![
                impeller2_wkt::Panel::Hierarchy,
                inner.inner,
                impeller2_wkt::Panel::Inspector,
            ],
            shares: [(0, 0.2), (1, 0.6), (2, 0.2)].into_iter().collect(),
            active: true,
        });
        Self { inner }
    }

    #[staticmethod]
    pub fn inspector() -> Self {
        Self {
            inner: impeller2_wkt::Panel::Inspector,
        }
    }

    #[staticmethod]
    pub fn hierarchy() -> Self {
        Self {
            inner: impeller2_wkt::Panel::Hierarchy,
        }
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (track_entity = None, track_rotation = true, fov = 45.0, active = false, pos = None, looking_at = None, show_grid = false, hdr = false, name = None))]
    pub fn viewport(
        track_entity: Option<EntityId>,
        track_rotation: bool,
        fov: f32,
        active: bool,
        pos: Option<PyArrayLike1<f32>>,
        looking_at: Option<PyArrayLike1<f32>>,
        show_grid: bool,
        hdr: bool,
        name: Option<String>,
    ) -> PyResult<Self> {
        let pos = if let Some(arr) = pos {
            let slice = arr.as_slice()?;
            if slice.len() != 3 {
                return Err(PyValueError::new_err("transform must be 3x1 array"));
            }
            Vector3::new(slice[0], slice[1], slice[2])
        } else {
            Vector3::new(5.0, 5.0, 10.0)
        };
        let mut viewport = impeller2_wkt::Viewport {
            track_entity: track_entity.map(|x| x.inner),
            fov,
            active,
            pos,
            track_rotation,
            show_grid,
            hdr,
            name,
            ..Default::default()
        };
        if let Some(pos) = looking_at {
            let pos = pos.as_slice()?;
            if pos.len() != 3 {
                return Err(PyValueError::new_err("transform must be 3x1 array"));
            }
            let pos = Vector3::new(pos[0], pos[1], pos[2]);
            viewport = viewport.looking_at(pos);
        }
        Ok(Self {
            inner: impeller2_wkt::Panel::Viewport(viewport),
        })
    }

    pub fn asset_name(&self) -> &'static str {
        impeller2_wkt::Panel::NAME
    }

    pub fn bytes(&self) -> Result<PyBufBytes, Error> {
        let bytes = postcard::to_allocvec(&self.inner).unwrap().into();
        Ok(PyBufBytes { bytes })
    }

    #[staticmethod]
    #[pyo3(signature = (*panels, active = false))]
    pub fn vsplit(panels: Vec<Panel>, active: bool) -> Self {
        Self {
            inner: impeller2_wkt::Panel::VSplit(Split {
                panels: panels.into_iter().map(|x| x.inner).collect(),
                active,
                shares: Default::default(),
            }),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (*panels, active = false))]
    pub fn hsplit(panels: Vec<Panel>, active: bool) -> Self {
        Self {
            inner: impeller2_wkt::Panel::HSplit(Split {
                panels: panels.into_iter().map(|x| x.inner).collect(),
                active,
                shares: Default::default(),
            }),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (*entities, name = None, ty = None))]
    pub fn graph(
        entities: Vec<GraphEntity>,
        name: Option<String>,
        ty: Option<String>,
    ) -> PyResult<Self> {
        let graph_type = match ty.as_deref() {
            None | Some("line") => GraphType::Line,
            Some("bar") => GraphType::Bar,
            Some("point") => GraphType::Point,
            _ => {
                return Err(PyValueError::new_err(
                    "invalid graph type please select either: line, point, or bar",
                ));
            }
        };
        let entities = entities
            .into_iter()
            .map(|x| impeller2_wkt::GraphEntity {
                entity_id: x.entity.inner,
                components: x.components,
            })
            .collect();
        Ok(Self {
            inner: impeller2_wkt::Panel::Graph(Graph {
                name,
                entities,
                graph_type,
                auto_y_range: true,
                y_range: 0.0..1.0,
            }),
        })
    }
}

#[pyclass]
#[derive(Clone)]
pub struct GraphEntity {
    pub entity: EntityId,
    pub components: Vec<GraphComponent>,
}

#[pymethods]
impl GraphEntity {
    #[new]
    #[pyo3(signature = (entity, *objs))]
    pub fn new(py: Python<'_>, entity: EntityId, objs: Vec<PyObject>) -> Result<Self, Error> {
        let components = objs
            .into_iter()
            .map(|obj| {
                let indexer = if let Ok(indexer) = obj.extract::<ShapeIndexer>(py) {
                    indexer
                } else {
                    Component::index(py, obj.clone_ref(py))?
                };
                let component = GraphComponent {
                    component_id: ComponentId::new(&indexer.component_name),
                    indexes: indexer.indexes(),
                    color: vec![],
                };
                Ok::<_, Error>(component)
            })
            .collect::<Result<_, _>>()?;
        Ok(Self { entity, components })
    }
}

#[pyclass]
pub struct Line3d {
    inner: impeller2_wkt::Line3d,
}

#[pymethods]
impl Line3d {
    #[new]
    #[pyo3(signature = (entity, component_name=None, line_width=None, color=None, index=None, perspective=None))]
    pub fn new(
        entity: EntityId,
        component_name: Option<String>,
        line_width: Option<f32>,
        color: Option<Color>,
        index: Option<Vec<usize>>,
        perspective: Option<bool>,
    ) -> PyResult<Self> {
        use impeller2_wkt::Color;
        const COLORS: &[Color] = &[
            Color::TURQUOISE,
            Color::SLATE,
            Color::PUMPKIN,
            Color::YOLK,
            Color::PEACH,
            Color::REDDISH,
            Color::HYPERBLUE,
            Color::MINT,
            Color::TURQUOISE,
        ];
        let component_name = component_name.unwrap_or_else(|| "world_pos".to_string());
        let line_width = line_width.unwrap_or(10.0);
        let index = if let Some(index) = index {
            if index.len() != 3 {
                return Err(PyValueError::new_err("index must be 3"));
            }
            [index[0], index[1], index[2]]
        } else if component_name == "world_pos" {
            [4, 5, 6]
        } else {
            [0, 1, 2]
        };
        let component_id = ComponentId::new(&component_name);
        let color = color
            .map(|c| c.inner)
            .unwrap_or_else(|| COLORS[component_id.0 as usize % COLORS.len()]);

        Ok(Self {
            inner: impeller2_wkt::Line3d {
                entity: entity.inner,
                component_id,
                line_width,
                color,
                index,
                perspective: perspective.unwrap_or_default(),
            },
        })
    }

    pub fn asset_name(&self) -> &'static str {
        impeller2_wkt::Line3d::NAME
    }

    pub fn bytes(&self) -> Result<PyBufBytes, Error> {
        let bytes = postcard::to_allocvec(&self.inner).unwrap().into();
        Ok(PyBufBytes { bytes })
    }
}
