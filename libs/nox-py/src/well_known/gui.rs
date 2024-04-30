use crate::*;

use conduit::well_known::{Graph, Split};
use nox_ecs::conduit::Asset;
use nox_ecs::{conduit, nox::nalgebra::Vector3};
use numpy::PyArrayLike1;
use pyo3::exceptions::PyValueError;

use crate::EntityId;

#[pyclass]
#[derive(Clone)]
pub struct Panel {
    inner: conduit::well_known::Panel,
}

#[pymethods]
impl Panel {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn viewport(
        track_entity: Option<EntityId>,
        track_rotation: Option<bool>,
        fov: Option<f32>,
        active: Option<bool>,
        pos: Option<PyArrayLike1<f32>>,
        looking_at: Option<PyArrayLike1<f32>>,
        show_grid: Option<bool>,
        hdr: Option<bool>,
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
        let track_rotation = track_rotation.unwrap_or(true);
        let mut viewport = conduit::well_known::Viewport {
            track_entity: track_entity.map(|x| x.inner),
            fov: fov.unwrap_or(45.0),
            active: active.unwrap_or_default(),
            pos,
            track_rotation,
            show_grid: show_grid.unwrap_or_default(),
            hdr: hdr.unwrap_or_default(),
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
            inner: conduit::well_known::Panel::Viewport(viewport),
        })
    }

    pub fn asset_id(&self) -> u64 {
        self.inner.asset_id().0
    }

    pub fn bytes(&self) -> Result<PyBufBytes, Error> {
        let bytes = postcard::to_allocvec(&self.inner).unwrap().into();
        Ok(PyBufBytes { bytes })
    }

    #[getter]
    pub fn __metadata__(&self, py: Python<'_>) -> Result<((Component,),), Error> {
        let name = conduit::well_known::Panel::ASSET_ID.component_name();
        Ok(((Component {
            name,
            ty: ComponentType::u64(py),
            asset: true,
            metadata: Default::default(),
        },),))
    }

    #[staticmethod]
    pub fn vsplit(panels: Vec<Panel>, active: Option<bool>) -> Self {
        let active = active.unwrap_or_default();
        Self {
            inner: conduit::well_known::Panel::VSplit(Split {
                panels: panels.into_iter().map(|x| x.inner).collect(),
                active,
            }),
        }
    }

    #[staticmethod]
    pub fn hsplit(panels: Vec<Panel>, active: Option<bool>) -> Self {
        let active = active.unwrap_or_default();
        Self {
            inner: conduit::well_known::Panel::HSplit(Split {
                panels: panels.into_iter().map(|x| x.inner).collect(),
                active,
            }),
        }
    }

    #[staticmethod]
    pub fn graph(entities: Vec<GraphEntity>) -> PyResult<Self> {
        let entities = entities
            .into_iter()
            .map(|x| conduit::well_known::GraphEntity {
                entity_id: x.entity.inner,
                components: x
                    .components
                    .into_iter()
                    .map(|x| conduit::well_known::GraphComponent {
                        component_id: x.component_id,
                        indexes: x.indexes,
                    })
                    .collect(),
            })
            .collect();
        Ok(Self {
            inner: conduit::well_known::Panel::Graph(Graph { entities }),
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
    pub fn new(py: Python<'_>, entity: EntityId, objs: Vec<PyObject>) -> Self {
        let mut components = vec![];
        for obj in &objs {
            if let Ok(obj) = obj.extract::<GraphComponent>(py) {
                components.push(obj)
            } else if let Ok(indexer) = obj.extract::<ShapeIndexer>(py) {
                components.push(GraphComponent::from_shape_indexer(&indexer))
            } else if let Ok(indexer) = obj.extract::<Vec<ShapeIndexer>>(py) {
                components.extend(indexer.iter().map(GraphComponent::from_shape_indexer))
            }
        }
        Self { entity, components }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct GraphComponent {
    pub component_id: ComponentId,
    pub indexes: Vec<usize>,
}

#[pymethods]
impl GraphComponent {
    #[new]
    pub fn new(component_name: &str, indexes: Vec<usize>) -> Self {
        Self {
            component_id: ComponentId::new(component_name),
            indexes,
        }
    }
}

impl GraphComponent {
    fn from_shape_indexer(indexer: &ShapeIndexer) -> Self {
        Self {
            component_id: ComponentId::new(&indexer.component_name),
            indexes: indexer.indexes(),
        }
    }
}
