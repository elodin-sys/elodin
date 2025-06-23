use crate::*;

use impeller2::component::Asset;
use impeller2_wkt::{Graph, GraphType, Split};
use pyo3::exceptions::PyValueError;

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
    #[pyo3(signature = (fov = 45.0, active = false, show_grid = false, hdr = false, name = None, pos = None, look_at = None))]
    pub fn viewport(
        fov: f32,
        active: bool,
        show_grid: bool,
        hdr: bool,
        name: Option<String>,
        pos: Option<String>,
        look_at: Option<String>,
    ) -> PyResult<Self> {
        let viewport = impeller2_wkt::Viewport {
            fov,
            active,
            show_grid,
            hdr,
            name,
            pos,
            look_at,
            aux: (),
        };
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
    #[pyo3(signature = (eql, name = None, ty = None))]
    pub fn graph(eql: String, name: Option<String>, ty: Option<String>) -> PyResult<Self> {
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
        Ok(Self {
            inner: impeller2_wkt::Panel::Graph(Graph {
                eql,
                name,
                graph_type,
                auto_y_range: true,
                y_range: 0.0..1.0,
                aux: (),
            }),
        })
    }
}

#[pyclass]
pub struct Line3d {
    inner: impeller2_wkt::Line3d,
}

#[pymethods]
impl Line3d {
    #[new]
    #[pyo3(signature = (eql, line_width=None, color=None, perspective=None))]
    pub fn new(
        eql: String,
        line_width: Option<f32>,
        color: Option<Color>,
        perspective: Option<bool>,
    ) -> PyResult<Self> {
        use impeller2_wkt::Color;
        let line_width = line_width.unwrap_or(10.0);
        let color = color.map(|c| c.inner).unwrap_or_else(|| Color::HYPERBLUE);

        Ok(Self {
            inner: impeller2_wkt::Line3d {
                eql,
                line_width,
                color,
                perspective: perspective.unwrap_or_default(),
                aux: (),
            },
        })
    }

    pub fn asset_name(&self) -> &'static str {
        impeller2_wkt::Line3d::<()>::NAME
    }

    pub fn bytes(&self) -> Result<PyBufBytes, Error> {
        let bytes = postcard::to_allocvec(&self.inner).unwrap().into();
        Ok(PyBufBytes { bytes })
    }
}
