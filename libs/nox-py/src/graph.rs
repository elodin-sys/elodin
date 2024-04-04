use super::*;

use nox_ecs::{graph::GraphQuery, nox::IntoOp};
use pyo3::types::{PyDict, PyList, PyTuple};

#[pyclass]
#[derive(Clone)]
pub struct GraphQueryInner {
    query: nox_ecs::graph::GraphQuery<(), ()>,
}

#[pymethods]
impl GraphQueryInner {
    #[staticmethod]
    fn from_builder(
        builder: &mut PipelineBuilder,
        edge_id: ComponentId,
        component_ids: Vec<ComponentId>,
    ) -> Result<GraphQueryInner, Error> {
        use bytes::Buf;
        use nox_ecs::graph::EdgeComponent;
        let col = builder.builder.world.column_by_id(edge_id.inner).unwrap();
        let ty = &col.column.metadata.component_type;
        let buf = &mut &col.column.buf[..];
        let len = col.column.len;
        let edges = (0..len)
            .map(move |_| {
                let (size, value) = ty.parse_value(buf).unwrap();
                buf.advance(size);
                nox_ecs::graph::Edge::from_value(value).unwrap()
            })
            .collect();
        let query = QueryInner::from_builder(builder, component_ids)?;
        let g_query = query.query;
        let query = GraphQuery::from_queries(edges, g_query)?;
        Ok(GraphQueryInner { query })
    }

    fn arrays(&self, py: Python<'_>) -> Result<PyObject, Error> {
        let dict = PyDict::new(py);
        for (len, (a, b)) in self.query.exprs.iter() {
            let a_list = PyList::empty(py);
            let b_list = PyList::empty(py);
            for x in &a.exprs {
                a_list.append(x.to_jax()?)?;
            }
            for x in &b.exprs {
                b_list.append(x.to_jax()?)?;
            }

            dict.set_item(len, (a_list, b_list))?;
        }
        Ok(dict.into())
    }

    fn map(&self, new_buf: PyObject, metadata: Metadata) -> QueryInner {
        let mut entity_map = BTreeMap::new();
        let mut len = 0;
        for (_, (from, _to)) in self.query.exprs.iter() {
            for (id, index) in from.entity_map.iter() {
                entity_map.insert(*id, index + len);
            }
            len += from.len;
        }
        let expr = Noxpr::jax(new_buf);
        QueryInner {
            query: nox_ecs::Query {
                exprs: vec![expr],
                entity_map,
                len,
                phantom_data: PhantomData,
            },
            metadata: vec![metadata],
        }
    }
}

#[pyclass]
pub struct Edge {
    inner: nox_ecs::graph::Edge,
}

#[pymethods]
impl Edge {
    #[new]
    pub fn new(from: EntityId, to: EntityId) -> Self {
        Self {
            inner: nox_ecs::graph::Edge {
                from: from.inner,
                to: to.inner,
            },
        }
    }

    fn flatten(&self) -> Result<((PyObject,), Option<()>), Error> {
        let jax = self.inner.clone().into_op().to_jax()?;
        Ok(((jax,), None))
    }

    #[staticmethod]
    fn unflatten(_aux: PyObject, _jax: PyObject) -> Self {
        todo!()
    }

    // #[staticmethod]
    // fn from_array(jax: PyObject) -> Self {
    //     todo!()
    // }

    #[getter]
    fn shape(&self) -> PyObject {
        Python::with_gil(|py| PyTuple::new(py, [2]).into())
    }

    #[staticmethod]
    fn from_array(_arr: PyObject) -> Self {
        todo!()
    }
}
