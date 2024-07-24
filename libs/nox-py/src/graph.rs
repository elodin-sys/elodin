use super::*;

use conduit::{ComponentExt, ComponentId};
use nox_ecs::{
    graph::{exprs_from_edges_queries, GraphQuery},
    nox::IntoOp,
};
use pyo3::types::{PyDict, PyList};

#[pyclass]
#[derive(Clone)]
pub struct GraphQueryInner {
    query: nox_ecs::graph::GraphQuery<()>,
}

#[pymethods]
impl GraphQueryInner {
    #[staticmethod]
    fn from_builder_total_edge(builder: &mut SystemBuilder) -> Result<GraphQueryInner, Error> {
        Ok(GraphQueryInner {
            query: GraphQuery {
                edges: builder.total_edges.clone(),
                phantom_data: PhantomData,
            },
        })
    }

    #[staticmethod]
    fn from_builder(
        builder: &mut SystemBuilder,
        edge_name: String,
        reverse: bool,
    ) -> Result<GraphQueryInner, Error> {
        let edge_id = ComponentId::new(&edge_name);
        let edges = builder
            .edge_map
            .get(&edge_id)
            .ok_or(nox_ecs::Error::ComponentNotFound)?;
        let edges = edges
            .iter()
            .map(|x| if reverse { x.reverse() } else { x.clone() })
            .collect();
        Ok(GraphQueryInner {
            query: GraphQuery {
                edges,
                phantom_data: PhantomData,
            },
        })
    }

    fn arrays(
        &self,
        py: Python<'_>,
        from_query: QueryInner,
        to_query: QueryInner,
    ) -> Result<PyObject, Error> {
        let dict = PyDict::new_bound(py);
        let exprs = exprs_from_edges_queries(&self.query.edges, from_query.query, to_query.query);
        for (len, (a, b)) in exprs.iter() {
            let a_list = PyList::empty_bound(py);
            let b_list = PyList::empty_bound(py);
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

    fn map(
        &self,
        from_query: QueryInner,
        to_query: QueryInner,
        new_buf: PyObject,
        metadata: Metadata,
    ) -> QueryInner {
        let mut entity_map = BTreeMap::new();
        let mut len = 0;
        let exprs = exprs_from_edges_queries(&self.query.edges, from_query.query, to_query.query);
        for (_, (from, _to)) in exprs.iter() {
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

    #[classattr]
    fn metadata() -> Metadata {
        Metadata {
            inner: nox_ecs::graph::Edge::metadata(),
        }
    }

    #[classattr]
    fn __metadata__() -> (Component,) {
        (Self::metadata().into(),)
    }

    #[staticmethod]
    fn from_array(_arr: PyObject) -> Self {
        todo!()
    }
}
