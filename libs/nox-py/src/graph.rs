// Python bindings for graph functionality
use super::*;

use impeller2::types::ComponentId;
use crate::ecs::graph as graph_mod;
use nox::Noxpr;
use std::marker::PhantomData;
use std::collections::HashMap;

#[pyclass]
#[derive(Clone)]
pub struct GraphQueryInner {
    query: graph_mod::GraphQuery<()>,
}

#[pymethods]
impl GraphQueryInner {
    #[staticmethod]
    fn from_builder_total_edge(builder: &mut SystemBuilder) -> Result<GraphQueryInner, Error> {
        Ok(GraphQueryInner {
            query: graph_mod::GraphQuery {
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
            .ok_or(crate::Error::ComponentNotFound)?;
        let edges = edges
            .iter()
            .map(|x| if reverse { x.reverse() } else { x.clone() })
            .collect();
        Ok(GraphQueryInner {
            query: graph_mod::GraphQuery {
                edges,
                phantom_data: PhantomData,
            },
        })
    }

    fn arrays(
        &self,
        _py: Python<'_>,
        from_query: &QueryInner,
        to_query: &QueryInner,
    ) -> Result<HashMap<usize, (Vec<PyObject>, Vec<PyObject>)>, Error> {
        let from = from_query.query.clone();
        let to = to_query.query.clone();
        let exprs = graph_mod::exprs_from_edges_queries(&self.query.edges, from, to);
        
        let mut result = HashMap::new();
        for (key, (from_q, to_q)) in exprs.iter() {
            let from_arrays = from_q.exprs.iter()
                .map(|expr| expr.to_jax())
                .collect::<Result<Vec<_>, _>>()?;
            let to_arrays = to_q.exprs.iter()
                .map(|expr| expr.to_jax())
                .collect::<Result<Vec<_>, _>>()?;
            result.insert(*key, (from_arrays, to_arrays));
        }
        Ok(result)
    }
    
    fn map(
        &self,
        from_query: &QueryInner,
        to_query: &QueryInner,
        new_buf: PyObject,
        _component: Component,
    ) -> Result<QueryInner, Error> {
        let from = from_query.query.clone();
        let to = to_query.query.clone();
        let exprs = graph_mod::exprs_from_edges_queries(&self.query.edges, from, to);
        let mut entity_map = BTreeMap::new();
        let mut len = 0;
        for (_, (from, _)) in exprs.iter() {
            for (id, index) in from.entity_map.iter() {
                entity_map.insert(*id, len + index);
            }
            len += from.len;
        }
        let expr = Noxpr::jax(new_buf);
        Ok(QueryInner {
            query: crate::ecs::Query {
                exprs: vec![expr],
                entity_map,
                len,
                phantom_data: PhantomData,
            },
            metadata: from_query.metadata.clone(),
        })
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Edge {
    inner: graph_mod::Edge,
}

#[pymethods]
impl Edge {
    #[new]
    pub fn new(from: EntityId, to: EntityId) -> Self {
        Self {
            inner: graph_mod::Edge {
                from: from.inner,
                to: to.inner,
            },
        }
    }

    fn from_(&self) -> EntityId {
        EntityId {
            inner: self.inner.from,
        }
    }

    fn to(&self) -> EntityId {
        EntityId {
            inner: self.inner.to,
        }
    }

    fn flatten(&self) -> ((u64, u64), Option<()>) {
        ((self.inner.from.0, self.inner.to.0), None)
    }

    #[staticmethod]
    fn unflatten(_aux: PyObject, data: (u64, u64)) -> Self {
        Self {
            inner: graph_mod::Edge {
                from: impeller2::types::EntityId(data.0),
                to: impeller2::types::EntityId(data.1),
            }
        }
    }

    #[classattr]
    fn metadata() -> Component {
        Component::from_component::<graph_mod::Edge>()
    }

    #[classattr]
    fn __metadata__() -> (Component,) {
        (Self::metadata(),)
    }
}

