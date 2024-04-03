use crate::*;

use std::{marker::PhantomData, sync::Arc};

use nox_ecs::join_query;
use nox_ecs::{join_many, nox::Noxpr};

#[pyclass]
#[derive(Clone)]
pub struct QueryInner {
    pub query: nox_ecs::Query<()>,
    pub metadata: Vec<Metadata>,
}

#[pymethods]
impl QueryInner {
    #[staticmethod]
    pub fn from_builder(
        builder: &mut PipelineBuilder,
        component_ids: Vec<ComponentId>,
    ) -> Result<QueryInner, Error> {
        let metadata = component_ids
            .iter()
            .map(|id| {
                builder
                    .builder
                    .world
                    .column_by_id(id.inner)
                    .map(|c| Metadata {
                        inner: Arc::new(c.column.metadata.clone()),
                    })
            })
            .collect::<Option<Vec<_>>>()
            .ok_or(Error::NoxEcs(nox_ecs::Error::ComponentNotFound))?;
        let query = component_ids
            .iter()
            .copied()
            .map(|id| {
                builder
                    .builder
                    .vars
                    .get(&id.inner)
                    .ok_or(nox_ecs::Error::ComponentNotFound)
            })
            .try_fold(None, |mut query, a| {
                let a = a?;
                if query.is_some() {
                    query = Some(join_many(query.take().unwrap(), &*a.borrow()));
                } else {
                    let a = a.borrow().clone();
                    let q: nox_ecs::Query<()> = a.into();
                    query = Some(q);
                }
                Ok::<_, Error>(query)
            })?
            .expect("query must not be empty");
        Ok(Self { query, metadata })
    }

    pub fn map(&self, new_buf: PyObject, metadata: Metadata) -> QueryInner {
        let expr = Noxpr::jax(new_buf);
        QueryInner {
            query: nox_ecs::Query {
                exprs: vec![expr],
                entity_map: self.query.entity_map.clone(),
                len: self.query.len,
                phantom_data: PhantomData,
            },
            metadata: vec![metadata],
        }
    }

    pub fn arrays(&self) -> Result<Vec<PyObject>, Error> {
        self.query
            .exprs
            .iter()
            .map(|e| e.to_jax().map_err(Error::from))
            .collect()
    }

    pub fn insert_into_builder(&self, builder: &mut PipelineBuilder) {
        self.query.insert_into_builder_erased(
            &mut builder.builder,
            self.metadata.iter().map(|m| m.inner.component_id),
        );
    }

    pub fn join_query(&self, other: &QueryInner) -> QueryInner {
        let query = join_query(self.query.clone(), other.query.clone());
        let metadata = self
            .metadata
            .iter()
            .cloned()
            .chain(other.metadata.iter().cloned())
            .collect();
        QueryInner { query, metadata }
    }
}
