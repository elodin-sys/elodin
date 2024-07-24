use crate::*;

use std::marker::PhantomData;

use conduit::ComponentId;
use nox_ecs::{join_many, nox::Noxpr};
use nox_ecs::{join_query, update_var, ComponentArray};

#[pyclass]
#[derive(Clone)]
pub struct QueryInner {
    pub query: nox_ecs::Query<()>,
    pub metadata: Vec<Metadata>,
}

#[pymethods]
impl QueryInner {
    #[staticmethod]
    pub fn from_arrays(
        arrays: Vec<PyObject>,
        //component_names: Vec<String>,
        metadata: QueryMetadata,
    ) -> Result<QueryInner, Error> {
        Ok(QueryInner {
            query: nox_ecs::Query {
                exprs: arrays.into_iter().map(Noxpr::jax).collect(),
                entity_map: metadata.entity_map,
                len: metadata.len,
                phantom_data: PhantomData,
            },
            metadata: metadata.metadata,
        })
    }

    #[staticmethod]
    pub fn from_builder(
        builder: SystemBuilder,
        component_ids: Vec<String>,
        args: Vec<PyObject>,
    ) -> Result<QueryInner, Error> {
        let (query, metadata) = component_ids
            .iter()
            .map(|id| {
                let id = ComponentId::new(id);
                let (meta, i) = builder
                    .get_var(id)
                    .ok_or(nox_ecs::Error::ComponentNotFound)?;
                let buffer = args.get(i).ok_or(nox_ecs::Error::ComponentNotFound)?;
                Ok::<_, Error>((
                    ComponentArray {
                        buffer: Noxpr::jax(buffer.clone()),
                        len: meta.len,
                        entity_map: meta.entity_map,
                        component_id: meta.metadata.component_id(),
                        phantom_data: PhantomData,
                    },
                    meta.metadata,
                ))
            })
            .try_fold((None, vec![]), |(mut query, mut metadata), res| {
                let (a, meta) = res?;
                metadata.push(meta);
                if query.is_some() {
                    query = Some(join_many(query.take().unwrap(), &a));
                } else {
                    let q: nox_ecs::Query<()> = a.into();
                    query = Some(q);
                }
                Ok::<_, Error>((query, metadata))
            })?;
        let query = query.ok_or(nox_ecs::Error::ComponentNotFound)?;
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

    pub fn output(&self, builder: SystemBuilder, args: Vec<PyObject>) -> Result<PyObject, Error> {
        let mut outputs = vec![];
        for (expr, id) in self
            .query
            .exprs
            .iter()
            .zip(self.metadata.iter().map(|m| m.inner.component_id()))
        {
            let Some((meta, index)) = builder.get_var(id) else {
                return Err(nox_ecs::Error::ComponentNotFound.into());
            };
            let buffer = args.get(index).ok_or(nox_ecs::Error::ComponentNotFound)?;

            if meta.entity_map == self.query.entity_map {
                outputs.push(expr.clone());
            } else {
                let out = update_var(
                    &meta.entity_map,
                    &self.query.entity_map,
                    &Noxpr::jax(buffer.clone()),
                    expr,
                );
                outputs.push(out);
            }
        }
        if outputs.len() == 1 {
            outputs.pop().unwrap().to_jax().map_err(Error::from)
        } else {
            Noxpr::tuple(outputs).to_jax().map_err(Error::from)
        }
    }

    pub fn arrays(&self) -> Result<Vec<PyObject>, Error> {
        self.query
            .exprs
            .iter()
            .map(|e| e.to_jax().map_err(Error::from))
            .collect()
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

#[pyclass]
#[derive(Clone)]
pub struct QueryMetadata {
    pub entity_map: BTreeMap<conduit::EntityId, usize>,
    pub len: usize,
    pub metadata: Vec<Metadata>,
}

#[pymethods]
impl QueryMetadata {
    pub fn merge(&mut self, other: QueryMetadata) {
        self.metadata.extend(other.metadata);
    }
}
