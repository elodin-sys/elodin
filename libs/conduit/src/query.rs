use crate::{ComponentId, Metadata, MetadataPair, MetadataQuery, Query};
use smallvec::{smallvec, SmallVec};
use std::collections::HashMap;

#[cfg_attr(feature = "bevy", derive(bevy::prelude::Resource))]
#[derive(serde::Serialize, serde::Deserialize, Clone, Default, Debug, PartialEq)]
pub struct MetadataStore {
    pub metadata: Vec<Metadata>,
    pub component_index: HashMap<ComponentId, usize>,
    pub metadata_index: HashMap<MetadataPair, ComponentId>,
}

impl MetadataStore {
    pub fn push(&mut self, metadata: Metadata) {
        for (key, tag) in &metadata.tags {
            self.metadata_index.insert(
                MetadataPair(key.clone(), tag.clone()),
                metadata.component_id,
            );
        }
        self.component_index
            .insert(metadata.component_id, self.metadata.len());
        self.metadata.push(metadata);
    }

    pub fn get_metadata(&self, component_id: &ComponentId) -> Option<&Metadata> {
        self.component_index
            .get(component_id)
            .and_then(|&index| self.metadata.get(index))
    }
}

impl Query {
    pub fn execute(&self, store: &MetadataStore) -> SmallVec<[QueryId; 2]> {
        match self {
            Query::All => store
                .metadata
                .iter()
                .map(|m| QueryId::Component(m.component_id))
                .collect(),
            Query::Metadata(q) => q.execute(store),
            Query::ComponentId(id) => smallvec![QueryId::Component(*id)],
            Query::With(id) => smallvec![QueryId::With(*id)],
            Query::And(q) => q.iter().flat_map(|q| q.execute(store)).collect(),
        }
    }
}

impl MetadataQuery {
    pub fn execute(&self, store: &MetadataStore) -> SmallVec<[QueryId; 2]> {
        match self {
            MetadataQuery::And(q) => q.iter().flat_map(|q| q.execute(store)).collect(),
            MetadataQuery::Equals(pair) => {
                let id = store.metadata_index.get(pair);
                id.into_iter().copied().map(QueryId::Component).collect()
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryId {
    With(ComponentId),
    Component(ComponentId),
}
