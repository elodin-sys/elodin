use crate::{ComponentId, Metadata, MetadataPair};
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
        let component_id = metadata.component_id();
        for (key, tag) in &metadata.tags {
            self.metadata_index
                .insert(MetadataPair(key.clone(), tag.clone()), component_id);
        }
        self.component_index
            .insert(component_id, self.metadata.len());
        self.metadata.push(metadata);
    }

    pub fn get_metadata(&self, component_id: &ComponentId) -> Option<&Metadata> {
        self.component_index
            .get(component_id)
            .and_then(|&index| self.metadata.get(index))
    }
}
