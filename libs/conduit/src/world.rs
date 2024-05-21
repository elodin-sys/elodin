use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Duration;

use bytemuck::Pod;

use crate::*;

// 16.67 ms
pub const DEFAULT_TIME_STEP: Duration = Duration::from_nanos(1_000_000_000 / 120);

pub type Buffers<B = Vec<u8>> = BTreeMap<ComponentId, B>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TimeStep(pub Duration);

impl Default for TimeStep {
    fn default() -> Self {
        Self(DEFAULT_TIME_STEP)
    }
}

#[derive(Default, Debug, PartialEq, Eq)]
pub struct World {
    pub host: Buffers,
    pub history: Vec<Buffers>,
    pub entity_ids: ustr::UstrMap<Vec<u8>>,
    pub dirty_components: HashSet<ComponentId>,
    pub component_map: HashMap<ComponentId, (ArchetypeName, Metadata)>,
    pub assets: AssetStore,
    pub tick: u64,
    pub entity_len: u64,
    pub time_step: TimeStep,
}

pub struct ColumnRef<'a, B: 'a> {
    pub column: B,
    pub entities: B,
    pub metadata: &'a Metadata,
}

pub struct Entity<'a> {
    id: EntityId,
    world: &'a mut World,
}

impl Entity<'_> {
    #[cfg(feature = "nox")]
    pub fn metadata(self, metadata: crate::well_known::EntityMetadata) -> Self {
        let metadata = self.world.insert_asset(metadata);
        self.world.insert_with_id(metadata, self.id);
        self
    }

    pub fn insert(self, archetype: impl Archetype + 'static) -> Self {
        self.world.insert_with_id(archetype, self.id);
        self
    }

    pub fn id(&self) -> EntityId {
        self.id
    }
}

impl From<Entity<'_>> for EntityId {
    fn from(val: Entity<'_>) -> Self {
        val.id
    }
}

impl World {
    pub fn new(
        mut history: Vec<Buffers>,
        entity_ids: ustr::UstrMap<Vec<u8>>,
        component_map: HashMap<ComponentId, (ArchetypeName, Metadata)>,
        asset_store: AssetStore,
        time_step: TimeStep,
    ) -> Self {
        let host = history.pop().unwrap_or_default();
        let tick = history.len() as u64;
        let max_entity_id = entity_ids
            .values()
            .map(|ids| bytemuck::try_cast_slice::<_, u64>(ids.as_slice()).unwrap())
            .flat_map(|ids| ids.iter())
            .copied()
            .max();
        let entity_len = max_entity_id.map(|id| id + 1).unwrap_or(0);
        let dirty_components = host.keys().copied().collect();
        Self {
            host,
            history,
            entity_ids,
            dirty_components,
            component_map,
            assets: asset_store,
            tick,
            entity_len,
            time_step,
        }
    }

    pub fn spawn(&mut self, archetype: impl Archetype + 'static) -> Entity<'_> {
        let entity_id = EntityId(self.entity_len);
        self.insert_with_id(archetype, entity_id);
        self.entity_len += 1;
        Entity {
            id: entity_id,
            world: self,
        }
    }

    pub fn insert_with_id<A: Archetype + 'static>(&mut self, archetype: A, entity_id: EntityId) {
        let archetype_name = A::name();
        for metadata in A::components() {
            let id = metadata.component_id();
            self.component_map.insert(id, (archetype_name, metadata));
            self.host.entry(id).or_default();
            self.dirty_components.insert(id);
        }
        self.entity_ids
            .entry(archetype_name)
            .or_default()
            .extend_from_slice(&entity_id.0.to_le_bytes());
        archetype.insert_into_world(self);
    }

    pub fn column_mut<C: Component + 'static>(&mut self) -> Option<ColumnRef<'_, &mut Vec<u8>>> {
        self.column_by_id_mut(C::component_id())
    }

    pub fn column<C: Component + 'static>(&self) -> Option<ColumnRef<'_, &Vec<u8>>> {
        self.column_by_id(C::component_id())
    }

    pub fn column_by_id(&self, id: ComponentId) -> Option<ColumnRef<'_, &Vec<u8>>> {
        let (table_id, metadata) = self.component_map.get(&id)?;
        let column = self.host.get(&id)?;
        let entities = self.entity_ids.get(table_id)?;
        Some(ColumnRef {
            column,
            entities,
            metadata,
        })
    }

    pub fn column_by_id_mut(&mut self, id: ComponentId) -> Option<ColumnRef<'_, &mut Vec<u8>>> {
        let (table_id, metadata) = self.component_map.get(&id)?;
        let column = self.host.get_mut(&id)?;
        let entities = self.entity_ids.get_mut(table_id)?;
        self.dirty_components.insert(id);
        Some(ColumnRef {
            column,
            entities,
            metadata,
        })
    }

    pub fn column_at_tick(
        &self,
        component_id: ComponentId,
        tick: u64,
    ) -> Option<ColumnRef<'_, &Vec<u8>>> {
        if tick == self.tick {
            return self.column_by_id(component_id);
        }
        let column = self.history.get(tick as usize)?.get(&component_id)?;
        let (archetype_name, metadata) = self.component_map.get(&component_id)?;
        let entities = self.entity_ids.get(archetype_name)?;
        Some(ColumnRef {
            column,
            entities,
            metadata,
        })
    }

    pub fn entity_ids(&self) -> HashSet<EntityId> {
        self.entity_ids
            .values()
            .flat_map(|ids| {
                bytemuck::cast_slice::<_, u64>(ids)
                    .iter()
                    .copied()
                    .map(EntityId)
            })
            .collect()
    }

    pub fn insert_asset<C: Asset + Send + Sync + 'static>(&mut self, asset: C) -> Handle<C> {
        self.assets.insert(asset)
    }

    pub fn insert_shape(
        &mut self,
        mesh: well_known::Mesh,
        material: well_known::Material,
    ) -> well_known::Shape {
        let mesh = self.insert_asset(mesh);
        let material = self.insert_asset(material);
        well_known::Shape { mesh, material }
    }

    pub fn advance_tick(&mut self) {
        self.history.push(self.host.clone());
        self.tick += 1;
    }
}

impl Clone for World {
    fn clone(&self) -> Self {
        let dirty_components = self.host.keys().copied().collect();
        Self {
            host: self.host.clone(),
            history: self.history.clone(),
            entity_ids: self.entity_ids.clone(),
            dirty_components,
            component_map: self.component_map.clone(),
            assets: self.assets.clone(),
            tick: self.tick,
            entity_len: self.entity_len,
            time_step: self.time_step,
        }
    }
}

impl<'a, B: 'a + AsRef<[u8]>> ColumnRef<'a, B> {
    pub fn typed_buf<T: Component + Pod>(&self) -> Option<&[T]> {
        if self.metadata.component_type.primitive_ty != T::component_type().primitive_ty {
            return None;
        }
        bytemuck::try_cast_slice(self.column.as_ref()).ok()
    }

    pub fn entity_ids(&self) -> impl Iterator<Item = EntityId> + '_ {
        bytemuck::cast_slice::<_, u64>(self.entities.as_ref())
            .iter()
            .copied()
            .map(EntityId)
    }

    pub fn entity_map(&self) -> BTreeMap<EntityId, usize> {
        self.entity_ids()
            .enumerate()
            .map(|(i, id)| (id, i))
            .collect()
    }

    pub fn values_iter(&self) -> impl Iterator<Item = ComponentValue<'_>> + '_ {
        let mut buf_offset = 0;
        std::iter::from_fn(move || {
            let buf = self.column.as_ref().get(buf_offset..)?;
            let (offset, value) = self.metadata.component_type.parse_value(buf).ok()?;
            buf_offset += offset;
            Some(value)
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = (EntityId, ComponentValue<'_>)> {
        self.entity_ids().zip(self.values_iter())
    }

    pub fn typed_iter<T: Component + ValueRepr>(&self) -> impl Iterator<Item = (EntityId, T)> + '_ {
        assert_eq!(self.metadata.component_type, T::component_type());
        self.entity_ids().zip(
            self.values_iter()
                .filter_map(|v| T::from_component_value(v)),
        )
    }

    pub fn len(&self) -> usize {
        self.column.as_ref().len() / self.metadata.component_type.size()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a> ColumnRef<'a, &'a mut Vec<u8>> {
    pub fn typed_buf_mut<T: Component + Pod>(&mut self) -> Option<&mut [T]> {
        if self.metadata.component_type != T::component_type() {
            return None;
        }
        bytemuck::try_cast_slice_mut(self.column.as_mut_slice()).ok()
    }

    pub fn update(&mut self, offset: usize, value: ComponentValue<'_>) -> Result<(), Error> {
        let size = self.metadata.component_type.size();
        let offset = offset * size;
        let out = &mut self.column[offset..offset + size];
        if let Some(bytes) = value.bytes() {
            if bytes.len() != out.len() {
                return Err(Error::ValueSizeMismatch);
            }
            out.copy_from_slice(bytes);
        }
        Ok(())
    }

    pub fn push_raw(&mut self, raw: &[u8]) {
        self.column.extend_from_slice(raw);
    }
}

#[cfg(feature = "nox")]
mod nox_impl {
    use super::*;
    use ::nox::{xla, Client};

    impl<'a, B: 'a + AsRef<[u8]>> ColumnRef<'a, B> {
        pub fn copy_to_client(&self, client: &Client) -> Result<xla::PjRtBuffer, xla::Error> {
            let mut dims = self.metadata.component_type.shape.clone();
            dims.insert(0, self.len() as i64);
            client.copy_raw_host_buffer(
                self.metadata.component_type.primitive_ty.element_type(),
                self.column.as_ref(),
                &dims[..],
            )
        }
    }
}
