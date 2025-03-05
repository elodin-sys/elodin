use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Duration;

use crate::utils::SchemaExt;
use assets::Handle;
use bytemuck::Pod;
use elodin_db::{ComponentSchema, MetadataExt};
use impeller2::com_de::FromComponentView;
use impeller2::{
    component::{Asset, Component},
    types::{ComponentView, EntityId},
};
use impeller2_wkt::{ComponentMetadata, Material, Mesh};
//use impeller::{well_known, Asset, AssetStore, Component, ComponentValue, ComponentMetadata, ValueRepr};

use crate::*;

// 16.67 ms
pub const DEFAULT_TIME_STEP: Duration = Duration::from_nanos(1_000_000_000 / 120);

pub type Buffers<B = Vec<u8>> = BTreeMap<ComponentId, Column<B>>;

#[derive(Debug, PartialEq, Eq, Default, Clone, Deserialize, Serialize)]
pub struct Column<B> {
    pub buffer: B,
    pub entity_ids: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TimeStep(pub Duration);

impl Default for TimeStep {
    fn default() -> Self {
        Self(DEFAULT_TIME_STEP)
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct World {
    pub host: Buffers,
    pub assets: AssetStore,
    pub dirty_components: HashSet<ComponentId>,
    pub metadata: WorldMetadata,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct WorldMetadata {
    pub entity_metadata: HashMap<EntityId, EntityMetadata>,
    pub component_map: HashMap<ComponentId, (ComponentSchema, ComponentMetadata)>,
    pub tick: u64,
    pub entity_len: u64,
    pub sim_time_step: TimeStep,
    pub run_time_step: TimeStep,
    pub default_playback_speed: f64,
    pub max_tick: u64,
}

impl MetadataExt for World {}

impl Default for WorldMetadata {
    fn default() -> Self {
        Self {
            component_map: Default::default(),
            entity_metadata: Default::default(),
            tick: Default::default(),
            entity_len: Default::default(),
            run_time_step: Default::default(),
            sim_time_step: Default::default(),
            default_playback_speed: 1.0,
            max_tick: u64::MAX,
        }
    }
}

impl Default for World {
    fn default() -> Self {
        let mut world = Self {
            host: Default::default(),
            dirty_components: Default::default(),
            assets: Default::default(),
            metadata: Default::default(),
        };

        world.add_globals();
        world
    }
}

pub struct ColumnRef<'a, B: 'a> {
    pub column: B,
    pub entities: B,
    pub schema: &'a ComponentSchema,
    pub metadata: &'a ComponentMetadata,
}

pub struct Entity<'a> {
    id: EntityId,
    world: &'a mut World,
}

impl Entity<'_> {
    pub fn metadata(self, metadata: impeller2_wkt::EntityMetadata) -> Self {
        self.world
            .metadata
            .entity_metadata
            .insert(self.id, metadata);
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
    pub fn component_map(&self) -> &HashMap<ComponentId, (ComponentSchema, ComponentMetadata)> {
        &self.metadata.component_map
    }

    pub fn tick(&self) -> u64 {
        self.metadata.tick
    }

    pub fn entity_len(&self) -> u64 {
        self.metadata.entity_len
    }

    pub fn entity_metadata(&self) -> &HashMap<EntityId, EntityMetadata> {
        &self.metadata.entity_metadata
    }

    pub fn sim_time_step(&self) -> TimeStep {
        self.metadata.sim_time_step
    }

    pub fn run_time_step(&self) -> TimeStep {
        self.metadata.run_time_step
    }

    pub fn max_tick(&self) -> u64 {
        self.metadata.max_tick
    }

    pub fn default_playback_speed(&self) -> f64 {
        self.metadata.default_playback_speed
    }

    fn add_globals(&mut self) {
        self.spawn(SystemGlobals::new(
            self.metadata.sim_time_step.0.as_secs_f64(),
        ))
        .metadata(EntityMetadata {
            entity_id: EntityId(0),
            name: "Globals".to_string(),
            metadata: Default::default(), //color: Color::WHITE,
        });
    }

    pub fn set_globals(&mut self) {
        let bytes = self.metadata.sim_time_step.0.as_secs_f64().to_le_bytes();
        let col = self
            .column_mut::<SimulationTimeStep>()
            .expect("no sim time step");
        col.column[..8].copy_from_slice(&bytes);
    }

    pub fn spawn(&mut self, archetype: impl Archetype + 'static) -> Entity<'_> {
        let entity_id = EntityId(self.metadata.entity_len);
        self.spawn_with_id(archetype, entity_id)
    }

    pub fn spawn_with_id(
        &mut self,
        archetype: impl Archetype + 'static,
        entity_id: EntityId,
    ) -> Entity<'_> {
        self.insert_with_id(archetype, entity_id);
        self.metadata.entity_len += 1;
        Entity {
            id: entity_id,
            world: self,
        }
    }

    pub fn insert_with_id<A: Archetype + 'static>(&mut self, archetype: A, entity_id: EntityId) {
        for (schema, metadata) in A::components() {
            let id = metadata.component_id;
            let buffer = self.host.entry(id).or_default();
            buffer
                .entity_ids
                .extend_from_slice(&entity_id.0.to_le_bytes());
            self.metadata
                .component_map
                .insert(id, (schema.into(), metadata));
            self.dirty_components.insert(id);
        }
        archetype.insert_into_world(self);
    }

    pub fn column_mut<C: Component + 'static>(&mut self) -> Option<ColumnRef<'_, &mut Vec<u8>>> {
        self.column_by_id_mut(C::COMPONENT_ID)
    }

    pub fn column<C: Component + 'static>(&self) -> Option<ColumnRef<'_, &Vec<u8>>> {
        self.column_by_id(C::COMPONENT_ID)
    }

    pub fn column_by_id(&self, id: ComponentId) -> Option<ColumnRef<'_, &Vec<u8>>> {
        let (schema, metadata) = self.metadata.component_map.get(&id).unwrap();
        let column = self.host.get(&id).unwrap();
        Some(ColumnRef {
            column: &column.buffer,
            entities: &column.entity_ids,
            metadata,
            schema,
        })
    }

    pub fn column_by_id_mut(&mut self, id: ComponentId) -> Option<ColumnRef<'_, &mut Vec<u8>>> {
        let (schema, metadata) = self.metadata.component_map.get(&id)?;
        let column = self.host.get_mut(&id)?;
        self.dirty_components.insert(id);
        Some(ColumnRef {
            column: &mut column.buffer,
            entities: &mut column.entity_ids,
            metadata,
            schema,
        })
    }

    pub fn write_to_dir(&mut self, world_dir: &Path) -> Result<(), Error> {
        self.write(world_dir.join("world"))?;
        Ok(())
    }

    pub fn entity_ids(&self) -> HashSet<EntityId> {
        self.host
            .values()
            .flat_map(|c| {
                bytemuck::cast_slice::<_, u64>(&c.entity_ids)
                    .iter()
                    .copied()
                    .map(EntityId)
            })
            .collect()
    }

    pub fn insert_asset<C: Asset + Send + Sync + 'static>(&mut self, asset: C) -> Handle<C> {
        self.assets.insert(asset)
    }

    pub fn insert_shape(&mut self, mesh: Mesh, material: Material) -> Shape {
        let mesh = self.insert_asset(mesh);
        let material = self.insert_asset(material);
        Shape { mesh, material }
    }

    pub fn advance_tick(&mut self) {
        self.metadata.tick += 1;
    }
}

impl Clone for World {
    fn clone(&self) -> Self {
        let dirty_components = self.host.keys().copied().collect();
        Self {
            host: self.host.clone(),
            dirty_components,
            metadata: WorldMetadata {
                component_map: self.metadata.component_map.clone(),
                entity_metadata: self.metadata.entity_metadata.clone(),
                tick: self.metadata.tick,
                entity_len: self.metadata.entity_len,
                run_time_step: self.metadata.run_time_step,
                sim_time_step: self.metadata.sim_time_step,
                default_playback_speed: self.metadata.default_playback_speed,
                max_tick: self.metadata.max_tick,
            },
            assets: self.assets.clone(),
        }
    }
}

impl<'a, B: 'a + AsRef<[u8]>> ColumnRef<'a, B> {
    pub fn typed_buf<T: Component + Pod>(&self) -> Option<&[T]> {
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

    pub fn values_iter(&self) -> impl Iterator<Item = ComponentView<'_>> + '_ {
        let mut buf_offset = 0;
        std::iter::from_fn(move || {
            let buf = self.column.as_ref().get(buf_offset..)?;
            let (offset, value) = self.schema.parse_value(buf).ok()?;
            buf_offset += offset;
            Some(value)
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = (EntityId, ComponentView<'_>)> {
        self.entity_ids().zip(self.values_iter())
    }

    pub fn typed_iter<T: Component + FromComponentView>(
        &self,
    ) -> impl Iterator<Item = (EntityId, T)> + '_ {
        self.entity_ids().zip(
            self.values_iter()
                .filter_map(|val| T::from_component_view(val).ok()),
        )
    }

    pub fn len(&self) -> usize {
        self.column.as_ref().len() / self.schema.size()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn buffer_ty(&self) -> ::nox::ArrayTy {
        let mut array_ty = self.schema.to_array_ty();
        array_ty.shape.insert(0, self.len() as i64);
        array_ty
    }
}

impl<'a> ColumnRef<'a, &'a mut Vec<u8>> {
    pub fn typed_buf_mut<T: Component + Pod>(&mut self) -> Option<&mut [T]> {
        if self.metadata.component_id != T::COMPONENT_ID {
            return None;
        }
        bytemuck::try_cast_slice_mut(self.column.as_mut_slice()).ok()
    }

    pub fn update(&mut self, offset: usize, value: ComponentView<'_>) -> Result<(), Error> {
        let size = self.schema.size();
        let offset = offset * size;
        let out = &mut self.column[offset..offset + size];
        let bytes = value.as_bytes();
        if bytes.len() != out.len() {
            return Err(Error::ValueSizeMismatch);
        }
        out.copy_from_slice(bytes);
        Ok(())
    }

    pub fn push_raw(&mut self, raw: &[u8]) {
        self.column.extend_from_slice(raw);
    }
}

use nox::{Client, xla};

impl<'a, B: 'a + AsRef<[u8]>> ColumnRef<'a, B> {
    pub fn copy_to_client(&self, client: &Client) -> Result<xla::PjRtBuffer, xla::Error> {
        let mut dims: SmallVec<[i64; 4]> = self.schema.dim.iter().map(|&x| x as i64).collect();
        dims.insert(0, self.len() as i64);
        client.copy_raw_host_buffer(self.schema.element_type(), self.column.as_ref(), &dims[..])
    }
}
