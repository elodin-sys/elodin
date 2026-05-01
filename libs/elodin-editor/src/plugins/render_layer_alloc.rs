//! Handle render layer allocation and deallocation. One allocates a
//! `RenderLayerLease` component from the `RenderLayerAllocator` resource. Add
//! it and its `RenderLayers` to whatever entity needs it. When all the
//! `RenderLayerLease`s are dropped, then the allocated render layer is
//! freed for use again.
//!
//! This can work with any render layer, and it uses the bits to find the next
//! available layer, so its fast and uses the lowest available number first.
//!
//! This module was created to avoid a maximum limit of 64 that were not
//! deallocated.
use crate::object_3d::ELLIPSOID_RENDER_LAYER;
use crate::plugins::gizmos::GIZMO_RENDER_LAYER;
use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use crossbeam_queue::SegQueue;
use std::sync::Arc;

/// Render layer shared by every viewport's infinite grid. Reserved so it is
/// never handed out by [`RenderLayerAllocator::alloc`].
pub const GRID_RENDER_LAYER: usize = 31;

pub(crate) fn plugin(app: &mut App) {
    app.register_type::<RenderLayerAllocator>()
        .init_resource::<RenderLayerAllocator>()
        .add_systems(First, process_dropped_render_layer_leases);
}

/// When the last [`Arc`] to this value is dropped, the layer index is pushed to
/// the shared queue; [`process_dropped_render_layer_leases`] then frees that
/// layer from [`RenderLayerAllocator`].
pub struct RenderLayerLeaseInner {
    pub layer: usize,
    dropped: Arc<SegQueue<usize>>,
}

/// This is where the magic happens.
impl Drop for RenderLayerLeaseInner {
    fn drop(&mut self) {
        self.dropped.push(self.layer);
    }
}

impl std::fmt::Debug for RenderLayerLeaseInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderLayerLeaseInner")
            .field("layer", &self.layer)
            .finish_non_exhaustive()
    }
}

fn render_layer_mask(layer: usize) -> RenderLayers {
    RenderLayers::none().with(layer)
}

/// Tracks render layers in use. The [`SegQueue`] is shared with every [`RenderLayerLeaseInner`] via
/// [`Arc`]; cloning a lease’s [`Arc`] delays freeing until all clones drop.
#[derive(Resource, Reflect)]
#[reflect(Resource)]
pub struct RenderLayerAllocator {
    in_use: RenderLayers,
    pub reserved: RenderLayers,
    #[reflect(ignore)]
    dropped: Arc<SegQueue<usize>>,
}

impl std::fmt::Debug for RenderLayerAllocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderLayerAllocator")
            .field("in_use", &self.in_use)
            .finish_non_exhaustive()
    }
}

impl RenderLayerAllocator {
    /// Return the first free layer index.
    ///
    /// We probe one virtual word past `bits.len()`: that word is conceptually
    /// all zeros (no layer used yet) and lets us return a layer beyond the
    /// current bitset size — `RenderLayers` is a growable `SmallVec<u64>`, so
    /// `union` will extend the storage on the next allocation. Without that
    /// extra iteration the allocator would re-introduce the historical 64-layer
    /// cap as soon as the first word filled up.
    fn first_free_layer_index(&self, in_use: &RenderLayers) -> Option<usize> {
        let bits = in_use.bits();
        for word_index in 0..bits.len() + 1 {
            let used = bits.get(word_index).copied().unwrap_or(0);
            let free = !used;
            if free != 0 {
                let layer = word_index * 64 + free.trailing_zeros() as usize;
                return Some(layer);
            }
        }
        None
    }

    /// Current union of in-use layers.
    pub fn in_use(&self) -> RenderLayers {
        self.in_use.clone()
    }

    /// Allocates the lowest free render layer excluding [`self.reserved`].
    pub fn alloc(&mut self) -> Option<RenderLayerLease> {
        let n = self.first_free_layer_index(&self.in_use)?;
        self.in_use = self.in_use.union(&render_layer_mask(n));
        Some(RenderLayerLease(Arc::new(RenderLayerLeaseInner {
            layer: n,
            dropped: Arc::clone(&self.dropped),
        })))
    }

    /// Free the given layer. Returns true if the layer was freed. Return false
    /// if a reserved layer is given or the layer was not allocated.
    fn free(&mut self, layer: usize) -> bool {
        let mask = render_layer_mask(layer);
        if self.reserved.intersects(&mask) {
            return false;
        }
        if self.in_use.intersects(&mask) {
            self.in_use = self.in_use.symmetric_difference(&mask);
            true
        } else {
            false
        }
    }

    /// Drains the drop queue and updates `in_use`. Used by
    /// [`process_dropped_render_layer_leases`] and by tests.
    pub fn drain_dropped(&mut self) {
        while let Some(layer) = self.dropped.pop() {
            if !self.free(layer) {
                warn!("Could not free render layer {layer}.");
            }
        }
    }
}

impl Default for RenderLayerAllocator {
    fn default() -> Self {
        let reserved = RenderLayers::layer(0)
            .with(ELLIPSOID_RENDER_LAYER)
            .with(GIZMO_RENDER_LAYER)
            .with(GRID_RENDER_LAYER);
        Self {
            in_use: reserved.clone(),
            reserved,
            dropped: Arc::new(SegQueue::new()),
        }
    }
}

fn process_dropped_render_layer_leases(mut alloc: ResMut<RenderLayerAllocator>) {
    alloc.drain_dropped();
}

/// Bevy component: holds [`Arc<RenderLayerLeaseInner>`]. Cloning the component clones the [`Arc`], so
/// the layer stays reserved until every clone is dropped (then one queue push, one free).
#[derive(Component, Clone, Debug)]
pub struct RenderLayerLease(pub Arc<RenderLayerLeaseInner>);

impl RenderLayerLease {
    /// Return the layer number.
    pub fn layer(&self) -> usize {
        self.0.layer
    }

    /// Convert this to render layers.
    pub fn render_layers(&self) -> RenderLayers {
        render_layer_mask(self.layer())
    }
}

/// Insertion helper for [`RenderLayerLease`].
///
/// `RenderLayerLease` is a regular Bevy component, so a second `insert` on the
/// same entity silently overwrites the first. The dropped `Arc` then frees the
/// layer back to the [`RenderLayerAllocator`], which reuses it for an unrelated
/// viewport. Two viewports end up sharing one render layer — visually visible
/// as cross-rendered frustums or gizmos.
///
/// Always go through [`EntityCommandsExt::insert_render_layer_lease`] when
/// touching an entity that already exists. Test builds panic on duplicate
/// insertion so CI catches the misuse; non-test builds keep the historical
/// permissive behavior.
pub trait EntityCommandsExt {
    /// Insert a [`RenderLayerLease`] together with its [`RenderLayers`] mask.
    /// Test builds panic if the entity already holds a `RenderLayerLease`.
    fn insert_render_layer_lease(&mut self, lease: RenderLayerLease) -> &mut Self;
}

impl EntityCommandsExt for EntityCommands<'_> {
    fn insert_render_layer_lease(&mut self, lease: RenderLayerLease) -> &mut Self {
        let layers = lease.render_layers();
        self.queue(move |mut entity: EntityWorldMut| {
            #[cfg(test)]
            assert!(
                entity.get::<RenderLayerLease>().is_none(),
                "RenderLayerLease already present on entity {:?}: a second insert \
                 would silently drop the previous lease and free its render layer \
                 while other entities may still rely on it (two viewports sharing \
                 one layer). Use a child entity, or remove the existing lease \
                 explicitly first.",
                entity.id()
            );
            entity.insert((lease, layers));
        });
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_layer_alloc() {
        let mut default = RenderLayerAllocator::default();
        let a = default.alloc().expect("layer 1");
        let b = default.alloc().expect("layer 2");
        assert_eq!(a.layer(), 1);
        assert_eq!(b.layer(), 2);
    }

    #[test]
    fn test_lease_drop_frees_after_drain() {
        let mut alloc = RenderLayerAllocator::default();
        let lease = alloc.alloc().expect("layer 1");
        assert_eq!(lease.layer(), 1);
        drop(lease);
        alloc.drain_dropped();
        let again = alloc.alloc().expect("layer 1 again");
        assert_eq!(again.layer(), 1);
    }

    #[test]
    fn test_reserved_layers_are_never_allocated() {
        let mut alloc = RenderLayerAllocator::default();
        let reserved = alloc.reserved.clone();
        for _ in 0..64 {
            let lease = alloc.alloc().expect("plenty of layers available");
            assert!(
                !reserved.intersects(&lease.render_layers()),
                "alloc returned a reserved layer: {}",
                lease.layer(),
            );
        }
    }

    #[test]
    fn test_alloc_grows_past_64_layers() {
        // Locks in the intent of the `+ 1` in `first_free_layer_index`: the
        // allocator must hand out layers beyond the first 64-bit word once the
        // word fills up, instead of returning `None` like the legacy
        // `RenderLayerAlloc`.
        let mut alloc = RenderLayerAllocator::default();
        let mut leases = Vec::new();
        let mut max_layer = 0;
        for _ in 0..200 {
            let lease = alloc.alloc().expect("allocator must not exhaust");
            max_layer = max_layer.max(lease.layer());
            leases.push(lease);
        }
        assert!(
            max_layer >= 64,
            "expected to allocate past layer 63, got max layer {max_layer}",
        );
        assert!(
            alloc.in_use().bits().len() >= 2,
            "in_use bitset should have grown past one 64-bit word",
        );
    }

    #[test]
    fn test_alloc_returns_layer_64_when_first_word_is_full() {
        // Direct trace of the scenario the supposed bug report describes:
        // after every non-reserved layer in the first word is taken, the next
        // alloc must return layer 64, not None.
        let mut alloc = RenderLayerAllocator::default();
        let mut leases = Vec::new();
        let first_word_reserved = alloc.reserved.bits().first().copied().unwrap_or_default();
        let first_word_allocations = 64 - first_word_reserved.count_ones() as usize;
        for _ in 0..first_word_allocations {
            leases.push(alloc.alloc().expect("layer < 64"));
        }
        assert_eq!(
            alloc.in_use().bits(),
            &[u64::MAX],
            "first 64-bit word should be saturated",
        );
        let next = alloc.alloc().expect("layer 64 must be available");
        assert_eq!(next.layer(), 64);
    }

    #[test]
    #[should_panic(expected = "RenderLayerLease already present")]
    fn test_insert_render_layer_lease_panics_on_double_insert() {
        // Reproduces the silent-overwrite bug the helper guards against:
        // inserting a second RenderLayerLease on an entity that already holds
        // one would drop the first Arc and free its layer behind the back of
        // any other entity still pointing at it.
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.init_resource::<RenderLayerAllocator>();

        let mut alloc = app.world_mut().resource_mut::<RenderLayerAllocator>();
        let lease_a = alloc.alloc().expect("layer 1");
        let lease_b = alloc.alloc().expect("layer 2");

        let entity = app.world_mut().spawn_empty().id();
        let mut commands = app.world_mut().commands();
        commands
            .entity(entity)
            .insert_render_layer_lease(lease_a)
            .insert_render_layer_lease(lease_b);
        app.world_mut().flush();
    }

    #[test]
    fn test_insert_render_layer_lease_inserts_lease_and_layers() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.init_resource::<RenderLayerAllocator>();

        let lease = app
            .world_mut()
            .resource_mut::<RenderLayerAllocator>()
            .alloc()
            .expect("layer 1");
        let expected_layers = lease.render_layers();

        let entity = app.world_mut().spawn_empty().id();
        let mut commands = app.world_mut().commands();
        commands.entity(entity).insert_render_layer_lease(lease);
        app.world_mut().flush();

        let entity_ref = app.world().entity(entity);
        assert!(entity_ref.contains::<RenderLayerLease>());
        assert_eq!(entity_ref.get::<RenderLayers>(), Some(&expected_layers));
    }

    #[test]
    fn test_arc_clone_delays_free_until_last_drop() {
        let mut alloc = RenderLayerAllocator::default();
        let a = alloc.alloc().expect("layer 1");
        let b = Arc::clone(&a.0);
        drop(a);
        alloc.drain_dropped();
        let c = alloc.alloc().expect("layer 2 while 1 still held");
        assert_eq!(c.layer(), 2);
        drop(b);
        alloc.drain_dropped();
        let d = alloc.alloc().expect("layer 1 free again");
        assert_eq!(d.layer(), 1);
    }
}
