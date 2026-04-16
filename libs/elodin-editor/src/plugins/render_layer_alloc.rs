//! Handle render layer allocation and deallocation. One allocates a
//! `RenderLayerLease` component from the `RenderLayerAllocator` resource. Add
//! it and its `Renderlayer` to whatever entity needs it. When all the
//! `RenderLayerLease`s are dropped, then the allocated render layer is
//! freed for use again.
//!
//! This can work with any render layer, and it uses the bits to find the next
//! available layer, so its fast and uses the lowest available number first.
//!
//! This module was created to avoid a maximum limit of 64 that were not
//! deallocated.
use crate::plugins::gizmos::GIZMO_RENDER_LAYER;
use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use crossbeam_queue::SegQueue;
use std::sync::Arc;

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

    /// Resets dynamic allocations.
    pub fn free_all(&mut self) {
        while self.dropped.pop().is_some() {}
        self.in_use = self.reserved.clone();
    }
}

impl Default for RenderLayerAllocator {
    fn default() -> Self {
        let reserved = RenderLayers::layer(0).with(GIZMO_RENDER_LAYER);
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
