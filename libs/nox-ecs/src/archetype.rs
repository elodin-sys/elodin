use impeller2::component::{Asset, Component};
use impeller2_wkt::ComponentMetadata;
use nox::NoxprNode;

use impeller2::schema::Schema;

use crate::{assets::Handle, World};

pub trait Archetype {
    fn components() -> Vec<(Schema<Vec<u64>>, ComponentMetadata)>;
    fn insert_into_world(self, world: &mut World);
}

impl<T: crate::Component + nox::ReprMonad<nox::Op> + 'static> crate::Archetype for T {
    fn components() -> Vec<(Schema<Vec<u64>>, ComponentMetadata)> {
        vec![(T::schema(), T::metadata())]
    }

    fn insert_into_world(self, world: &mut crate::World) {
        use std::ops::Deref;
        let mut col = world.column_mut::<T>().unwrap();
        let op = self.into_inner();
        let NoxprNode::Constant(c) = op.deref() else {
            panic!("push into host column must be constant expr");
        };
        col.push_raw(c.data.raw_buf());
    }
}

impl<T: Asset + 'static> Archetype for Handle<T> {
    fn insert_into_world(self, world: &mut crate::World) {
        let mut col = world.column_mut::<Self>().unwrap();
        col.push_raw(&self.id.to_le_bytes());
    }

    fn components() -> Vec<(Schema<Vec<u64>>, ComponentMetadata)> {
        vec![(Self::schema(), Self::metadata())]
    }
}

pub trait ComponentExt: Component {
    fn metadata() -> ComponentMetadata {
        ComponentMetadata {
            name: Self::NAME.into(),
            asset: Self::ASSET,
            metadata: Default::default(),
            component_id: Self::COMPONENT_ID,
        }
    }
}

impl<C: Component> ComponentExt for C {}
