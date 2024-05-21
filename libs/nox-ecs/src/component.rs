use nox::{IntoOp, Scalar, ScalarExt};

use nox_ecs_macros::Component;

pub trait Component:
    conduit::Component + IntoOp + for<'a> nox::FromBuilder<Item<'a> = Self>
{
}

#[derive(Component)]
pub struct WorldPos(pub nox::SpatialTransform<f64>);

#[derive(Component)]
pub struct Seed(pub Scalar<u64>);

#[derive(Component)]
pub struct Time(pub Scalar<f64>);

impl Seed {
    pub fn zero() -> Self {
        Seed(0u64.constant())
    }
}

impl Time {
    pub fn zero() -> Self {
        Time(0f64.constant())
    }
}

#[cfg(test)]
mod tests {
    use crate::{Seed, WorldPos};
    use conduit::Component;

    #[test]
    fn component_names() {
        assert_eq!(WorldPos::name(), "world_pos");
        assert_eq!(Seed::name(), "seed");
    }
}
