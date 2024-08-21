use nox::{IntoOp, Scalar};

use nox_ecs_macros::Component;

pub trait Component:
    impeller::Component + IntoOp + for<'a> nox::FromBuilder<Item<'a> = Self>
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
        Seed(0u64.into())
    }
}

impl Time {
    pub fn zero() -> Self {
        Time(Scalar::from(0f64))
    }
}

#[cfg(test)]
mod tests {
    use crate::{Seed, WorldPos};
    use impeller::Component;

    #[test]
    fn component_names() {
        assert_eq!(WorldPos::NAME, "world_pos");
        assert_eq!(Seed::NAME, "seed");
    }
}
