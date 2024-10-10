use nox::{Op, OwnedRepr, ReprMonad, Scalar};
use nox_ecs_macros::{Component, ReprMonad};

pub trait Component:
    impeller::Component + for<'a> nox::FromBuilder<Item<'a> = Self> + ReprMonad<Op>
{
}

#[derive(Component, ReprMonad)]
pub struct WorldPos<R: OwnedRepr = Op>(pub nox::SpatialTransform<f64, R>);

#[derive(Component, ReprMonad)]
pub struct Seed<R: OwnedRepr = Op>(pub Scalar<u64, R>);

#[derive(Component, ReprMonad)]
pub struct Time<R: OwnedRepr = Op>(pub Scalar<f64, R>);

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
    use nox::Op;

    #[test]
    fn component_names() {
        assert_eq!(WorldPos::<Op>::NAME, "world_pos");
        assert_eq!(Seed::<Op>::NAME, "seed");
    }
}
