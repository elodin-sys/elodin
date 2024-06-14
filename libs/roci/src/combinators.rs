use crate::{Componentize, Decomponentize, DriverMode, System};

pub struct Pipe<L: System, R: System> {
    left: L,
    right: R,
}

impl<LW, RW, L, R> System for Pipe<L, R>
where
    LW: Default + Componentize + Decomponentize,
    RW: Default + Componentize + Decomponentize,
    L: System<World = LW>,
    R: System<World = RW>,
    L::Driver: DriverMode<Output = <R::Driver as DriverMode>::Input>,
{
    type World = (LW, RW);
    type Driver = R::Driver;

    fn update(&mut self, (lw, rw): &mut Self::World) {
        self.left.update(lw);
        lw.sink_columns(rw);
        self.right.update(rw);
    }

    const MAX_SIZE: usize = Self::World::MAX_SIZE;

    fn init_world(&mut self) -> Self::World {
        (self.left.init_world(), self.right.init_world())
    }
}

pub trait PipeExt {
    fn pipe<R: System>(self, right: R) -> Pipe<Self, R>
    where
        Self: System + Sized;
}

impl<L: System> PipeExt for L {
    fn pipe<R: System>(self, right: R) -> Pipe<Self, R> {
        Pipe { left: self, right }
    }
}
