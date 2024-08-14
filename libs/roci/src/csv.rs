use std::{io, marker::PhantomData, path::Path};

use serde::Serialize;

use crate::{drivers::DriverMode, Componentize, Decomponentize, System};

pub struct CSVLogger<W, D> {
    writer: ::csv::Writer<std::fs::File>,
    phantom: PhantomData<(W, D)>,
}

impl<W, D> CSVLogger<W, D>
where
    W: Serialize + Default + Componentize + Decomponentize,
    D: DriverMode,
{
    pub fn try_from_path(path: impl AsRef<Path>) -> io::Result<Self> {
        let writer = ::csv::WriterBuilder::new()
            .has_headers(false)
            .from_path(path)?;
        Ok(Self {
            writer,
            phantom: PhantomData,
        })
    }
}

impl<W, D> System for CSVLogger<W, D>
where
    W: Serialize + Default + Componentize + Decomponentize,
    D: DriverMode,
{
    type World = W;

    type Driver = D;

    fn update(&mut self, world: &mut Self::World) {
        if let Err(err) = self.writer.serialize(world) {
            todo!("{:?}", err)
        }
        self.writer.flush().unwrap();
    }
}
