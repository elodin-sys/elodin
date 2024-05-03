use std::path::Path;

use crate::{polars::PolarsWorld, Error, World};

#[derive(Default, Debug, Clone)]
pub struct History {
    pub worlds: Vec<PolarsWorld>,
}

impl History {
    pub fn compact_to_world(&self) -> Result<PolarsWorld, Error> {
        self.worlds
            .iter()
            .try_fold(PolarsWorld::default(), |mut final_world, world| {
                final_world.vstack(world)?;
                Ok::<_, Error>(final_world)
            })
    }

    pub fn write_to_dir(&self, dir: impl AsRef<Path>) -> Result<(), Error> {
        let mut world = self.compact_to_world()?;
        world.write_to_dir(dir)?;
        Ok(())
    }

    pub fn push_world(&mut self, host: &World) -> Result<(), Error> {
        let mut world = host.to_polars()?;
        world.add_tick()?;
        self.worlds.push(world);
        Ok(())
    }
}
