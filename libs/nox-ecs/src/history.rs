use std::path::Path;

use polars::{frame::DataFrame, series::Series};

use crate::{polars::PolarsWorld, Error, World};

#[derive(Default, Debug, Clone)]
pub struct History {
    worlds: Vec<PolarsWorld>,
}

impl History {
    pub fn compact_to_world(&self) -> Result<Option<PolarsWorld>, Error> {
        let Some(mut final_world) = self.worlds.first().cloned() else {
            return Ok(None);
        };
        for df in final_world.archetypes.values_mut() {
            add_time(df, 0)?;
        }
        for (time, world) in self.worlds.iter().enumerate().skip(1) {
            let mut world = world.clone();
            for (tick_df, final_df) in world
                .archetypes
                .values_mut()
                .zip(final_world.archetypes.values_mut())
            {
                add_time(tick_df, time)?;
                final_df.vstack(tick_df)?;
            }
        }
        Ok(Some(final_world))
    }

    pub fn write_to_dir(&self, dir: impl AsRef<Path>) -> Result<(), Error> {
        let Some(mut world) = self.compact_to_world()? else {
            return Ok(());
        };
        world.write_to_dir(dir)?;
        Ok(())
    }

    pub fn push_world(&mut self, host: &World) -> Result<(), Error> {
        let world = host.to_polars()?;
        self.worlds.push(world);
        Ok(())
    }
}

fn add_time(df: &mut DataFrame, time: usize) -> Result<(), Error> {
    let len = df
        .get_columns()
        .first()
        .map(|s| s.len())
        .unwrap_or_default();
    let series: Series = std::iter::repeat(time as u64).take(len).collect();
    df.with_column(series.with_name("time"))?;
    Ok(())
}
