use arrow::array::ArrayData;
use conduit::{ComponentId, ComponentType, EntityId, Metadata, PrimitiveTy};
use polars::prelude::*;
use polars::{frame::DataFrame, series::Series};
use polars_arrow::{
    array::{Array, PrimitiveArray},
    datatypes::ArrowDataType,
};
use std::collections::HashMap;
use std::{fs::File, path::Path};

use crate::{ArchetypeName, Buffers, Error};

#[derive(Debug, Clone)]
pub struct PolarsWorld {
    pub archetypes: ustr::UstrMap<DataFrame>,
    pub metadata: ustr::UstrMap<Vec<Metadata>>,
}

impl PolarsWorld {
    pub fn new(
        component_map: &HashMap<ComponentId, (ArchetypeName, Metadata)>,
        entity_ids: &ustr::UstrMap<Vec<u8>>,
        history: &[Buffers],
    ) -> Result<Self, Error> {
        let metadata = component_map.iter().fold(
            ustr::UstrMap::<Vec<Metadata>>::default(),
            |mut archetype_metadata, (_, (archetype_name, component_metadata))| {
                archetype_metadata
                    .entry(*archetype_name)
                    .or_default()
                    .push(component_metadata.clone());
                archetype_metadata
            },
        );
        let ticks = history.len();
        let mut archetypes = ustr::UstrMap::default();
        for (archetype_name, components) in metadata.iter() {
            let entity_buf = &entity_ids[archetype_name];
            let len = entity_buf.len() / std::mem::size_of::<EntityId>();

            let entity_series = to_series(&entity_buf.repeat(ticks), &EntityId::metadata())?;
            let tick_series = (0..ticks)
                .flat_map(|tick| std::iter::repeat(tick as u64).take(len))
                .collect::<Series>()
                .with_name("tick");

            let mut df = components
                .iter()
                .map(|metadata| {
                    let component_id = metadata.component_id();
                    let buf = history
                        .iter()
                        .map(|buffers| buffers[&component_id].as_slice())
                        .collect::<Vec<&[u8]>>()
                        .concat();
                    to_series(&buf, metadata)
                })
                .collect::<Result<DataFrame, Error>>()?;
            df.with_column(tick_series)?;
            df.with_column(entity_series)?;
            archetypes.insert(*archetype_name, df);
        }
        Ok(Self {
            archetypes,
            metadata,
        })
    }

    pub fn vstack(mut samples: Vec<Self>) -> Result<Self, Error> {
        let mut world = samples.pop().unwrap();
        for sample in samples.into_iter() {
            for (archetype_name, df) in &mut world.archetypes {
                df.vstack_mut(&sample.archetypes[archetype_name])?;
            }
        }
        Ok(world)
    }

    pub fn join_archetypes(&self) -> Result<DataFrame, Error> {
        let mut tables = self.archetypes.values();
        let init = tables.next().cloned().unwrap_or_default();
        let mut keys = vec![EntityId::NAME, "tick"];
        if init.get_column_names().contains(&"sample_number") {
            keys.push("sample_number");
        }
        tables
            .try_fold(init, |agg, df| {
                agg.join(
                    df,
                    &keys,
                    &keys,
                    JoinArgs::new(JoinType::Outer { coalesce: true }),
                )
            })
            .map_err(Error::from)
    }

    pub fn add_sample_number(&mut self, sample_number: usize) -> Result<(), Error> {
        for df in self.archetypes.values_mut() {
            let len = df
                .get_columns()
                .first()
                .map(|s| s.len())
                .unwrap_or_default();
            let series: Series = std::iter::repeat(sample_number as u64).take(len).collect();
            df.with_column(series.with_name("sample_number"))?;
        }
        Ok(())
    }

    pub fn component_map(&self) -> HashMap<ComponentId, (ArchetypeName, Metadata)> {
        self.metadata
            .iter()
            .flat_map(|(archetype_name, metadata)| {
                metadata.iter().map(move |metadata| {
                    (metadata.component_id(), (*archetype_name, metadata.clone()))
                })
            })
            .collect()
    }

    pub fn history(&self) -> Result<Vec<Buffers>, Error> {
        let ticks = self.tick() + 1;
        let mut history = std::iter::repeat_with(Buffers::default)
            .take(ticks as usize)
            .collect::<Vec<_>>();
        for df in self.archetypes.values() {
            df.get_columns()
                .iter()
                .filter(|s| s.name() != "tick" && s.name() != EntityId::NAME)
                .for_each(|series| {
                    let component_id = ComponentId::new(series.name());
                    let buf = series.to_bytes();
                    for (index, chunk) in buf.chunks_exact(buf.len() / ticks as usize).enumerate() {
                        history[index].insert(component_id, chunk.to_vec());
                    }
                });
        }
        Ok(history)
    }

    pub fn write_to_dir(&mut self, path: impl AsRef<Path>) -> Result<(), Error> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;
        let mut metadata = File::create(path.join("metadata.json"))?;
        serde_json::to_writer(&mut metadata, &self.metadata)?;
        for (archetype_name, df) in &mut self.archetypes {
            let path = path.join(format!("{}.parquet", archetype_name));
            let file = std::fs::File::create(&path)?;
            ParquetWriter::new(file).finish(df)?;
        }
        Ok(())
    }

    pub fn read_from_dir(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let mut archetypes = ustr::UstrMap::default();
        let mut metadata = File::open(path.join("metadata.json"))?;
        let metadata: ustr::UstrMap<Vec<Metadata>> = serde_json::from_reader(&mut metadata)?;
        for name in metadata.keys() {
            let path = path.join(format!("{}.parquet", name));
            let file = File::open(&path)?;
            let df = polars::prelude::ParquetReader::new(file).finish()?;
            archetypes.insert(*name, df);
        }
        Ok(Self {
            archetypes,
            metadata,
        })
    }

    pub fn entity_ids(&self) -> Result<ustr::UstrMap<Vec<u8>>, Error> {
        self.archetypes
            .iter()
            .map(|(name, df)| {
                let entity_id_buf = df.column(EntityId::NAME)?.to_bytes();
                Ok((*name, entity_id_buf))
            })
            .collect()
    }

    pub fn tick(&self) -> u64 {
        self.archetypes
            .values()
            .next()
            .expect("at least one archetype exists")
            .column("tick")
            .expect("tick column exists")
            .max::<u64>()
            .unwrap()
            .expect("tick value is not null")
    }

    pub fn entity_len(&self) -> u64 {
        self.archetypes
            .values()
            .map(|df| {
                df.column(EntityId::NAME)
                    .expect("entity id column exists")
                    .max::<u64>()
                    .unwrap()
                    .expect("entity id is not null")
            })
            .max()
            .expect("at least one archetype exists")
    }
}

pub fn to_series(buf: &[u8], metadata: &Metadata) -> Result<Series, Error> {
    let component_type = &metadata.component_type;
    let array = match component_type.primitive_ty {
        PrimitiveTy::F64 => tensor_array(component_type, prim_array::<f64>(buf)),
        PrimitiveTy::F32 => tensor_array(component_type, prim_array::<f32>(buf)),
        PrimitiveTy::U64 => tensor_array(component_type, prim_array::<u64>(buf)),
        PrimitiveTy::U32 => tensor_array(component_type, prim_array::<u32>(buf)),
        PrimitiveTy::U16 => tensor_array(component_type, prim_array::<u16>(buf)),
        PrimitiveTy::U8 => tensor_array(component_type, prim_array::<u8>(buf)),
        PrimitiveTy::I64 => tensor_array(component_type, prim_array::<i64>(buf)),
        PrimitiveTy::I32 => tensor_array(component_type, prim_array::<i32>(buf)),
        PrimitiveTy::I16 => tensor_array(component_type, prim_array::<i16>(buf)),
        PrimitiveTy::I8 => tensor_array(component_type, prim_array::<i8>(buf)),
        PrimitiveTy::Bool => todo!(),
    };
    Series::from_arrow(&metadata.name, array).map_err(Error::from)
}

fn prim_array<T: polars_arrow::types::NativeType>(buf: &[u8]) -> Box<dyn Array> {
    let buf = bytemuck::cast_slice::<_, T>(buf);
    Box::new(PrimitiveArray::from_slice(buf))
}

fn arrow_data_type(ty: PrimitiveTy) -> ArrowDataType {
    match ty {
        PrimitiveTy::U8 => ArrowDataType::UInt8,
        PrimitiveTy::U16 => ArrowDataType::UInt16,
        PrimitiveTy::U32 => ArrowDataType::UInt32,
        PrimitiveTy::U64 => ArrowDataType::UInt64,
        PrimitiveTy::I8 => ArrowDataType::Int8,
        PrimitiveTy::I16 => ArrowDataType::Int16,
        PrimitiveTy::I32 => ArrowDataType::Int32,
        PrimitiveTy::I64 => ArrowDataType::Int64,
        PrimitiveTy::F32 => ArrowDataType::Float32,
        PrimitiveTy::F64 => ArrowDataType::Float64,
        PrimitiveTy::Bool => ArrowDataType::Boolean,
    }
}

fn tensor_array(ty: &ComponentType, inner: Box<dyn Array>) -> Box<dyn Array> {
    let data_type = arrow_data_type(ty.primitive_ty);
    if ty.shape.is_empty() {
        return inner;
    }
    let data_type = ArrowDataType::FixedSizeList(
        Box::new(polars_arrow::datatypes::Field::new(
            "inner", data_type, false,
        )),
        ty.shape.iter().map(|i| *i as usize).product(),
    );
    Box::new(polars_arrow::array::FixedSizeListArray::new(
        data_type, inner, None,
    ))
    // let metadata = HashMap::from_iter([(
    //     "ARROW:extension:metadata".to_string(),
    //     format!("{{ \"shape\": {:?} }}", shape),
    // )]);
    // (data_type, Some(metadata))
}

pub trait SeriesExt {
    fn to_bytes(&self) -> Vec<u8>;
}

impl SeriesExt for Series {
    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::default();
        for chunk in self.chunks() {
            let data = polars_arrow::array::to_data(chunk.as_ref());
            recurse_array_data(&data, &mut out);
        }
        out
    }
}

pub fn recurse_array_data(array_data: &ArrayData, out: &mut Vec<u8>) {
    for child in array_data.child_data() {
        recurse_array_data(child, out)
    }
    for buffer in array_data.buffers() {
        out.extend_from_slice(buffer.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        six_dof::{Body, Force, Inertia, WorldAccel, WorldVel},
        Archetype, World, WorldPos,
    };
    use nox::{
        nalgebra::{self, vector},
        SpatialForce, SpatialInertia, SpatialMotion, SpatialTransform,
    };
    use polars::prelude::*;
    use polars_arrow::array::Float64Array;

    use super::*;

    #[test]
    fn test_convert_to_df() {
        let mut world = World::default();

        world.spawn(Body {
            pos: WorldPos(SpatialTransform {
                inner: vector![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0].into(),
            }),
            vel: WorldVel(SpatialMotion {
                inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 1.0].into(),
            }),
            accel: WorldAccel(SpatialMotion {
                inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
            force: Force(SpatialForce {
                inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
            mass: Inertia(SpatialInertia {
                inner: vector![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0].into(),
            }),
        });
        let polars = world.polars().unwrap();
        let df = polars.archetypes[&Body::name()].clone();
        let out = df
            .lazy()
            .select(&[col(&WorldPos::name())])
            .collect()
            .unwrap();
        let pos = out
            .iter()
            .next()
            .unwrap()
            .array()
            .unwrap()
            .get(0)
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .filter_map(|f| f.copied())
            .collect::<Vec<_>>();
        assert_eq!(pos, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_write_read_file() {
        let mut world = World::default();
        world.spawn(Body {
            pos: WorldPos(SpatialTransform {
                inner: vector![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0].into(),
            }),
            vel: WorldVel(SpatialMotion {
                inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 1.0].into(),
            }),
            accel: WorldAccel(SpatialMotion {
                inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
            force: Force(SpatialForce {
                inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
            mass: Inertia(SpatialInertia {
                inner: vector![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0].into(),
            }),
        });
        let mut polars = world.polars().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let dir = dir.path();
        polars.write_to_dir(dir).unwrap();
        let new_polars = PolarsWorld::read_from_dir(dir).unwrap();
        assert_eq!(polars.archetypes, new_polars.archetypes);
    }

    #[test]
    fn test_to_world() {
        let mut world = World::default();

        world.spawn(Body {
            pos: WorldPos(SpatialTransform {
                inner: vector![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0].into(),
            }),
            vel: WorldVel(SpatialMotion {
                inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 1.0].into(),
            }),
            accel: WorldAccel(SpatialMotion {
                inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
            force: Force(SpatialForce {
                inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
            mass: Inertia(SpatialInertia {
                inner: vector![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0].into(),
            }),
        });
        let polars = world.polars().unwrap();
        let buffers = &polars.history().unwrap()[0];
        assert_eq!(buffers, &world.host);
    }

    #[test]
    fn test_write_read_world() {
        let mut world = World::default();
        world.spawn(Body {
            pos: WorldPos(SpatialTransform {
                inner: vector![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0].into(),
            }),
            vel: WorldVel(SpatialMotion {
                inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 1.0].into(),
            }),
            accel: WorldAccel(SpatialMotion {
                inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
            force: Force(SpatialForce {
                inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
            mass: Inertia(SpatialInertia {
                inner: vector![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0].into(),
            }),
        });
        let mut polars = world.polars().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let dir = dir.path();
        polars.write_to_dir(dir).unwrap();
        let new_polars = PolarsWorld::read_from_dir(dir).unwrap();
        let buffers = &new_polars.history().unwrap()[0];
        assert_eq!(buffers, &world.host);
    }
}
