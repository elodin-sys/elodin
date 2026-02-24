use impeller2::component::Component;
use impeller2_wkt::ComponentMetadata;
use nox::NoxprNode;

use impeller2::schema::Schema;

use crate::PyComponent;
use crate::World;

use numpy::PyUntypedArray;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyAnyMethods;
use pyo3::{Bound, FromPyObject, PyAny, PyResult};

pub trait Archetype {
    fn components() -> Vec<(Schema<Vec<u64>>, ComponentMetadata)>;
    fn insert_into_world(self, world: &mut World);
}

impl<T: crate::Component + nox::ReprMonad<nox::Op> + 'static> Archetype for T {
    fn components() -> Vec<(Schema<Vec<u64>>, ComponentMetadata)> {
        vec![(T::schema(), T::metadata())]
    }

    fn insert_into_world(self, world: &mut World) {
        use std::ops::Deref;
        let mut col = world.column_mut::<T>().unwrap();
        let op = self.into_inner();
        let NoxprNode::Constant(c) = op.deref() else {
            panic!("push into host column must be constant expr");
        };
        col.push_raw(c.data.raw_buf());
    }
}

pub trait ComponentExt: Component {
    fn metadata() -> ComponentMetadata {
        ComponentMetadata {
            name: Self::NAME.into(),
            metadata: Default::default(),
            component_id: Self::COMPONENT_ID,
        }
    }
}

impl<C: Component> ComponentExt for C {}

#[derive(Debug)]
pub struct ArchetypeData<'py> {
    pub component_data: Vec<PyComponent>,
    pub arrays: Vec<Bound<'py, PyUntypedArray>>,
}

impl ArchetypeData<'_> {
    pub fn component_names(&self) -> Vec<String> {
        self.component_data
            .iter()
            .map(|data| data.name.to_string())
            .collect::<Vec<_>>()
    }
}

impl<'s> FromPyObject<'s> for ArchetypeData<'s> {
    fn extract_bound(archetype: &Bound<'s, PyAny>) -> PyResult<Self> {
        let component_data = archetype
            .call_method0("component_data")?
            .extract::<Vec<PyComponent>>()?;
        let arrays = archetype.call_method0("arrays")?;
        let arrays = arrays.extract::<Vec<Bound<'_, numpy::PyUntypedArray>>>()?;
        Ok(Self {
            component_data,
            arrays,
        })
    }
}

pub enum Spawnable<'py> {
    Archetypes(Vec<ArchetypeData<'py>>),
}

impl<'py> FromPyObject<'py> for Spawnable<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(archetype_seq) = ob.extract::<Vec<ArchetypeData>>() {
            Ok(Self::Archetypes(archetype_seq))
        } else if ob.getattr("component_data").is_ok() {
            let archetype = ArchetypeData::extract_bound(ob)?;
            Ok(Self::Archetypes(vec![archetype]))
        } else {
            Err(PyValueError::new_err("Not spawnable"))
        }
    }
}
