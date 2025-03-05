use crate::*;

use numpy::PyUntypedArray;
use pyo3::types::PyAnyMethods;
use pyo3::{FromPyObject, PyAny};

#[derive(Debug)]
pub struct Archetype<'py> {
    pub component_data: Vec<Component>,
    pub arrays: Vec<Bound<'py, PyUntypedArray>>,
}

impl Archetype<'_> {
    pub fn component_names(&self) -> Vec<String> {
        self.component_data
            .iter()
            .map(|data| data.name.to_string())
            .collect::<Vec<_>>()
    }
}

impl<'s> FromPyObject<'s> for Archetype<'s> {
    fn extract_bound(archetype: &Bound<'s, PyAny>) -> PyResult<Self> {
        let component_data = archetype
            .call_method0("component_data")?
            .extract::<Vec<Component>>()?;
        let arrays = archetype.call_method0("arrays")?;
        let arrays = arrays.extract::<Vec<Bound<'_, numpy::PyUntypedArray>>>()?;
        Ok(Self {
            component_data,
            arrays,
        })
    }
}

pub enum Spawnable<'py> {
    Archetypes(Vec<Archetype<'py>>),
    Asset { name: String, bytes: PyBufBytes },
}

impl<'py> FromPyObject<'py> for Spawnable<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(archetype_seq) = ob.extract::<Vec<Archetype>>() {
            Ok(Self::Archetypes(archetype_seq))
        } else if ob.getattr("component_data").is_ok() {
            let archetype = Archetype::extract_bound(ob)?;
            Ok(Self::Archetypes(vec![archetype]))
        } else {
            let name = ob.call_method0("asset_name")?.extract()?;
            let bytes = ob.call_method0("bytes")?.extract()?;
            Ok(Self::Asset { name, bytes })
        }
    }
}
