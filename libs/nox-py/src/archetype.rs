use crate::*;

use numpy::PyUntypedArray;

#[derive(Debug)]
pub struct Archetype<'py> {
    pub component_data: Vec<Component>,
    pub arrays: Vec<&'py PyUntypedArray>,
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
    fn extract(archetype: &'s PyAny) -> PyResult<Self> {
        let component_data = archetype
            .call_method0("component_data")?
            .extract::<Vec<Component>>()?;
        let arrays = archetype.call_method0("arrays")?;
        let arrays = arrays.extract::<Vec<&numpy::PyUntypedArray>>()?;
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
    fn extract(ob: &'py PyAny) -> PyResult<Self> {
        if let Ok(archetype_seq) = ob.extract::<Vec<Archetype>>() {
            Ok(Self::Archetypes(archetype_seq))
        } else if ob.getattr("component_data").is_ok() {
            let archetype = Archetype::extract(ob)?;
            Ok(Self::Archetypes(vec![archetype]))
        } else {
            let name = ob.call_method0("asset_name")?.extract()?;
            let bytes = ob.call_method0("bytes")?.extract()?;
            Ok(Self::Asset { name, bytes })
        }
    }
}
