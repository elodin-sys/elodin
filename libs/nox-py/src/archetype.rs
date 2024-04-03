use crate::*;

use conduit::AssetId;
use nox_ecs::{conduit, ArchetypeName};

use numpy::PyUntypedArray;

pub struct Archetype<'py> {
    pub component_datas: Vec<Component>,
    pub component_ids: Vec<ComponentId>,
    pub arrays: Vec<&'py PyUntypedArray>,
    pub archetype_name: ArchetypeName,
}

impl<'s> FromPyObject<'s> for Archetype<'s> {
    fn extract(archetype: &'s PyAny) -> PyResult<Self> {
        let archetype_name = archetype
            .call_method0("archetype_name")?
            .extract::<String>()?;
        let archetype_name = ArchetypeName::from(archetype_name.as_str());
        let component_datas = archetype
            .call_method0("component_data")?
            .extract::<Vec<Component>>()?;
        let component_ids = component_datas
            .iter()
            .map(|data| data.id)
            .collect::<Vec<_>>();
        let arrays = archetype.call_method0("arrays")?;
        let arrays = arrays.extract::<Vec<&numpy::PyUntypedArray>>()?;
        Ok(Self {
            component_datas,
            component_ids,
            arrays,
            archetype_name,
        })
    }
}

pub enum Spawnable<'py> {
    Archetype(Archetype<'py>),
    Asset { id: AssetId, bytes: PyBufBytes },
}

impl<'py> FromPyObject<'py> for Spawnable<'py> {
    fn extract(ob: &'py PyAny) -> PyResult<Self> {
        if let Ok(archetype) = Archetype::extract(ob) {
            Ok(Self::Archetype(archetype))
        } else {
            let id: u64 = ob.call_method0("asset_id")?.extract()?;
            let bytes = ob.call_method0("bytes")?.extract()?;
            Ok(Self::Asset {
                id: AssetId(id),
                bytes,
            })
        }
    }
}
