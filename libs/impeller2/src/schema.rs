use serde::{Deserialize, Serialize};

use crate::{
    buf::{Buf, ByteBufExt},
    error::Error,
    table::ShapeEntry,
    types::{ComponentId, PrimType},
};

#[derive(Serialize, Deserialize)]
pub struct Schema<DataBuf: Buf<u8>> {
    component_id: ComponentId,
    shape_entry: ShapeEntry,
    #[serde(bound(deserialize = ""))]
    data: DataBuf,
}

impl<D: Buf<u8>> Schema<D> {
    pub fn new(
        component_id: ComponentId,
        prim_type: PrimType,
        shape: &[u64],
    ) -> Result<Self, Error> {
        let mut data = D::default();
        let rank = shape.len() as u64;
        let shape_offset = data.extend_aligned(shape)? as u64;
        Ok(Self {
            component_id,
            shape_entry: ShapeEntry {
                prim_type,
                rank,
                shape_offset,
            },
            data,
        })
    }

    pub fn component_id(&self) -> ComponentId {
        self.component_id
    }

    pub fn prim_type(&self) -> PrimType {
        self.shape_entry.prim_type
    }

    pub fn shape(&self) -> &[usize] {
        self.shape_entry
            .parse_shape(self.data.as_slice())
            .expect("shape not found")
    }
}
