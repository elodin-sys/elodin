use std::ops::Deref;

use bytemuck::Pod;
use elodin_conduit::{ComponentId, ComponentType, ComponentValue};
use nox::{
    xla::{ArrayElement, PjRtBuffer},
    Client, NoxprNode,
};
use smallvec::SmallVec;

use crate::{Component, DynArrayView, Error};

/// A type erased columnar data store located on the host CPU
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct HostColumn {
    pub(crate) buf: Vec<u8>,
    pub(crate) len: usize,
    pub(crate) component_type: ComponentType,
    pub(crate) component_id: ComponentId,
    pub asset: bool,
}

impl HostColumn {
    pub fn new(component_type: ComponentType, component_id: ComponentId) -> Self {
        HostColumn {
            buf: vec![],
            component_type,
            len: 0,
            asset: false,
            component_id,
        }
    }

    pub fn push<T: Component + 'static>(&mut self, val: T) {
        assert_eq!(self.component_type, T::component_type());
        let op = val.into_op();
        let NoxprNode::Constant(c) = op.deref() else {
            panic!("push into host column must be constant expr");
        };
        self.asset = T::is_asset();
        self.push_raw(c.data.raw_buf());
    }

    pub fn push_raw(&mut self, raw: &[u8]) {
        self.buf.extend_from_slice(raw);
        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn copy_to_client(&self, client: &Client) -> Result<PjRtBuffer, Error> {
        let mut dims: heapless::Vec<usize, 3> = heapless::Vec::default();
        dims.extend(self.component_type.dims().iter().map(|d| *d as usize));
        dims.push(self.len).unwrap();
        client
            .0
            .copy_raw_host_buffer(self.component_type.element_type(), &self.buf, &dims[..])
            .map_err(Error::from)
    }

    pub fn values_iter(&self) -> impl Iterator<Item = ComponentValue<'_>> + '_ {
        let mut buf_offset = 0;
        std::iter::from_fn(move || {
            let buf = self.buf.get(buf_offset..)?;
            let (offset, value) = self.component_type.parse(buf)?;
            buf_offset += offset;
            Some(value)
        })
    }

    pub fn iter<T: elodin_conduit::Component>(&self) -> impl Iterator<Item = T> + '_ {
        assert_eq!(self.component_type, T::component_type());
        self.values_iter()
            .filter_map(|v| T::from_component_value(v))
    }

    pub fn component_type(&self) -> ComponentType {
        self.component_type
    }

    pub fn raw_buf(&self) -> &[u8] {
        &self.buf
    }

    pub fn typed_buf<T: ArrayElement + Pod>(&self) -> Option<&[T]> {
        if self.component_type.element_type() != T::TY {
            return None;
        }
        bytemuck::try_cast_slice(self.buf.as_slice()).ok()
    }

    pub fn ndarray<T: ArrayElement + Pod>(&self) -> Option<ndarray::ArrayViewD<'_, T>> {
        let shape: SmallVec<[usize; 4]> = std::iter::once(self.len)
            .chain(self.component_type.dims().iter().map(|x| *x as usize))
            .collect();
        let buf = self.typed_buf::<T>()?;
        ndarray::ArrayViewD::from_shape(&shape[..], buf).ok()
    }

    pub fn dyn_ndarray(&self) -> Option<DynArrayView<'_>> {
        let elem_type = self.component_type.element_type();
        match elem_type {
            nox::xla::ElementType::Pred => {
                todo!()
            }
            nox::xla::ElementType::S8 => {
                let buf = self.ndarray::<i8>()?;
                Some(DynArrayView::I8(buf))
            }
            nox::xla::ElementType::S16 => {
                let buf = self.ndarray::<i16>()?;
                Some(DynArrayView::I16(buf))
            }
            nox::xla::ElementType::S32 => {
                let buf = self.ndarray::<i32>()?;
                Some(DynArrayView::I32(buf))
            }
            nox::xla::ElementType::S64 => {
                let buf = self.ndarray::<i64>()?;
                Some(DynArrayView::I64(buf))
            }
            nox::xla::ElementType::U8 => {
                let buf = self.ndarray::<u8>()?;
                Some(DynArrayView::U8(buf))
            }
            nox::xla::ElementType::U16 => {
                let buf = self.ndarray::<u16>()?;
                Some(DynArrayView::U16(buf))
            }
            nox::xla::ElementType::U32 => {
                let buf = self.ndarray::<u32>()?;
                Some(DynArrayView::U32(buf))
            }
            nox::xla::ElementType::U64 => {
                let buf = self.ndarray::<u64>()?;
                Some(DynArrayView::U64(buf))
            }
            nox::xla::ElementType::F32 => {
                let buf = self.ndarray::<f32>()?;
                Some(DynArrayView::F32(buf))
            }
            nox::xla::ElementType::F64 => {
                let buf = self.ndarray::<f64>()?;
                Some(DynArrayView::F64(buf))
            }
            nox::xla::ElementType::F16 => {
                todo!()
            }
            nox::xla::ElementType::Bf16 => {
                todo!()
            }
            nox::xla::ElementType::C64 => {
                todo!()
            }
            nox::xla::ElementType::C128 => {
                todo!()
            }
        }
    }
}
