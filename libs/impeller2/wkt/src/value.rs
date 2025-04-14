use impeller2::types::{ComponentView, ElementValue, PrimType};
use nox::{Array, ArrayBuf, Dyn};
use serde::{Deserialize, Serialize};
use zerocopy::IntoBytes;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ComponentValue {
    U8(Array<u8, Dyn>),
    U16(Array<u16, Dyn>),
    U32(Array<u32, Dyn>),
    U64(Array<u64, Dyn>),
    I8(Array<i8, Dyn>),
    I16(Array<i16, Dyn>),
    I32(Array<i32, Dyn>),
    I64(Array<i64, Dyn>),
    Bool(Array<bool, Dyn>),
    F32(Array<f32, Dyn>),
    F64(Array<f64, Dyn>),
}

impl ComponentValue {
    pub fn zeros(shape: &[usize], prim_type: PrimType) -> Self {
        match prim_type {
            PrimType::U8 => Self::U8(Array::zeroed(shape)),
            PrimType::U16 => Self::U16(Array::zeroed(shape)),
            PrimType::U32 => Self::U32(Array::zeroed(shape)),
            PrimType::U64 => Self::U64(Array::zeroed(shape)),
            PrimType::I8 => Self::I8(Array::zeroed(shape)),
            PrimType::I16 => Self::I16(Array::zeroed(shape)),
            PrimType::I32 => Self::I32(Array::zeroed(shape)),
            PrimType::I64 => Self::I64(Array::zeroed(shape)),
            PrimType::Bool => Self::Bool(Array::zeroed(shape)),
            PrimType::F32 => Self::F32(Array::zeroed(shape)),
            PrimType::F64 => Self::F64(Array::zeroed(shape)),
        }
    }

    pub fn fill_zeros(&mut self) {
        match self {
            Self::U8(a) => {
                a.buf.as_mut_buf().fill(0);
            }
            Self::U16(a) => {
                a.buf.as_mut_buf().fill(0);
            }
            Self::U32(a) => {
                a.buf.as_mut_buf().fill(0);
            }
            Self::U64(a) => {
                a.buf.as_mut_buf().fill(0);
            }
            Self::I8(a) => {
                a.buf.as_mut_buf().fill(0);
            }
            Self::I16(a) => {
                a.buf.as_mut_buf().fill(0);
            }
            Self::I32(a) => {
                a.buf.as_mut_buf().fill(0);
            }
            Self::I64(a) => {
                a.buf.as_mut_buf().fill(0);
            }
            Self::Bool(a) => {
                a.buf.as_mut_buf().fill(false);
            }
            Self::F32(a) => {
                a.buf.as_mut_buf().fill(0.0);
            }
            Self::F64(a) => {
                a.buf.as_mut_buf().fill(0.0);
            }
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::U8(arr) => arr.shape(),
            Self::U16(arr) => arr.shape(),
            Self::U32(arr) => arr.shape(),
            Self::U64(arr) => arr.shape(),
            Self::I8(arr) => arr.shape(),
            Self::I16(arr) => arr.shape(),
            Self::I32(arr) => arr.shape(),
            Self::I64(arr) => arr.shape(),
            Self::Bool(arr) => arr.shape(),
            Self::F32(arr) => arr.shape(),
            Self::F64(arr) => arr.shape(),
        }
    }

    pub fn add_view(&mut self, view: ComponentView<'_>) {
        match (self, view) {
            (Self::U8(arr), ComponentView::U8(view)) => {
                for (i, &val) in view.buf().iter().enumerate() {
                    if let Some(r) = arr.buf.as_mut_buf().get_mut(i) {
                        *r = r.saturating_add(val);
                    }
                }
            },
            (Self::U16(arr), ComponentView::U16(view)) => {
                for (i, &val) in view.buf().iter().enumerate() {
                    if let Some(r) = arr.buf.as_mut_buf().get_mut(i) {
                        *r = r.saturating_add(val);
                    }
                }
            },
            (Self::U32(arr), ComponentView::U32(view)) => {
                for (i, &val) in view.buf().iter().enumerate() {
                    if let Some(r) = arr.buf.as_mut_buf().get_mut(i) {
                        *r = r.saturating_add(val);
                    }
                }
            },
            (Self::U64(arr), ComponentView::U64(view)) => {
                for (i, &val) in view.buf().iter().enumerate() {
                    if let Some(r) = arr.buf.as_mut_buf().get_mut(i) {
                        *r = r.saturating_add(val);
                    }
                }
            },
            (Self::I8(arr), ComponentView::I8(view)) => {
                for (i, &val) in view.buf().iter().enumerate() {
                    if let Some(r) = arr.buf.as_mut_buf().get_mut(i) {
                        *r = r.saturating_add(val);
                    }
                }
            },
            (Self::I16(arr), ComponentView::I16(view)) => {
                for (i, &val) in view.buf().iter().enumerate() {
                    if let Some(r) = arr.buf.as_mut_buf().get_mut(i) {
                        *r = r.saturating_add(val);
                    }
                }
            },
            (Self::I32(arr), ComponentView::I32(view)) => {
                for (i, &val) in view.buf().iter().enumerate() {
                    if let Some(r) = arr.buf.as_mut_buf().get_mut(i) {
                        *r = r.saturating_add(val);
                    }
                }
            },
            (Self::I64(arr), ComponentView::I64(view)) => {
                for (i, &val) in view.buf().iter().enumerate() {
                    if let Some(r) = arr.buf.as_mut_buf().get_mut(i) {
                        *r = r.saturating_add(val);
                    }
                }
            },
            (Self::Bool(arr), ComponentView::Bool(view)) => {
                for (i, &val) in view.buf().iter().enumerate() {
                    if let Some(r) = arr.buf.as_mut_buf().get_mut(i) {
                        *r = *r || val; // Logical OR for booleans
                    }
                }
            },
            (Self::F32(arr), ComponentView::F32(view)) => {
                for (i, &val) in view.buf().iter().enumerate() {
                    if let Some(r) = arr.buf.as_mut_buf().get_mut(i) {
                        *r += val;
                    }
                }
            },
            (Self::F64(arr), ComponentView::F64(view)) => {
                for (i, &val) in view.buf().iter().enumerate() {
                    if let Some(r) = arr.buf.as_mut_buf().get_mut(i) {
                        *r += val;
                    }
                }
            },
            _ => panic!("Cannot add values of different types"),
        }
    }

    pub fn div(&mut self, count: f64) {
        match self {
            Self::U8(a) => {
                for r in a.buf.as_mut_buf().iter_mut() {
                    *r = (*r as f64 / count) as u8;
                }
            },
            Self::U16(a) => {
                for r in a.buf.as_mut_buf().iter_mut() {
                    *r = (*r as f64 / count) as u16;
                }
            },
            Self::U32(a) => {
                for r in a.buf.as_mut_buf().iter_mut() {
                    *r = (*r as f64 / count) as u32;
                }
            },
            Self::U64(a) => {
                for r in a.buf.as_mut_buf().iter_mut() {
                    *r = (*r as f64 / count) as u64;
                }
            },
            Self::I8(a) => {
                for r in a.buf.as_mut_buf().iter_mut() {
                    *r = (*r as f64 / count) as i8;
                }
            },
            Self::I16(a) => {
                for r in a.buf.as_mut_buf().iter_mut() {
                    *r = (*r as f64 / count) as i16;
                }
            },
            Self::I32(a) => {
                for r in a.buf.as_mut_buf().iter_mut() {
                    *r = (*r as f64 / count) as i32;
                }
            },
            Self::I64(a) => {
                for r in a.buf.as_mut_buf().iter_mut() {
                    *r = (*r as f64 / count ) as i64;
                }
            },
            Self::Bool(_) => panic!("Cannot divide boolean values"),
            Self::F32(a) => {
                for r in a.buf.as_mut_buf().iter_mut() {
                    *r /= count as f32;
                }
            },
            Self::F64(a) => {
                for r in a.buf.as_mut_buf().iter_mut() {
                    *r /= count ;
                }
            },
        }
    }
    pub fn copy_from_view(&mut self, view: ComponentView<'_>) {
        match (self, view) {
            (Self::U8(arr), ComponentView::U8(view)) => {
                arr.buf.as_mut_buf().copy_from_slice(view.buf());
            }
            (Self::U16(arr), ComponentView::U16(view)) => {
                arr.buf.as_mut_buf().copy_from_slice(view.buf());
            }
            (Self::U32(arr), ComponentView::U32(view)) => {
                arr.buf.as_mut_buf().copy_from_slice(view.buf());
            }
            (Self::U64(arr), ComponentView::U64(view)) => {
                arr.buf.as_mut_buf().copy_from_slice(view.buf());
            }
            (Self::I8(arr), ComponentView::I8(view)) => {
                arr.buf.as_mut_buf().copy_from_slice(view.buf());
            }
            (Self::I16(arr), ComponentView::I16(view)) => {
                arr.buf.as_mut_buf().copy_from_slice(view.buf());
            }
            (Self::I32(arr), ComponentView::I32(view)) => {
                arr.buf.as_mut_buf().copy_from_slice(view.buf());
            }
            (Self::I64(arr), ComponentView::I64(view)) => {
                arr.buf.as_mut_buf().copy_from_slice(view.buf());
            }
            (Self::Bool(arr), ComponentView::Bool(view)) => {
                arr.buf.as_mut_buf().copy_from_slice(view.buf());
            }
            (Self::F32(arr), ComponentView::F32(view)) => {
                arr.buf.as_mut_buf().copy_from_slice(view.buf());
            }
            (Self::F64(arr), ComponentView::F64(view)) => {
                arr.buf.as_mut_buf().copy_from_slice(view.buf());
            }
            _ => panic!("Incompatible component value and view types"),
        }
    }

    pub fn from_view(view: ComponentView<'_>) -> Self {
        match view {
            ComponentView::U8(view) => Self::U8(view.to_dyn_owned()),
            ComponentView::U16(view) => Self::U16(view.to_dyn_owned()),
            ComponentView::U32(view) => Self::U32(view.to_dyn_owned()),
            ComponentView::U64(view) => Self::U64(view.to_dyn_owned()),
            ComponentView::I8(view) => Self::I8(view.to_dyn_owned()),
            ComponentView::I16(view) => Self::I16(view.to_dyn_owned()),
            ComponentView::I32(view) => Self::I32(view.to_dyn_owned()),
            ComponentView::I64(view) => Self::I64(view.to_dyn_owned()),
            ComponentView::Bool(view) => Self::Bool(view.to_dyn_owned()),
            ComponentView::F32(view) => Self::F32(view.to_dyn_owned()),
            ComponentView::F64(view) => Self::F64(view.to_dyn_owned()),
        }
    }

    pub fn iter<'i>(&'i self) -> Box<dyn Iterator<Item = ElementValue> + 'i> {
        match self {
            ComponentValue::U8(u8) => {
                Box::new(u8.buf.as_buf().iter().map(|&x| ElementValue::U8(x)))
            }
            ComponentValue::U16(u16) => {
                Box::new(u16.buf.as_buf().iter().map(|&x| ElementValue::U16(x)))
            }
            ComponentValue::U32(u32) => {
                Box::new(u32.buf.as_buf().iter().map(|&x| ElementValue::U32(x)))
            }
            ComponentValue::U64(u64) => {
                Box::new(u64.buf.as_buf().iter().map(|&x| ElementValue::U64(x)))
            }
            ComponentValue::I8(i8) => {
                Box::new(i8.buf.as_buf().iter().map(|&x| ElementValue::I8(x)))
            }
            ComponentValue::I16(i16) => {
                Box::new(i16.buf.as_buf().iter().map(|&x| ElementValue::I16(x)))
            }
            ComponentValue::I32(i32) => {
                Box::new(i32.buf.as_buf().iter().map(|&x| ElementValue::I32(x)))
            }
            ComponentValue::I64(i64) => {
                Box::new(i64.buf.as_buf().iter().map(|&x| ElementValue::I64(x)))
            }
            ComponentValue::Bool(bool) => {
                Box::new(bool.buf.as_buf().iter().map(|&x| ElementValue::Bool(x)))
            }
            ComponentValue::F32(f32) => {
                Box::new(f32.buf.as_buf().iter().map(|&x| ElementValue::F32(x)))
            }
            ComponentValue::F64(f64) => {
                Box::new(f64.buf.as_buf().iter().map(|&x| ElementValue::F64(x)))
            }
        }
    }

    pub fn get(&self, i: usize) -> Option<ElementValue> {
        match self {
            ComponentValue::U8(x) => x.buf.as_buf().get(i).map(|&x| ElementValue::U8(x)),
            ComponentValue::U16(x) => x.buf.as_buf().get(i).map(|&x| ElementValue::U16(x)),
            ComponentValue::U32(x) => x.buf.as_buf().get(i).map(|&x| ElementValue::U32(x)),
            ComponentValue::U64(x) => x.buf.as_buf().get(i).map(|&x| ElementValue::U64(x)),
            ComponentValue::I8(x) => x.buf.as_buf().get(i).map(|&x| ElementValue::I8(x)),
            ComponentValue::I16(x) => x.buf.as_buf().get(i).map(|&x| ElementValue::I16(x)),
            ComponentValue::I32(x) => x.buf.as_buf().get(i).map(|&x| ElementValue::I32(x)),
            ComponentValue::I64(x) => x.buf.as_buf().get(i).map(|&x| ElementValue::I64(x)),
            ComponentValue::Bool(x) => x.buf.as_buf().get(i).map(|&x| ElementValue::Bool(x)),
            ComponentValue::F32(x) => x.buf.as_buf().get(i).map(|&x| ElementValue::F32(x)),
            ComponentValue::F64(x) => x.buf.as_buf().get(i).map(|&x| ElementValue::F64(x)),
        }
    }

    pub fn prim_type(&self) -> PrimType {
        match self {
            ComponentValue::U8(_) => PrimType::U8,
            ComponentValue::U16(_) => PrimType::U16,
            ComponentValue::U32(_) => PrimType::U32,
            ComponentValue::U64(_) => PrimType::U64,
            ComponentValue::I8(_) => PrimType::I8,
            ComponentValue::I16(_) => PrimType::I16,
            ComponentValue::I32(_) => PrimType::I32,
            ComponentValue::I64(_) => PrimType::I64,
            ComponentValue::Bool(_) => PrimType::Bool,
            ComponentValue::F32(_) => PrimType::F32,
            ComponentValue::F64(_) => PrimType::F64,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        match self {
            ComponentValue::U8(x) => x.buf.as_buf().as_bytes(),
            ComponentValue::U16(x) => x.buf.as_buf().as_bytes(),
            ComponentValue::U32(x) => x.buf.as_buf().as_bytes(),
            ComponentValue::U64(x) => x.buf.as_buf().as_bytes(),
            ComponentValue::I8(x) => x.buf.as_buf().as_bytes(),
            ComponentValue::I16(x) => x.buf.as_buf().as_bytes(),
            ComponentValue::I32(x) => x.buf.as_buf().as_bytes(),
            ComponentValue::I64(x) => x.buf.as_buf().as_bytes(),
            ComponentValue::Bool(x) => x.buf.as_buf().as_bytes(),
            ComponentValue::F32(x) => x.buf.as_buf().as_bytes(),
            ComponentValue::F64(x) => x.buf.as_buf().as_bytes(),
        }
    }
}

#[derive(Debug)]
pub enum ElementValueMut<'a> {
    U8(&'a mut u8),
    U16(&'a mut u16),
    U32(&'a mut u32),
    U64(&'a mut u64),
    I8(&'a mut i8),
    I16(&'a mut i16),
    I32(&'a mut i32),
    I64(&'a mut i64),
    F64(&'a mut f64),
    F32(&'a mut f32),
    Bool(&'a mut bool),
}
