use crate::{ArrayElement, Error, NativeType, PrimitiveType, RawShape, Result, Shape, Status};
use bytemuck::AnyBitPattern;
use cpp::{cpp, cpp_class};

use num_traits::FromPrimitive;
use std::pin::Pin;

cpp! {{
    #include "xla/client/lib/constants.h"
    #include "xla/client/lib/matrix.h"
    #include "xla/statusor.h"
    #include "xla/literal_util.h"
    using namespace xla;
}}
cpp_class!(pub unsafe struct Literal as "std::shared_ptr<Literal>");

impl Literal {
    pub fn raw_buf(&self) -> &[u8] {
        let len: Pin<&mut usize> = std::pin::pin!(0);
        let data = unsafe {
            let data = cpp!([self as "std::shared_ptr<Literal>*", len as "size_t*"] -> *const u8 as "const uint8_t*" {
                *len = (*self)->size_bytes();
                return (const uint8_t*) (*self)->untyped_data();
            });
            std::slice::from_raw_parts(data, *len)
        };
        data
    }

    pub fn primitive_type(&self) -> Result<PrimitiveType> {
        let ty = unsafe {
            cpp!([self as "std::shared_ptr<Literal>*"] -> i32 as "int32_t" {
                return (*self)->shape().element_type();
            })
        };
        match FromPrimitive::from_i32(ty) {
            None => Err(Error::UnexpectedElementType(ty)),
            Some(ty) => Ok(ty),
        }
    }

    pub fn element_count(&self) -> usize {
        unsafe {
            cpp!([self as "std::shared_ptr<Literal>*"] -> usize as "size_t" {
                return (*self)->element_count();
            })
        }
    }

    pub fn typed_buf<T: ArrayElement + AnyBitPattern>(&self) -> Result<&[T]> {
        let ty = self.primitive_type()?.element_type()?;
        if ty != T::TY {
            Err(Error::ElementTypeMismatch {
                on_device: ty,
                on_host: T::TY,
            })?
        }
        bytemuck::try_cast_slice(self.raw_buf()).map_err(Error::PodCastError)
    }

    pub fn reshape(&self, dims: &[i64]) -> Result<Literal> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let dims_ptr = dims.as_ptr();
        let dims_len = dims.len();
        let lit = unsafe {
            cpp!([self as "std::shared_ptr<Literal>*", dims_ptr as "const int64_t*", dims_len as "size_t", out_status as "Status*"] -> Literal as "std::shared_ptr<Literal>" {
                auto status = (*self)->Reshape(absl::Span(dims_ptr, dims_len));
                if (status.ok()) {
                    return std::make_shared<Literal>(std::move(status.value()));
                }else{
                    *out_status = Status(status.status());
                    return std::make_shared<Literal>(Literal());
                }
            })
        };
        out_status.to_result()?;
        Ok(lit)
    }

    pub fn vector<T: NativeType>(vals: &[T]) -> Literal {
        T::create_r1(vals)
    }

    pub fn scalar<T: NativeType>(val: T) -> Literal {
        val.literal()
    }

    pub fn raw_shape(&self) -> RawShape {
        unsafe {
            cpp!([self as "std::shared_ptr<Literal>*"] -> RawShape as "Shape" {
                return (*self)->shape();
            })
        }
    }

    pub fn shape(&self) -> Result<Shape> {
        self.raw_shape().shape()
    }

    pub fn decompose_tuple(&mut self) -> Result<Vec<Literal>> {
        match self.shape()? {
            Shape::Array(_) => Ok(vec![]),
            Shape::Tuple(shapes) => {
                let tuple_len = shapes.len();
                let mut out = vec![Literal::default(); tuple_len];
                let out_ptr = out.as_mut_ptr();
                unsafe {
                    cpp!([self as "std::shared_ptr<Literal>*", out_ptr as "std::shared_ptr<Literal>*"] {
                        auto lits = (*self)->DecomposeTuple();
                        size_t i = 0;
                        for (auto& lit : lits) {
                            out_ptr[i] = std::make_shared<Literal>(std::move(lit));
                            i++;
                        }
                    });
                }
                Ok(out)
            }
        }
    }
}
