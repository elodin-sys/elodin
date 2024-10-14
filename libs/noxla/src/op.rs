use cpp::{cpp, cpp_class};

use super::ArrayShape;
use crate::Result;
use crate::{PrimitiveType, Status};
use crate::{XlaBuilder, XlaComputation};
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::{Add, Div, Mul, Sub};
use std::pin::Pin;

cpp! {{
    #include "xla/client/xla_builder.h"
    #include "xla/client/lib/constants.h"
    #include "xla/client/lib/matrix.h"
    #include "xla/statusor.h"
    #include "xla/literal_util.h"
    using namespace xla;
}}

cpp_class!(pub unsafe struct XlaOpRaw as "XlaOp");
cpp_class!(pub unsafe struct DotDimensionNumbers as "DotDimensionNumbers");
cpp_class!(pub unsafe struct ScatterDimensionNumbers as "std::unique_ptr<ScatterDimensionNumbers>");

#[derive(Clone)]
pub struct XlaOp {
    pub(crate) raw: XlaOpRaw,
    pub(crate) builder: XlaBuilder,
}

#[repr(transparent)]
pub struct XlaOpRef<'a> {
    _raw: XlaOpRaw, // we directly cast `XlaOpRef` to `xla::XlaOp` in cpp so this field is actually used
    _phantom: PhantomData<&'a ()>,
}

impl XlaOp {
    pub fn as_ref(&self) -> XlaOpRef<'_> {
        XlaOpRef {
            _raw: self.raw,
            _phantom: PhantomData,
        }
    }

    pub fn build(&self) -> Result<XlaComputation> {
        let op = &self.raw;
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let comp = unsafe {
            cpp!([op as "XlaOp*", out_status as "Status*"] -> XlaComputation as "XlaComputation" {
                auto builder = op->builder();
                auto status = builder->Build(*op, false);
                if (status.ok()) {
                    return std::move(status.value());
                }else{
                    *out_status = Status(status.status());
                    return XlaComputation();
                }
            })
        };
        out_status.to_result()?;
        Ok(comp)
    }

    fn wrap(&self, raw: XlaOpRaw) -> Self {
        Self {
            raw,
            builder: self.builder.clone(),
        }
    }

    pub fn add(&self, rhs: &Self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Add(*op, *rhs));
                } catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Sub(*op, *rhs));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Mul(*op, *rhs));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn div(&self, rhs: &Self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Div(*op, *rhs));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn rem(&self, rhs: &Self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Rem(*op, *rhs));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn neg(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Neg(*op));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn abs(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Abs(*op));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn sqrt(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Sqrt(*op));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn pow(&self, rhs: &Self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Pow(*op, *rhs));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn dot(&self, rhs: &Self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Dot(*op, *rhs));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn dot_general(&self, rhs: &Self, dims: DotDimensionNumbers) -> Self {
        let op = &self.raw;
        let dims = ManuallyDrop::new(dims);
        let raw = unsafe {
            cpp!([op as "const XlaOp*", rhs as "const XlaOp*", dims as "DotDimensionNumbers"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(DotGeneral(*op, *rhs, dims));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn atan2(&self, rhs: &Self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Atan2(*op, *rhs));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn max(&self, rhs: &Self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Max(*op, *rhs));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn min(&self, rhs: &Self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Min(*op, *rhs));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn or(&self, rhs: &Self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Or(*op, *rhs));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn and(&self, rhs: &Self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(And(*op, *rhs));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn xor(&self, rhs: &Self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Xor(*op, *rhs));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn eq(&self, rhs: &Self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Eq(*op, *rhs));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn ne(&self, rhs: &Self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Ne(*op, *rhs));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn ge(&self, rhs: &Self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Ge(*op, *rhs));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn gt(&self, rhs: &Self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Gt(*op, *rhs));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn le(&self, rhs: &Self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Le(*op, *rhs));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn lt(&self, rhs: &Self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", rhs as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Lt(*op, *rhs));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn not(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Not(*op));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn exp(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Exp(*op));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn expm1(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Expm1(*op));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn floor(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Floor(*op));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn ceil(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Ceil(*op));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn round(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Round(*op));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn log(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Log(*op));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn log1p(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Log1p(*op));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn logistic(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Logistic(*op));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn sign(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Sign(*op));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn clz(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Clz(*op));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn cos(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Cos(*op));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn sin(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Sin(*op));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn tanh(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Tanh(*op));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn real(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Real(*op));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn imag(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Imag(*op));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn rsqrt(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Rsqrt(*op));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn cbrt(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Cbrt(*op));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn is_finite(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(IsFinite(*op));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn lower_triangle(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(LowerTriangle(*op));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn upper_triangle(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(UpperTriangle(*op));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    /*pub fn einsum1(&self, config: &str,) -> Self {
     * let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", config as "const char*"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(Einsum(*op, config));
            })
        }
    }

    pub fn einsum2(&self, arg2: &Self, config: &str) -> Self {
    let op = &self.raw;
        unsafe {
            cpp!([op as "const XlaOp*", arg2 as "const XlaOp*", config as "const char*"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(Einsum(*op, arg2, config));
            })
        }
    }*/

    pub fn clamp(&self, min: &Self, max: &Self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", min as "const XlaOp*", max as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Clamp(*op, *min, *max));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn copy(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Copy(*op));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn zeros_like(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(ZerosLike(*op));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn zero_like(&self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    const Shape *shape = op->builder()->GetShapePtr(*op).value();
                    return XlaOp(Zero(op->builder(), shape->element_type()));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn reshape(&self, ds: &[i64]) -> Self {
        let op = &self.raw;
        let ds_ptr = ds.as_ptr();
        let ds_len = ds.len();
        let raw = unsafe {
            cpp!([op as "const XlaOp*", ds_ptr as "const int64_t*", ds_len as "size_t"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Reshape(*op, absl::Span(ds_ptr, ds_len)));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn broadcast(&self, ds: &[i64]) -> Self {
        let op = &self.raw;
        let ds_ptr = ds.as_ptr();
        let ds_len = ds.len();
        let raw = unsafe {
            cpp!([op as "const XlaOp*", ds_ptr as "const int64_t*", ds_len as "size_t"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Broadcast(*op, absl::Span(ds_ptr, ds_len)));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn broadcast_in_dim(&self, dims: &[i64], broadcast_dims: &[i64]) -> Self {
        let op = &self.raw;
        let dims_ptr = dims.as_ptr();
        let dims_len = dims.len();
        let broadcast_dims_ptr = broadcast_dims.as_ptr();
        let broadcast_dims_len = broadcast_dims.len();
        let raw = unsafe {
            cpp!([op as "const XlaOp*", dims_ptr as "const int64_t*", dims_len as "size_t", broadcast_dims_ptr as "const int64_t*", broadcast_dims_len as "int64_t"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(BroadcastInDim(*op, absl::Span(dims_ptr, dims_len), absl::Span(broadcast_dims_ptr, broadcast_dims_len)));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn collapse(&self, ds: &[i64]) -> Self {
        let op = &self.raw;
        let ds_ptr = ds.as_ptr();
        let ds_len = ds.len();
        let raw = unsafe {
            cpp!([op as "const XlaOp*", ds_ptr as "const int64_t*", ds_len as "size_t"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Collapse(*op, absl::Span(ds_ptr, ds_len)));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn transpose(&self, dims: &[i64]) -> Self {
        let op = &self.raw;
        let dims_ptr = dims.as_ptr();
        let dims_len = dims.len();
        let raw = unsafe {
            cpp!([op as "const XlaOp*", dims_ptr as "const int64_t*", dims_len as "size_t"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Transpose(*op, absl::Span(dims_ptr, dims_len)));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn select(&self, on_true: &Self, on_false: &Self) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", on_true as "const XlaOp*", on_false as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Select(*op, *on_true, *on_false));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn rng_uniform(&self, sigma: &Self, shape: &ArrayShape) -> Self {
        let op = &self.raw;
        let dims = shape.dims();
        let dims_ptr = dims.as_ptr();
        let dims_len = dims.len();
        let prim_type = shape.primitive_type() as i32;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", sigma as "const XlaOp*", dims_ptr as "const int64_t*", dims_len as "size_t", prim_type as "int32_t"] -> XlaOpRaw as "XlaOp" {
                try {
                    auto shape = ShapeUtil::MakeShape((PrimitiveType)prim_type, absl::Span(dims_ptr, dims_len));
                    return XlaOp(RngUniform(*op, *sigma, shape));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn rng_normal(&self, sigma: &Self, shape: &ArrayShape) -> Self {
        let op = &self.raw;
        let dims = shape.dims();
        let dims_ptr = dims.as_ptr();
        let dims_len = dims.len();
        let prim_type = shape.primitive_type() as i32;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", sigma as "const XlaOp*", dims_ptr as "const int64_t*", dims_len as "size_t", prim_type as "int32_t"] -> XlaOpRaw as "XlaOp" {
                try {
                    auto shape = ShapeUtil::MakeShape((PrimitiveType)prim_type, absl::Span(dims_ptr, dims_len));
                    return XlaOp(RngNormal(*op, *sigma, shape));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn slice(&self, start_indices: &[i64], limit_indices: &[i64], strides: &[i64]) -> Self {
        let op = &self.raw;
        let start_indices_ptr = start_indices.as_ptr();
        let start_indices_len = start_indices.len();
        let limit_indices_ptr = limit_indices.as_ptr();
        let limit_indices_len = limit_indices.len();
        let strides_ptr = strides.as_ptr();
        let strides_len = strides.len();
        let raw = unsafe {
            cpp!([op as "const XlaOp*", start_indices_ptr as "const int64_t*", start_indices_len as "size_t", limit_indices_ptr as "const int64_t*", limit_indices_len as "size_t", strides_ptr as "const int64_t*", strides_len as "size_t"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Slice(*op, absl::Span(start_indices_ptr, start_indices_len), absl::Span(limit_indices_ptr, limit_indices_len), absl::Span(strides_ptr, strides_len)));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn slice_in_dim(&self, start_index: i64, limit_index: i64, stride: i64, dim: i64) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", start_index as "int64_t", limit_index as "int64_t", dim as "int64_t", stride as "int64_t"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(SliceInDim(*op, start_index, limit_index, stride, dim));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn dynamic_slice(&self, start_indices: &[XlaOpRef<'_>], size_indices: &[i64]) -> Self {
        let op = &self.raw;
        let start_indices_ptr = start_indices.as_ptr();
        let start_indices_len = start_indices.len();
        let size_indices_ptr = size_indices.as_ptr();
        let size_indices_len = size_indices.len();
        let raw = unsafe {
            cpp!([op as "const XlaOp*", start_indices_ptr as "const XlaOp*", start_indices_len as "size_t", size_indices_ptr as "const int64_t*", size_indices_len as "size_t"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(DynamicSlice(*op, absl::Span(start_indices_ptr, start_indices_len), absl::Span(size_indices_ptr, size_indices_len)));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn dynamic_update_slice(&self, update: &XlaOp, start_indices: &[XlaOpRef<'_>]) -> Self {
        let op = &self.raw;
        let start_indices_ptr = start_indices.as_ptr();
        let start_indices_len = start_indices.len();
        let raw = unsafe {
            cpp!([op as "const XlaOp*", update as "const XlaOp*", start_indices_ptr as "const XlaOp*", start_indices_len as "size_t"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(DynamicUpdateSlice(*op, *update, absl::Span(start_indices_ptr, start_indices_len)));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn get_tuple_element(&self, index: i64) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", index as "int64_t"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(GetTupleElement(*op, index));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn gather(
        &self,
        indices: &Self,
        offset_dims: &[i64],
        collapsed_slice_dims: &[i64],
        start_index_map: &[i64],
        slice_sizes: &[i64],
        index_vector_dim: i64,
    ) -> Self {
        let op = &self.raw;
        let offset_dims_ptr = offset_dims.as_ptr();
        let offset_dims_len = offset_dims.len();
        let slice_dims_ptr = collapsed_slice_dims.as_ptr();
        let slice_dims_len = collapsed_slice_dims.len();
        let start_index_map_ptr = start_index_map.as_ptr();
        let start_index_map_len = start_index_map.len();
        let slice_sizes_ptr = slice_sizes.as_ptr();
        let slice_sizes_len = slice_sizes.len();
        let raw = unsafe {
            cpp!([
                op as "const XlaOp*",
                indices as "const XlaOp*",
                offset_dims_ptr as "const int64_t*",
                offset_dims_len as "size_t",
                slice_dims_ptr as "const int64_t*",
                slice_dims_len as "size_t",
                start_index_map_ptr as "const int64_t*",
                start_index_map_len as "size_t",
                slice_sizes_ptr as "const int64_t*",
                slice_sizes_len as "size_t",
                index_vector_dim as "int64_t"
            ] -> XlaOpRaw as "XlaOp" {
                    GatherDimensionNumbers dn;
                    for (size_t i = 0; i < offset_dims_len; ++i) {
                        dn.add_offset_dims(offset_dims_ptr[i]);
                    }
                    for (size_t i = 0; i < slice_dims_len; ++i) {
                        dn.add_collapsed_slice_dims(slice_dims_ptr[i]);
                    }
                    for (size_t i = 0; i < start_index_map_len; ++i) {
                        dn.add_start_index_map(start_index_map_ptr[i]);
                    }
                    dn.set_index_vector_dim(index_vector_dim);
                    auto ss = absl::Span<const int64_t>(slice_sizes_ptr, slice_sizes_len);
                    return XlaOp(Gather(*op, *indices, dn, ss));
            })
        };
        self.wrap(raw)
    }

    pub fn convert_element_type(&self, ty: PrimitiveType) -> Self {
        let op = &self.raw;
        let ty = ty as i32;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", ty as "int32_t"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(ConvertElementType(*op, (PrimitiveType)ty));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn get_dimension_size(&self, dim: i64) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", dim as "int64_t"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(GetDimensionSize(*op, dim));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn reduce(&self, init_value: &Self, comp: &XlaComputation, dims: &[i64]) -> Self {
        let op = &self.raw;
        let dims_ptr = dims.as_ptr();
        let dims_len = dims.len();
        let raw = unsafe {
            cpp!([op as "const XlaOp*", init_value as "const XlaOp*", comp as "const XlaComputation*", dims_ptr as "const int64_t*", dims_len as "size_t"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Reduce(*op, *init_value, *comp, absl::Span(dims_ptr, dims_len)));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    pub fn conditional(
        &self,
        true_op: &Self,
        on_true: &XlaComputation,
        false_op: &Self,
        on_false: &XlaComputation,
    ) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", true_op as "const XlaOp*", on_true as "const XlaComputation*", false_op as "const XlaOp*", on_false as "const XlaComputation*"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Conditional(*op, *true_op, *on_true, *false_op, *on_false));
                }catch(std::exception& e) {
                    return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        self.wrap(raw)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn scatter(
        &self,
        inputs: &[XlaOpRef<'_>],
        scatter_indices: &XlaOp,
        updates: &[XlaOpRef<'_>],
        update_comp: &XlaComputation,
        scatter_dimension_numbers: ScatterDimensionNumbers,
        indices_are_sorted: bool,
        unique_indices: bool,
    ) -> XlaOp {
        let inputs_ptr = inputs.as_ptr();
        let inputs_len = inputs.len();
        let updates_ptr = updates.as_ptr();
        let updates_len = updates.len();
        let scatter_indices = &scatter_indices.raw;
        let raw = unsafe {
            cpp!([
                inputs_ptr as "const XlaOp*", inputs_len as "size_t",
                scatter_indices as "XlaOp*",
                update_comp as "const XlaComputation*",
                updates_ptr as "const XlaOp*", updates_len as "size_t",
                scatter_dimension_numbers as "std::unique_ptr<ScatterDimensionNumbers>",
                indices_are_sorted as "bool",
                unique_indices as "bool"
            ] -> XlaOpRaw as "XlaOp" {
                return XlaOp(
                    Scatter(
                        absl::Span(inputs_ptr, inputs_len),
                        *scatter_indices,
                        absl::Span(updates_ptr, updates_len),
                        *update_comp,
                        *scatter_dimension_numbers.get(),
                        indices_are_sorted,
                        unique_indices
                    )
                );
            })
        };
        XlaOp {
            raw,
            builder: self.builder.clone(),
        }
    }

    pub fn cholesky(&self, lower: bool) -> Self {
        let op = &self.raw;
        let raw = unsafe {
            cpp!([op as "const XlaOp*", lower as "bool"] -> XlaOpRaw as "XlaOp" {
                    try {
                        return XlaOp(Cholesky(*op, lower));
                    }catch(std::exception& e) {
                        return XlaOp(op->builder()->ReportError(tsl::errors::Internal(e.what())));
                    }
                }
            )
        };
        self.wrap(raw)
    }

    pub fn builder(&self) -> &XlaBuilder {
        &self.builder
    }
}

macro_rules! bin_op_impl {
    ($trait:ident, $op:tt) => {
        impl $trait for XlaOp {
            type Output = XlaOp;
            fn $op(self, rhs: Self) -> Self {
                XlaOp::$op(&self, &rhs)
            }
        }

        impl<'a> $trait<&'a Self> for &'a XlaOp {
            type Output = XlaOp;
            fn $op(self, rhs: &'a Self) -> XlaOp {
                XlaOp::$op(self, rhs)
            }
        }
    };
}

bin_op_impl!(Add, add);
bin_op_impl!(Sub, sub);
bin_op_impl!(Mul, mul);
bin_op_impl!(Div, div);

impl DotDimensionNumbers {
    pub fn new() -> DotDimensionNumbers {
        unsafe {
            cpp!([] -> DotDimensionNumbers as "DotDimensionNumbers" {
                return DotDimensionNumbers();
            })
        }
    }

    pub fn add_lhs_contracting_dimensions(&mut self, dim: i64) {
        unsafe {
            cpp!([self as "DotDimensionNumbers*", dim as "int64_t"] {
                self->add_lhs_contracting_dimensions(dim);
            })
        }
    }

    pub fn add_rhs_contracting_dimensions(&mut self, dim: i64) {
        unsafe {
            cpp!([self as "DotDimensionNumbers*", dim as "int64_t"] {
                self->add_rhs_contracting_dimensions(dim);
            })
        }
    }

    pub fn add_lhs_batch_dimensions(&mut self, dim: i64) {
        unsafe {
            cpp!([self as "DotDimensionNumbers*", dim as "int64_t"] {
                self->add_lhs_batch_dimensions(dim);
            })
        }
    }

    pub fn add_rhs_batch_dimensions(&mut self, dim: i64) {
        unsafe {
            cpp!([self as "DotDimensionNumbers*", dim as "int64_t"] {
                self->add_rhs_batch_dimensions(dim);
            })
        }
    }
}

impl ScatterDimensionNumbers {
    pub fn new() -> ScatterDimensionNumbers {
        unsafe {
            cpp!([] -> ScatterDimensionNumbers as "std::unique_ptr<ScatterDimensionNumbers>" {
                return std::make_unique<ScatterDimensionNumbers>();
            })
        }
    }

    pub fn add_window_dim(&mut self, dim: i64) {
        unsafe {
            cpp!([self as "std::unique_ptr<ScatterDimensionNumbers>*", dim as "int64_t"] {
                self->get()->add_update_window_dims(dim);
            })
        }
    }

    pub fn add_inserted_window_dim(&mut self, dim: i64) {
        unsafe {
            cpp!([self as "std::unique_ptr<ScatterDimensionNumbers>*", dim as "int64_t"] {
                self->get()->add_inserted_window_dims(dim);
            })
        }
    }

    pub fn add_scatter_dims_to_operand_dims(&mut self, dim: i64) {
        unsafe {
            cpp!([self as "std::unique_ptr<ScatterDimensionNumbers>*", dim as "int64_t"] {
                self->get()->add_scatter_dims_to_operand_dims(dim);
            })
        }
    }

    pub fn set_index_vector_dim(&mut self, dim: i64) {
        unsafe {
            cpp!([self as "std::unique_ptr<ScatterDimensionNumbers>*", dim as "int64_t"] {
                self->get()->set_index_vector_dim(dim);
            })
        }
    }
}
