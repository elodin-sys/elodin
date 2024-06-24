use crate::{
    ElementType, Literal, NativeType, Result, Shape, Status, XlaComputation, XlaOp, XlaOpRaw,
    XlaOpRef,
};
use cpp::{cpp, cpp_class};
use cxx::let_cxx_string;
use std::pin::Pin;

cpp! {{
    #include "xla/client/xla_builder.h"
    using namespace xla;
}}

cpp_class!(pub unsafe struct XlaBuilder as "std::shared_ptr<XlaBuilder>");

impl XlaBuilder {
    pub fn new(name: &str) -> Self {
        let_cxx_string!(name = name);
        unsafe {
            cpp!( [name as "std::string*"] -> XlaBuilder as "std::shared_ptr<XlaBuilder>" {
                std::shared_ptr<XlaBuilder> builder(new XlaBuilder(*name));
                return builder;
            })
        }
    }

    pub fn build(&self, op: &XlaOp) -> Result<XlaComputation> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let comp = unsafe {
            cpp!([self as "std::shared_ptr<XlaBuilder>*", op as "XlaOp*", out_status as "Status*"] -> XlaComputation as "XlaComputation" {
                auto status = (*self)->Build(*op, false);
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

    pub fn concat_in_dim(&self, others: &[XlaOpRef<'_>], dim: i64) -> XlaOp {
        let others_ptr = others.as_ptr();
        let others_len = others.len();
        let raw = unsafe {
            cpp!([self as "std::shared_ptr<XlaBuilder>*", others_ptr as "const XlaOp*", others_len as "size_t", dim as "int64_t"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(ConcatInDim(self->get(), absl::Span(others_ptr, others_len), dim));
            })
        };
        XlaOp {
            raw,
            builder: self.clone(),
        }
    }

    pub fn tuple(&self, elems: &[XlaOpRef<'_>]) -> XlaOp {
        let elems_ptr = elems.as_ptr();
        let elems_len = elems.len();
        let raw = unsafe {
            cpp!([self as "std::shared_ptr<XlaBuilder>*", elems_ptr as "const XlaOp*", elems_len as "size_t"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(Tuple(self->get(), absl::Span(elems_ptr, elems_len)));
            })
        };
        XlaOp {
            raw,
            builder: self.clone(),
        }
    }

    pub fn map(&self, args: &[XlaOpRef<'_>], comp: &XlaComputation, dims: &[i64]) -> XlaOp {
        let args_ptr = args.as_ptr();
        let args_len = args.len();
        let dims_ptr = dims.as_ptr();
        let dims_len = dims.len();
        let raw = unsafe {
            cpp!([self as "std::shared_ptr<XlaBuilder>*", args_ptr as "const XlaOp*", args_len as "size_t", comp as "const XlaComputation*", dims_ptr as "const int64_t*", dims_len as "size_t"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(Map(self->get(), absl::Span(args_ptr, args_len), *comp, absl::Span(dims_ptr, dims_len)));
            })
        };
        XlaOp {
            raw,
            builder: self.clone(),
        }
    }

    pub fn parameter(&self, num: i64, shape: Shape, name: &str) -> Result<XlaOp> {
        let raw_shape = shape.raw_shape();
        let_cxx_string!(name = name);
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let op = unsafe {
            cpp!([self as "std::shared_ptr<XlaBuilder>*", num as "int64_t", name as "std::string*", raw_shape as "Shape"] -> XlaOpRaw as "XlaOp" {
                try {
                    return XlaOp(Parameter((self->get()), num, raw_shape, *name));
                }catch(std::exception& e) {
                    return XlaOp((*self)->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        out_status.to_result()?;
        Ok(XlaOp {
            raw: op,
            builder: self.clone(),
        })
    }

    /// Create a node with a constant value defined by the specified literal.
    pub fn constant_literal(&self, literal: &Literal) -> Result<XlaOp> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let op = unsafe {
            cpp!([self as "std::shared_ptr<XlaBuilder>*", literal as "std::shared_ptr<Literal>*"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(ConstantLiteral(self->get(), *literal->get()));
            })
        };
        out_status.to_result()?;
        Ok(XlaOp {
            raw: op,
            builder: self.clone(),
        })
    }

    pub fn constant<T: NativeType>(&self, val: T) -> XlaOp {
        T::constant_r0(self, val)
    }

    pub fn constant_vector<T: NativeType>(&self, vals: &[T]) -> XlaOp {
        T::constant_r1(self, vals)
    }

    pub fn setup_alias(&self, param_num: u64, output_index: u64) -> Result<()> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        unsafe {
            cpp!([self as "std::shared_ptr<XlaBuilder>*", param_num as "uint64_t", output_index as "uint64_t", out_status as "Status*"] {
                try {
                    (*self)->SetUpAlias({(int64_t) output_index}, (int64_t) param_num, {}, HloInputOutputAliasConfig::AliasKind::kMustAlias);
                }catch(std::exception& e) {
                    *out_status = Status(tsl::errors::Internal(e.what()));
                }
            })
        };
        out_status.to_result()
    }
    pub fn iota(&self, dims: &[i64], elem_type: ElementType, iota_dim: i64) -> XlaOp {
        let dims_ptr = dims.as_ptr();
        let dims_len = dims.len();
        let prim_type = elem_type.primitive_type() as i32;
        let raw = unsafe {
            cpp!([self as "std::shared_ptr<XlaBuilder>*", dims_ptr as "const int64_t*", dims_len as "size_t", prim_type as "int32_t", iota_dim as "int64_t"] -> XlaOpRaw as "XlaOp" {
                try {
                    auto shape = ShapeUtil::MakeShape((PrimitiveType)prim_type, absl::Span(dims_ptr, dims_len));
                    return XlaOp(Iota(self->get(), shape, iota_dim));
                }catch(std::exception& e) {
                    return XlaOp(self->get()->ReportError(tsl::errors::Internal(e.what())));
                }
            })
        };
        XlaOp {
            raw,
            builder: self.clone(),
        }
    }
}
