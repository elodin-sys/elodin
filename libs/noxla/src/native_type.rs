use crate::{Literal, XlaBuilder, XlaOp, XlaOpRaw};
use cpp::cpp;
use zerocopy::{FromBytes, Immutable};

cpp! {{
    #include "xla/client/xla_builder.h"
    #include "xla/client/lib/constants.h"
    #include "xla/client/lib/matrix.h"
    #include "xla/statusor.h"
    #include "xla/literal_util.h"
    #include "xla/pjrt/pjrt_api.h"
    #include "xla/pjrt/pjrt_c_api_client.h"
    #include "xla/pjrt/pjrt_client.h"
    using namespace xla;
}}

/// A type implementing the `NativeType` trait can be directly converted to constant ops or
/// literals.
pub trait NativeType: Copy + FromBytes + Immutable {
    fn constant_r0(builder: &XlaBuilder, value: Self) -> XlaOp;
    fn constant_r1(builder: &XlaBuilder, value: &[Self]) -> XlaOp;
    fn literal(self) -> Literal;
    fn create_r1(slice: &[Self]) -> Literal;
}

impl NativeType for f64 {
    fn constant_r0(builder: &XlaBuilder, value: Self) -> XlaOp {
        let raw = unsafe {
            cpp!([builder as "std::shared_ptr<XlaBuilder>*", value as "double"] ->XlaOpRaw as "XlaOp" {
                return XlaOp(ConstantR0<double>(builder->get(), value));
            })
        };
        XlaOp {
            raw,
            builder: builder.clone(),
        }
    }

    fn constant_r1(builder: &XlaBuilder, value: &[Self]) -> XlaOp {
        let value_ptr = value.as_ptr();
        let value_len = value.len();
        let raw = unsafe {
            cpp!([builder as "std::shared_ptr<XlaBuilder>*", value_ptr as "const double*", value_len as "size_t"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(ConstantR1<double>(builder->get(), absl::Span<const double>(value_ptr, value_len)));
            })
        };
        XlaOp {
            raw,
            builder: builder.clone(),
        }
    }

    fn literal(self) -> Literal {
        unsafe {
            cpp!([self as "double"] -> Literal as "std::shared_ptr<Literal>" {
                return std::make_shared<Literal>(LiteralUtil::CreateR0<double>(self));
            })
        }
    }

    fn create_r1(slice: &[Self]) -> Literal {
        let value_ptr = slice.as_ptr();
        let value_len = slice.len();
        unsafe {
            cpp!([value_ptr as "const double*", value_len as "size_t"] -> Literal as "std::shared_ptr<Literal>" {
                return std::make_shared<Literal>(LiteralUtil::CreateR1<double>(absl::Span<const double>(value_ptr, value_len)));
            })
        }
    }
}

impl NativeType for f32 {
    fn constant_r0(builder: &XlaBuilder, value: Self) -> XlaOp {
        let raw = unsafe {
            cpp!([builder as "std::shared_ptr<XlaBuilder>*", value as "float"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(ConstantR0<float>(builder->get(), value));
            })
        };
        XlaOp {
            raw,
            builder: builder.clone(),
        }
    }

    fn constant_r1(builder: &XlaBuilder, value: &[Self]) -> XlaOp {
        let value_ptr = value.as_ptr();
        let value_len = value.len();
        let raw = unsafe {
            cpp!([builder as "std::shared_ptr<XlaBuilder>*", value_ptr as "const float*", value_len as "size_t"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(ConstantR1<float>(builder->get(), absl::Span<const float>(value_ptr, value_len)));
            })
        };
        XlaOp {
            raw,
            builder: builder.clone(),
        }
    }

    fn literal(self) -> Literal {
        unsafe {
            cpp!([self as "float"] -> Literal as "std::shared_ptr<Literal>" {
                return std::make_shared<Literal>(LiteralUtil::CreateR0<float>(self));
            })
        }
    }

    fn create_r1(slice: &[Self]) -> Literal {
        let value_ptr = slice.as_ptr();
        let value_len = slice.len();
        unsafe {
            cpp!([value_ptr as "const float*", value_len as "size_t"] -> Literal as "std::shared_ptr<Literal>" {
                return std::make_shared<Literal>(LiteralUtil::CreateR1<float>(absl::Span<const float>(value_ptr, value_len)));
            })
        }
    }
}

impl NativeType for u64 {
    fn constant_r0(builder: &XlaBuilder, value: Self) -> XlaOp {
        let raw = unsafe {
            cpp!([builder as "std::shared_ptr<XlaBuilder>*", value as "uint64_t"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(ConstantR0<uint64_t>(builder->get(), value));
            })
        };
        XlaOp {
            raw,
            builder: builder.clone(),
        }
    }

    fn constant_r1(builder: &XlaBuilder, value: &[Self]) -> XlaOp {
        let value_ptr = value.as_ptr();
        let value_len = value.len();
        let raw = unsafe {
            cpp!([builder as "std::shared_ptr<XlaBuilder>*", value_ptr as "const uint64_t*", value_len as "size_t"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(ConstantR1<uint64_t>(builder->get(), absl::Span<const uint64_t>(value_ptr, value_len)));
            })
        };
        XlaOp {
            raw,
            builder: builder.clone(),
        }
    }

    fn literal(self) -> Literal {
        unsafe {
            cpp!([self as "uint64_t"] -> Literal as "std::shared_ptr<Literal>" {
                return std::make_shared<Literal>(LiteralUtil::CreateR0<uint64_t>(self));
            })
        }
    }

    fn create_r1(slice: &[Self]) -> Literal {
        let value_ptr = slice.as_ptr();
        let value_len = slice.len();
        unsafe {
            cpp!([value_ptr as "const uint64_t*", value_len as "size_t"] -> Literal as "std::shared_ptr<Literal>" {
                return std::make_shared<Literal>(LiteralUtil::CreateR1<uint64_t>(absl::Span<const uint64_t>(value_ptr, value_len)));
            })
        }
    }
}

impl NativeType for u32 {
    fn constant_r0(builder: &XlaBuilder, value: Self) -> XlaOp {
        let raw = unsafe {
            cpp!([builder as "std::shared_ptr<XlaBuilder>*", value as "uint32_t"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(ConstantR0<uint32_t>(builder->get(), value));
            })
        };
        XlaOp {
            raw,
            builder: builder.clone(),
        }
    }

    fn constant_r1(builder: &XlaBuilder, value: &[Self]) -> XlaOp {
        let value_ptr = value.as_ptr();
        let value_len = value.len();
        let raw = unsafe {
            cpp!([builder as "std::shared_ptr<XlaBuilder>*", value_ptr as "const uint32_t*", value_len as "size_t"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(ConstantR1<uint32_t>(builder->get(), absl::Span<const uint32_t>(value_ptr, value_len)));
            })
        };
        XlaOp {
            raw,
            builder: builder.clone(),
        }
    }

    fn literal(self) -> Literal {
        unsafe {
            cpp!([self as "uint32_t"] -> Literal as "std::shared_ptr<Literal>" {
                return std::make_shared<Literal>(LiteralUtil::CreateR0<uint32_t>(self));
            })
        }
    }

    fn create_r1(slice: &[Self]) -> Literal {
        let value_ptr = slice.as_ptr();
        let value_len = slice.len();
        unsafe {
            cpp!([value_ptr as "const uint32_t*", value_len as "size_t"] -> Literal as "std::shared_ptr<Literal>" {
                return std::make_shared<Literal>(LiteralUtil::CreateR1<uint32_t>(absl::Span<const uint32_t>(value_ptr, value_len)));
            })
        }
    }
}

impl NativeType for u16 {
    fn constant_r0(builder: &XlaBuilder, value: Self) -> XlaOp {
        let raw = unsafe {
            cpp!([builder as "std::shared_ptr<XlaBuilder>*", value as "uint16_t"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(ConstantR0<uint16_t>(builder->get(), value));
            })
        };
        XlaOp {
            raw,
            builder: builder.clone(),
        }
    }

    fn constant_r1(builder: &XlaBuilder, value: &[Self]) -> XlaOp {
        let value_ptr = value.as_ptr();
        let value_len = value.len();
        let raw = unsafe {
            cpp!([builder as "std::shared_ptr<XlaBuilder>*", value_ptr as "const uint16_t*", value_len as "size_t"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(ConstantR1<uint16_t>(builder->get(), absl::Span<const uint16_t>(value_ptr, value_len)));
            })
        };
        XlaOp {
            raw,
            builder: builder.clone(),
        }
    }

    fn literal(self) -> Literal {
        unsafe {
            cpp!([self as "uint16_t"] -> Literal as "std::shared_ptr<Literal>" {
                return std::make_shared<Literal>(LiteralUtil::CreateR0<uint16_t>(self));
            })
        }
    }

    fn create_r1(slice: &[Self]) -> Literal {
        let value_ptr = slice.as_ptr();
        let value_len = slice.len();
        unsafe {
            cpp!([value_ptr as "const uint16_t*", value_len as "size_t"] -> Literal as "std::shared_ptr<Literal>" {
                return std::make_shared<Literal>(LiteralUtil::CreateR1<uint16_t>(absl::Span<const uint16_t>(value_ptr, value_len)));
            })
        }
    }
}

impl NativeType for i64 {
    fn constant_r0(builder: &XlaBuilder, value: Self) -> XlaOp {
        let raw = unsafe {
            cpp!([builder as "std::shared_ptr<XlaBuilder>*", value as "int64_t"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(ConstantR0<int64_t>(builder->get(), value));
            })
        };
        XlaOp {
            raw,
            builder: builder.clone(),
        }
    }

    fn constant_r1(builder: &XlaBuilder, value: &[Self]) -> XlaOp {
        let value_ptr = value.as_ptr();
        let value_len = value.len();
        let raw = unsafe {
            cpp!([builder as "std::shared_ptr<XlaBuilder>*", value_ptr as "const int64_t*", value_len as "size_t"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(ConstantR1<int64_t>(builder->get(), absl::Span<const int64_t>(value_ptr, value_len)));
            })
        };
        XlaOp {
            raw,
            builder: builder.clone(),
        }
    }

    fn literal(self) -> Literal {
        unsafe {
            cpp!([self as "int64_t"] -> Literal as "std::shared_ptr<Literal>" {
                return std::make_shared<Literal>(LiteralUtil::CreateR0<int64_t>(self));
            })
        }
    }

    fn create_r1(slice: &[Self]) -> Literal {
        let value_ptr = slice.as_ptr();
        let value_len = slice.len();
        unsafe {
            cpp!([value_ptr as "const int64_t*", value_len as "size_t"] -> Literal as "std::shared_ptr<Literal>" {
                return std::make_shared<Literal>(LiteralUtil::CreateR1<int64_t>(absl::Span<const int64_t>(value_ptr, value_len)));
            })
        }
    }
}

impl NativeType for i32 {
    fn constant_r0(builder: &XlaBuilder, value: Self) -> XlaOp {
        let raw = unsafe {
            cpp!([builder as "std::shared_ptr<XlaBuilder>*", value as "int32_t"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(ConstantR0<int32_t>(builder->get(), value));
            })
        };
        XlaOp {
            raw,
            builder: builder.clone(),
        }
    }

    fn constant_r1(builder: &XlaBuilder, value: &[Self]) -> XlaOp {
        let value_ptr = value.as_ptr();
        let value_len = value.len();
        let raw = unsafe {
            cpp!([builder as "std::shared_ptr<XlaBuilder>*", value_ptr as "const int32_t*", value_len as "size_t"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(ConstantR1<int32_t>(builder->get(), absl::Span<const int32_t>(value_ptr, value_len)));
            })
        };
        XlaOp {
            raw,
            builder: builder.clone(),
        }
    }

    fn literal(self) -> Literal {
        unsafe {
            cpp!([self as "int32_t"] -> Literal as "std::shared_ptr<Literal>" {
                return std::make_shared<Literal>(LiteralUtil::CreateR0<int32_t>(self));
            })
        }
    }

    fn create_r1(slice: &[Self]) -> Literal {
        let value_ptr = slice.as_ptr();
        let value_len = slice.len();
        unsafe {
            cpp!([value_ptr as "const int32_t*", value_len as "size_t"] -> Literal as "std::shared_ptr<Literal>" {
                return std::make_shared<Literal>(LiteralUtil::CreateR1<int32_t>(absl::Span<const int32_t>(value_ptr, value_len)));
            })
        }
    }
}

impl NativeType for i16 {
    fn constant_r0(builder: &XlaBuilder, value: Self) -> XlaOp {
        let raw = unsafe {
            cpp!([builder as "std::shared_ptr<XlaBuilder>*", value as "int16_t"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(ConstantR0<int16_t>(builder->get(), value));
            })
        };
        XlaOp {
            raw,
            builder: builder.clone(),
        }
    }

    fn constant_r1(builder: &XlaBuilder, value: &[Self]) -> XlaOp {
        let value_ptr = value.as_ptr();
        let value_len = value.len();
        let raw = unsafe {
            cpp!([builder as "std::shared_ptr<XlaBuilder>*", value_ptr as "const int16_t*", value_len as "size_t"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(ConstantR1<int16_t>(builder->get(), absl::Span<const int16_t>(value_ptr, value_len)));
            })
        };
        XlaOp {
            raw,
            builder: builder.clone(),
        }
    }

    fn literal(self) -> Literal {
        unsafe {
            cpp!([self as "int16_t"] -> Literal as "std::shared_ptr<Literal>" {
                return std::make_shared<Literal>(LiteralUtil::CreateR0<int16_t>(self));
            })
        }
    }

    fn create_r1(slice: &[Self]) -> Literal {
        let value_ptr = slice.as_ptr();
        let value_len = slice.len();
        unsafe {
            cpp!([value_ptr as "const int16_t*", value_len as "size_t"] -> Literal as "std::shared_ptr<Literal>" {
                return std::make_shared<Literal>(LiteralUtil::CreateR1<int16_t>(absl::Span<const int16_t>(value_ptr, value_len)));
            })
        }
    }
}
