use crate::{Literal, RawShape, Result, Status};

use cpp::{cpp, cpp_class};

use std::{marker::PhantomData, mem::ManuallyDrop, pin::Pin};

cpp! {{
    #include "xla/client/xla_builder.h"
    #include "xla/client/lib/constants.h"
    #include "xla/client/lib/matrix.h"
    #include "xla/statusor.h"
    #include "xla/literal_util.h"
    using namespace xla;
}}

cpp_class!(pub unsafe struct PjRtBuffer as "std::unique_ptr<PjRtBuffer>");
cpp_class!(pub unsafe struct BufferArgsInner as "std::unique_ptr<std::vector<PjRtBuffer*>>");
cpp_class!(pub unsafe struct BufferArgsInnerRaw as "std::vector<PjRtBuffer*>");

impl PjRtBuffer {
    pub(crate) fn is_null(&self) -> bool {
        unsafe {
            cpp!([self as "const std::shared_ptr<PjRtBuffer>*"] -> bool as "bool" {
                return self == nullptr;
            })
        }
    }

    pub fn is_on_cpu(&self) -> bool {
        unsafe {
            cpp!([self as "const std::shared_ptr<PjRtBuffer>*"] -> bool as "bool" {
                return (*self)->IsOnCpu();
            })
        }
    }

    pub fn shape(&self) -> RawShape {
        unsafe {
            cpp!([self as "std::unique_ptr<PjRtBuffer>*"] -> RawShape as "xla::Shape" {
                auto device_shape = (*self)->on_device_shape();
                auto host_shape = ShapeUtil::DeviceShapeToHostShape(device_shape);
                return host_shape;
            })
        }
    }

    pub(crate) fn copy_into(&self, dst: &mut Vec<u8>) -> Result<()> {
        let shape = self.shape();
        let len = shape.size();
        dst.clear();
        dst.reserve_exact(len);
        let dst_ptr = dst.as_mut_ptr();

        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        unsafe {
            cpp!([self as "std::unique_ptr<PjRtBuffer>*", dst_ptr as "char*", shape as "xla::Shape", out_status as "Status*"] {
                auto literal = std::make_unique<xla::MutableBorrowingLiteral>(dst_ptr, shape);
                *out_status = (*self)->ToLiteralSync(literal.get());
            });
            out_status.to_result()?;
            dst.set_len(len);
        }
        Ok(())
    }

    pub fn to_literal_sync(&self) -> Result<Literal> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let lit = unsafe {
            cpp!([self as "std::unique_ptr<PjRtBuffer>*", out_status as "Status*"] -> Literal as "std::shared_ptr<Literal>" {
                auto status = (*self)->ToLiteralSync();
                if (status.ok()) {
                    return std::move(status.value());
                }else{
                    *out_status = Status(status.status());
                    return std::make_shared<Literal>(Literal());
                }
            })
        };
        out_status.to_result()?;
        Ok(lit)
    }
}

pub struct BufferArgsRef<'a> {
    phantom_data: PhantomData<&'a ()>,
    pub(crate) untuple_result: bool,
    pub(crate) buffers: BufferArgsInner,
}

impl<'a> Default for BufferArgsRef<'a> {
    fn default() -> Self {
        Self {
            phantom_data: Default::default(),
            buffers: unsafe {
                cpp!([] -> BufferArgsInner as "std::unique_ptr<std::vector<PjRtBuffer*>>" {
                    std::unique_ptr<std::vector<PjRtBuffer*>> vec (new std::vector<PjRtBuffer*> {});
                    return vec;
                })
            },
            untuple_result: false,
        }
    }
}

impl<'a> BufferArgsRef<'a> {
    pub fn push(&mut self, buf: &'a PjRtBuffer) {
        let inner = &mut self.buffers;
        let buf = buf as *const PjRtBuffer;
        unsafe {
            cpp!([inner as "std::unique_ptr<std::vector<PjRtBuffer*>>*", buf as "std::unique_ptr<PjRtBuffer>*"] {
                auto buf_ptr = buf->get();
                (*inner)->push_back(buf_ptr);
            })
        };
    }

    pub fn untuple_result(mut self, untuple_result: bool) -> Self {
        self.untuple_result = untuple_result;
        self
    }
}

impl<'a> FromIterator<&'a PjRtBuffer> for BufferArgsRef<'a> {
    fn from_iter<T: IntoIterator<Item = &'a PjRtBuffer>>(iter: T) -> Self {
        let mut args = BufferArgsRef::default();
        for buf in iter {
            args.push(buf);
        }
        args
    }
}

impl<'a, const N: usize> From<[&'a PjRtBuffer; N]> for BufferArgsRef<'a> {
    fn from(value: [&'a PjRtBuffer; N]) -> Self {
        value.into_iter().collect()
    }
}

pub trait BufferArgs {
    fn get(&self) -> &'_ BufferArgsInnerRaw;
    fn untuple_result(&self) -> bool;
}

impl BufferArgs for BufferArgsRef<'_> {
    fn get(&self) -> &'_ BufferArgsInnerRaw {
        let inner = &self.buffers;
        unsafe {
            let ptr = cpp!([inner as "std::unique_ptr<std::vector<PjRtBuffer*>>*"] -> *const BufferArgsInnerRaw as "std::vector<PjRtBuffer*>*" {
                return inner->get();
            });
            &*ptr
        }
    }

    fn untuple_result(&self) -> bool {
        self.untuple_result
    }
}

pub struct BufferArgsOwned {
    pub(crate) untuple_result: bool,
    pub(crate) buffers: BufferArgsInner,
}

impl BufferArgsOwned {
    pub fn push(&mut self, buf: PjRtBuffer) {
        let inner = &mut self.buffers;
        let mut buf = ManuallyDrop::new(buf);
        unsafe {
            cpp!([inner as "std::unique_ptr<std::vector<PjRtBuffer*>>*", mut buf as "std::unique_ptr<PjRtBuffer>"] {
                (*inner)->push_back(buf.release());
            })
        };
    }
}

impl BufferArgs for BufferArgsOwned {
    fn get(&self) -> &'_ BufferArgsInnerRaw {
        let inner = &self.buffers;
        unsafe {
            let ptr = cpp!([inner as "std::unique_ptr<std::vector<PjRtBuffer*>>*"] -> *const BufferArgsInnerRaw as "std::vector<PjRtBuffer*>*" {
                return inner->get();
            });
            &*ptr
        }
    }

    fn untuple_result(&self) -> bool {
        self.untuple_result
    }
}

impl Default for BufferArgsOwned {
    fn default() -> Self {
        Self {
            buffers: unsafe {
                cpp!([] -> BufferArgsInner as "std::unique_ptr<std::vector<PjRtBuffer*>>" {
                    std::unique_ptr<std::vector<PjRtBuffer*>> vec (new std::vector<PjRtBuffer*> {});
                    return vec;
                })
            },
            untuple_result: false,
        }
    }
}

impl Drop for BufferArgsOwned {
    fn drop(&mut self) {
        let inner = &mut self.buffers;
        unsafe {
            cpp!([inner as "std::unique_ptr<std::vector<PjRtBuffer*>>*"] {
                for(auto ptr : (*inner->get())) {
                    delete ptr;
                }
            })
        }
    }
}

impl FromIterator<PjRtBuffer> for BufferArgsOwned {
    fn from_iter<T: IntoIterator<Item = PjRtBuffer>>(iter: T) -> Self {
        let mut args = BufferArgsOwned::default();
        for buf in iter {
            args.push(buf);
        }
        args
    }
}

impl<A: BufferArgs> BufferArgs for &'_ A {
    fn get(&self) -> &'_ BufferArgsInnerRaw {
        A::get(*self)
    }

    fn untuple_result(&self) -> bool {
        A::untuple_result(*self)
    }
}
