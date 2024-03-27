use crate::{
    ArrayElement, Error, Literal, PjRtBuffer, PjRtLoadedExecutable, Result, Status, XlaComputation,
};
use cpp::{cpp, cpp_class};
use std::pin::Pin;

cpp! {{
    #include "xla/pjrt/pjrt_api.h"
    #include "xla/pjrt/pjrt_c_api_client.h"
    #include "xla/pjrt/pjrt_client.h"
    #include "xla/pjrt/pjrt_stream_executor_client.h"
    #include "xla/pjrt/tfrt_cpu_pjrt_client.h"
    #include "xla/pjrt/gpu/gpu_helpers.h"
    #include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
    #include "jaxlib/cpu/lapack_kernels.h"
    #include "xla/service/custom_call_target_registry.h"
    using namespace xla;

    #pragma clang diagnostic ignored "-Wundefined-var-template"
    extern "C" {

        jax::Trsm<float>::FnType strsm_;
        jax::Trsm<double>::FnType dtrsm_;
        jax::Trsm<std::complex<float>>::FnType ctrsm_;
        jax::Trsm<std::complex<double>>::FnType ztrsm_;

        jax::Getrf<float>::FnType sgetrf_;
        jax::Getrf<double>::FnType dgetrf_;
        jax::Getrf<std::complex<float>>::FnType cgetrf_;
        jax::Getrf<std::complex<double>>::FnType zgetrf_;

        jax::Geqrf<float>::FnType sgeqrf_;
        jax::Geqrf<double>::FnType dgeqrf_;
        jax::Geqrf<std::complex<float>>::FnType cgeqrf_;
        jax::Geqrf<std::complex<double>>::FnType zgeqrf_;

        jax::Orgqr<float>::FnType sorgqr_;
        jax::Orgqr<double>::FnType dorgqr_;
        jax::Orgqr<std::complex<float>>::FnType cungqr_;
        jax::Orgqr<std::complex<double>>::FnType zungqr_;

        jax::Potrf<float>::FnType spotrf_;
        jax::Potrf<double>::FnType dpotrf_;
        jax::Potrf<std::complex<float>>::FnType cpotrf_;
        jax::Potrf<std::complex<double>>::FnType zpotrf_;

        jax::RealGesdd<float>::FnType sgesdd_;
        jax::RealGesdd<double>::FnType dgesdd_;
        jax::ComplexGesdd<std::complex<float>>::FnType cgesdd_;
        jax::ComplexGesdd<std::complex<double>>::FnType zgesdd_;

        jax::RealSyevd<float>::FnType ssyevd_;
        jax::RealSyevd<double>::FnType dsyevd_;
        jax::ComplexHeevd<std::complex<float>>::FnType cheevd_;
        jax::ComplexHeevd<std::complex<double>>::FnType zheevd_;

        jax::RealGeev<float>::FnType sgeev_;
        jax::RealGeev<double>::FnType dgeev_;
        jax::ComplexGeev<std::complex<float>>::FnType cgeev_;
        jax::ComplexGeev<std::complex<double>>::FnType zgeev_;

        jax::RealGees<float>::FnType sgees_;
        jax::RealGees<double>::FnType dgees_;
        jax::ComplexGees<std::complex<float>>::FnType cgees_;
        jax::ComplexGees<std::complex<double>>::FnType zgees_;

        jax::Gehrd<float>::FnType sgehrd_;
        jax::Gehrd<double>::FnType dgehrd_;
        jax::Gehrd<std::complex<float>>::FnType cgehrd_;
        jax::Gehrd<std::complex<double>>::FnType zgehrd_;

        jax::Sytrd<float>::FnType ssytrd_;
        jax::Sytrd<double>::FnType dsytrd_;
        jax::Sytrd<std::complex<float>>::FnType chetrd_;
        jax::Sytrd<std::complex<double>>::FnType zhetrd_;
    }

    namespace jax {
    static auto init = []() -> int {
        Trsm<float>::fn = strsm_;
        Trsm<double>::fn = dtrsm_;
        Trsm<std::complex<float>>::fn = ctrsm_;
        Trsm<std::complex<double>>::fn = ztrsm_;
        Getrf<float>::fn = sgetrf_;
        Getrf<double>::fn = dgetrf_;
        Getrf<std::complex<float>>::fn = cgetrf_;
        Getrf<std::complex<double>>::fn = zgetrf_;
        Geqrf<float>::fn = sgeqrf_;
        Geqrf<double>::fn = dgeqrf_;
        Geqrf<std::complex<float>>::fn = cgeqrf_;
        Geqrf<std::complex<double>>::fn = zgeqrf_;
        Orgqr<float>::fn = sorgqr_;
        Orgqr<double>::fn = dorgqr_;
        Orgqr<std::complex<float>>::fn = cungqr_;
        Orgqr<std::complex<double>>::fn = zungqr_;
        Potrf<float>::fn = spotrf_;
        Potrf<double>::fn = dpotrf_;
        Potrf<std::complex<float>>::fn = cpotrf_;
        Potrf<std::complex<double>>::fn = zpotrf_;
        RealGesdd<float>::fn = sgesdd_;
        RealGesdd<double>::fn = dgesdd_;
        ComplexGesdd<std::complex<float>>::fn = cgesdd_;
        ComplexGesdd<std::complex<double>>::fn = zgesdd_;
        RealSyevd<float>::fn = ssyevd_;
        RealSyevd<double>::fn = dsyevd_;
        ComplexHeevd<std::complex<float>>::fn = cheevd_;
        ComplexHeevd<std::complex<double>>::fn = zheevd_;
        RealGeev<float>::fn = sgeev_;
        RealGeev<double>::fn = dgeev_;
        ComplexGeev<std::complex<float>>::fn = cgeev_;
        ComplexGeev<std::complex<double>>::fn = zgeev_;
        RealGees<float>::fn = sgees_;
        RealGees<double>::fn = dgees_;
        ComplexGees<std::complex<float>>::fn = cgees_;
        ComplexGees<std::complex<double>>::fn = zgees_;
        Gehrd<float>::fn = sgehrd_;
        Gehrd<double>::fn = dgehrd_;
        Gehrd<std::complex<float>>::fn = cgehrd_;
        Gehrd<std::complex<double>>::fn = zgehrd_;
        Sytrd<float>::fn = ssytrd_;
        Sytrd<double>::fn = dsytrd_;
        Sytrd<std::complex<float>>::fn = chetrd_;
        Sytrd<std::complex<double>>::fn = zhetrd_;

        return 0;
    }();
}
}}

cpp_class!(pub unsafe struct PjRtClient as "std::shared_ptr<PjRtClient>");

fn init_cpu_lapack() {
    cpp! {{
        namespace jax {
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("blas_strsm", Trsm<float>::Kernel,
                                                "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("blas_dtrsm", Trsm<double>::Kernel,
                                                "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("blas_ctrsm",
                                                Trsm<std::complex<float>>::Kernel,
                                                "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("blas_ztrsm",
                                                Trsm<std::complex<double>>::Kernel,
                                                "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_sgetrf", Getrf<float>::Kernel,
                                                "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_dgetrf", Getrf<double>::Kernel,
                                                "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_cgetrf",
                                                Getrf<std::complex<float>>::Kernel,
                                                "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_zgetrf",
                                                Getrf<std::complex<double>>::Kernel,
                                                "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_sgeqrf", Geqrf<float>::Kernel,
                                                "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_dgeqrf", Geqrf<double>::Kernel,
                                                "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_cgeqrf",
                                                Geqrf<std::complex<float>>::Kernel,
                                                "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_zgeqrf",
                                                Geqrf<std::complex<double>>::Kernel,
                                                "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_sorgqr", Orgqr<float>::Kernel,
                                                "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_dorgqr", Orgqr<double>::Kernel,
                                                "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_cungqr",
                                                Orgqr<std::complex<float>>::Kernel,
                                                "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_zungqr",
                                                Orgqr<std::complex<double>>::Kernel,
                                                "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_spotrf", Potrf<float>::Kernel,
                                                "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_dpotrf", Potrf<double>::Kernel,
                                                "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_cpotrf",
                                                Potrf<std::complex<float>>::Kernel,
                                                "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_zpotrf",
                                                Potrf<std::complex<double>>::Kernel,
                                                "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_sgesdd",
                                                RealGesdd<float>::Kernel, "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_dgesdd",
                                                RealGesdd<double>::Kernel, "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
            "lapack_cgesdd", ComplexGesdd<std::complex<float>>::Kernel, "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
            "lapack_zgesdd", ComplexGesdd<std::complex<double>>::Kernel, "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_ssyevd",
                                                RealSyevd<float>::Kernel, "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_dsyevd",
                                                RealSyevd<double>::Kernel, "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
            "lapack_cheevd", ComplexHeevd<std::complex<float>>::Kernel, "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
            "lapack_zheevd", ComplexHeevd<std::complex<double>>::Kernel, "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_sgeev",
                                                RealGeev<float>::Kernel, "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_dgeev",
                                                RealGeev<double>::Kernel, "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
            "lapack_cgeev", ComplexGeev<std::complex<float>>::Kernel, "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
            "lapack_zgeev", ComplexGeev<std::complex<double>>::Kernel, "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_sgees",
                                                RealGees<float>::Kernel, "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_dgees",
                                                RealGees<double>::Kernel, "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
            "lapack_cgees", ComplexGees<std::complex<float>>::Kernel, "Host");
        XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
            "lapack_zgees", ComplexGees<std::complex<double>>::Kernel, "Host");
        }
    }}
}

impl PjRtClient {
    pub fn cpu() -> Result<Self> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        init_cpu_lapack();
        let client = unsafe {
            cpp!([out_status as "Status*"] -> PjRtClient as "std::shared_ptr<PjRtClient>" {
                auto status = xla::GetTfrtCpuClient(false);
                if (status.ok()) {
                    return std::shared_ptr(std::move(status.value()));
                }else{
                    *out_status = Status(status.status());
                    return std::shared_ptr<PjRtClient>();
                }
            })
        };
        out_status.to_result()?;
        if client.is_null() {
            let backtrace = std::backtrace::Backtrace::capture().to_string();
            return Err(Error::XlaError {
                msg: "Unexpected null pointer".to_string(),
                backtrace,
            });
        }
        Ok(client)
    }

    pub fn gpu(memory_fraction: f64, preallocate: bool) -> Result<Self> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let client = unsafe {
            cpp!([out_status as "Status*", memory_fraction as "double", preallocate as "bool"] -> PjRtClient as "std::shared_ptr<PjRtClient>" {
                GpuAllocatorConfig allocator = {.memory_fraction = memory_fraction,
                                       .preallocate = preallocate};
                auto status = GetStreamExecutorGpuClient(false, allocator, 0, 0);
                if (status.ok()) {
                    return std::shared_ptr(std::move(status.value()));
                }else{
                    *out_status = Status(status.status());
                    return std::shared_ptr<PjRtClient>();
                }
            })
        };
        out_status.to_result()?;
        if client.is_null() {
            let backtrace = std::backtrace::Backtrace::capture().to_string();
            return Err(Error::XlaError {
                msg: "Unexpected null pointer".to_string(),
                backtrace,
            });
        }
        Ok(client)
    }

    pub fn copy_host_buffer<T: ArrayElement>(&self, buf: &[T], dims: &[i64]) -> Result<PjRtBuffer> {
        let element_count: usize = dims.iter().product::<i64>() as usize;
        if element_count != buf.len() {
            return Err(Error::WrongElementCount {
                dims: dims.to_vec(),
                element_count,
            });
        }
        let buf_ptr = buf.as_ptr();
        let dims_ptr = dims.as_ptr();
        let dims_len = dims.len();
        let prim_type = T::TY.primitive_type() as i32;
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let buffer = unsafe {
            cpp!([self as "std::shared_ptr<PjRtClient>*", buf_ptr as "const uint8_t*", out_status as "Status*", dims_ptr as "const int64_t*", dims_len as "size_t", prim_type as "int32_t"] -> PjRtBuffer as "std::unique_ptr<PjRtBuffer>" {
                auto client = *self;
                auto device = client->devices()[0];
                auto status = client->BufferFromHostBuffer(
                    buf_ptr,
                    (PrimitiveType)prim_type,
                    absl::Span(dims_ptr, dims_len), {},
                    PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, []() {}, device
                );
                if (status.ok()) {
                    return std::unique_ptr(std::move(status.value()));
                }else{
                    *out_status = Status(status.status());
                    return std::unique_ptr<PjRtBuffer>();
                }
            })
        };
        out_status.to_result()?;
        if buffer.is_null() {
            let backtrace = std::backtrace::Backtrace::capture().to_string();
            return Err(Error::XlaError {
                msg: "Unexpected null pointer".to_string(),
                backtrace,
            });
        }
        Ok(buffer)
    }

    pub fn copy_raw_host_buffer(
        &self,
        ty: super::ElementType,
        buf: &[u8],
        dims: &[i64],
    ) -> Result<PjRtBuffer> {
        let element_count: usize = dims.iter().product::<i64>() as usize;
        let element_size_in_bytes = ty.element_size_in_bytes();
        if element_count * element_size_in_bytes != buf.len() {
            Err(Error::WrongElementCount {
                dims: dims.to_vec(),
                element_count,
            })?
        }
        let buf_ptr = buf.as_ptr();
        let dims_ptr = dims.as_ptr();
        let dims_len = dims.len();
        let prim_type = ty.primitive_type() as i32;
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let buffer = unsafe {
            cpp!([self as "std::shared_ptr<PjRtClient>*", buf_ptr as "const uint8_t*", out_status as "Status*", dims_ptr as "const int64_t*", dims_len as "size_t", prim_type as "int32_t"] -> PjRtBuffer as "std::unique_ptr<PjRtBuffer>" {
                auto client = *self;
                auto device = client->devices()[0];
                auto status = client->BufferFromHostBuffer(
                    buf_ptr,
                    (PrimitiveType)prim_type,
                    absl::Span(dims_ptr, dims_len), {},
                    PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, []() {}, device
                );
                if (status.ok()) {
                    return std::unique_ptr(std::move(status.value()));
                }else{
                    *out_status = Status(status.status());
                    return std::unique_ptr<PjRtBuffer>();
                }
            })
        };
        out_status.to_result()?;
        if buffer.is_null() {
            let backtrace = std::backtrace::Backtrace::capture().to_string();
            return Err(Error::XlaError {
                msg: "Unexpected null pointer".to_string(),
                backtrace,
            });
        }
        Ok(buffer)
    }

    pub fn copy_literal(&self, literal: &Literal) -> Result<PjRtBuffer> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let buffer = unsafe {
            cpp!([self as "std::shared_ptr<PjRtClient>*", literal as "const std::shared_ptr<Literal>*", out_status as "Status*"] -> PjRtBuffer as "std::unique_ptr<PjRtBuffer>" {
                auto client = *self;
                auto device = client->devices()[0];
                auto status = client->BufferFromHostLiteral(*literal->get(), device);
                if (status.ok()) {
                    return std::unique_ptr(std::move(status.value()));
                }else{
                    *out_status = Status(status.status());
                    return std::unique_ptr<PjRtBuffer>();
                }
            })
        };
        out_status.to_result()?;
        if buffer.is_null() {
            let backtrace = std::backtrace::Backtrace::capture().to_string();
            return Err(Error::XlaError {
                msg: "Unexpected null pointer".to_string(),
                backtrace,
            });
        }
        Ok(buffer)
    }

    pub fn compile(&self, comp: &XlaComputation) -> Result<PjRtLoadedExecutable> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let exec = unsafe {
            cpp!([self as "std::shared_ptr<PjRtClient>*", comp as "const XlaComputation*", out_status as "Status*"] -> PjRtLoadedExecutable as "std::shared_ptr<PjRtLoadedExecutable>" {
                auto client = *self;
                CompileOptions options;
                auto status = client->Compile(*comp, options);
                if (status.ok()) {
                    return std::shared_ptr(std::move(status.value()));
                }else{
                    *out_status = Status(status.status());
                    return std::shared_ptr<PjRtLoadedExecutable>();
                }
            })
        };
        out_status.to_result()?;
        if exec.is_null() {
            let backtrace = std::backtrace::Backtrace::capture().to_string();
            return Err(Error::XlaError {
                msg: "Unexpected null pointer".to_string(),
                backtrace,
            });
        }
        Ok(exec)
    }

    pub(crate) fn is_null(&self) -> bool {
        unsafe {
            cpp!([self as "const std::shared_ptr<PjRtClient>*"] -> bool as "bool" {
                return self == nullptr;
            })
        }
    }
}
