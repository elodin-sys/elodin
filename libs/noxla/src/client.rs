use crate::{
    ArrayElement, CompileOptions, Error, Literal, PjRtBuffer, PjRtLoadedExecutable, Result, Status,
    XlaComputation,
};
use cpp::{cpp, cpp_class};
use std::pin::Pin;

cpp! {{
    #pragma GCC diagnostic ignored "-Wignored-attributes"
    #include "xla/pjrt/pjrt_api.h"
    #include "xla/pjrt/pjrt_c_api_client.h"
    #include "xla/pjrt/pjrt_client.h"
    #include "xla/pjrt/pjrt_stream_executor_client.h"
    #include "xla/pjrt/tfrt_cpu_pjrt_client.h"
    #include "xla/pjrt/gpu/gpu_helpers.h"
    #include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
    #include "xla/service/custom_call_target_registry.h"
    #include "jaxlib/cpu/lapack_kernels.h"
    #include "xla/service/custom_call_target_registry.h"
    #include "tsl/platform/cpu_info.h"

    #ifdef EL_CUDA
    #include "xla/ffi/api/api.h"
    #include "xla/ffi/ffi_api.h"
    #include "jaxlib/gpu/blas_kernels.h"
    #include "jaxlib/gpu/cholesky_update_kernel.h"
    #include "jaxlib/gpu/lu_pivot_kernels.h"
    #include "jaxlib/gpu/prng_kernels.h"
    #include "jaxlib/gpu/rnn_kernels.h"
    #include "jaxlib/gpu/solver_kernels.h"
    #include "jaxlib/gpu/sparse_kernels.h"
    //#include "jaxlib/gpu/triton_kernels.h"
    //#include "jaxlib/gpu/vendor.h"
    #endif
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
        #ifdef EL_CUDA
        namespace jax::cuda {
            XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cublas_getrf_batched", GetrfBatched,
            "CUDA");
            XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cublas_geqrf_batched", GeqrfBatched,
            "CUDA");
            XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cudnn_rnn", RNNForward, "CUDA");
            XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cudnn_rnn_bwd", RNNBackward, "CUDA");
            XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cu_cholesky_update",
            CholeskyUpdate, "CUDA");
            //XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cu_lu_pivots_to_permutation", LuPivotsToPermutation, "CUDA");
            XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cu_threefry2x32", ThreeFry2x32,
            "CUDA");
            XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusolver_getrf", Getrf, "CUDA");
            XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusolver_geqrf", Geqrf, "CUDA");
            XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusolver_csrlsvqr", Csrlsvqr, "CUDA");
            XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusolver_orgqr", Orgqr, "CUDA");
            XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusolver_syevd", Syevd, "CUDA");
            XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusolver_syevj", Syevj, "CUDA");
            XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusolver_sytrd", Sytrd, "CUDA");
            XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusolver_gesvd", Gesvd, "CUDA");
            XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusolver_gesvdj", Gesvdj, "CUDA");
        }
        #endif
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

    #[cfg(feature = "cuda")]
    #[allow(unused_variables)]
    pub fn gpu(memory_fraction: f64, preallocate: bool) -> Result<Self> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        init_cpu_lapack();
        let client = unsafe {
            cpp!([out_status as "__attribute__((unused)) Status*", memory_fraction as "__attribute__((unused)) double", preallocate as "__attribute__((unused)) bool"] -> PjRtClient as "std::shared_ptr<PjRtClient>" {
                #ifdef EL_CUDA
                auto reg = CustomCallTargetRegistry::Global();
                xla::ffi::Ffi::RegisterStaticHandler(
                    xla::ffi::GetXlaFfiApi(),
                    "cu_lu_pivots_to_permutation",
                    "CUDA",
                    reinterpret_cast<XLA_FFI_Handler*>(jax::cuda::LuPivotsToPermutation)
                );
                auto syms = reg->registered_symbols("CUDA");
                auto is_null = CustomCallTargetRegistry::Global()->Lookup("cu_lu_pivots_to_permutation", std::string("CUDA"));
                GpuAllocatorConfig allocator = {.memory_fraction = memory_fraction,
                                       .preallocate = preallocate};
                GpuClientOptions options = {
                    .allocator_config = allocator,
                    .platform_name = "CUDA"
                };
                auto status = GetStreamExecutorGpuClient(options);
                if (status.ok()) {
                    return std::shared_ptr(std::move(status.value()));
                }else{
                    *out_status = Status(status.status());
                    return std::shared_ptr<PjRtClient>();
                }
                #else
                return std::shared_ptr<PjRtClient>();
                #endif
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

    pub fn compile_with_options(
        &self,
        comp: &XlaComputation,
        options: CompileOptions,
    ) -> Result<PjRtLoadedExecutable> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let mut options = options.0;
        let exec = unsafe {
            cpp!([self as "std::shared_ptr<PjRtClient>*", mut options as "CompileOptions", comp as "const XlaComputation*", out_status as "Status*"] -> PjRtLoadedExecutable as "std::shared_ptr<PjRtLoadedExecutable>" {
                auto client = *self;
                auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(tsl::Env::Default(), "", tsl::port::MaxParallelism());
                options.executable_build_options.set_compile_thread_pool(thread_pool.get());
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

    pub fn compile_with_default_options(
        &self,
        comp: &XlaComputation,
    ) -> Result<PjRtLoadedExecutable> {
        self.compile_with_options(comp, Default::default())
    }

    pub(crate) fn is_null(&self) -> bool {
        unsafe {
            cpp!([self as "const std::shared_ptr<PjRtClient>*"] -> bool as "bool" {
                return self == nullptr;
            })
        }
    }

    pub fn to_host_vec(&self, buffer: &PjRtBuffer) -> Result<Vec<u8>> {
        let mut dst = Vec::default();
        self.copy_into_host_vec(buffer, &mut dst)?;
        Ok(dst)
    }

    pub fn copy_into_host_vec(&self, buffer: &PjRtBuffer, dst: &mut Vec<u8>) -> Result<()> {
        if !buffer.is_on_cpu() {
            // fallback to regular copy if buffer is not on CPU
            return buffer.copy_into(dst);
        }

        let len = buffer.shape().size();
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let src: &[u8] = unsafe {
            let src_ptr = cpp!([self as "std::shared_ptr<PjRtClient>*", buffer as "const std::unique_ptr<PjRtBuffer>*", out_status as "Status*"] -> *const u8 as "std::uintptr_t" {
                auto status = (*self)->UnsafeBufferPointer(buffer->get());
                if (status.ok()) {
                    return status.value();
                } else {
                    *out_status = Status(status.status());
                    return 0;
                }
            });
            std::slice::from_raw_parts(src_ptr, len)
        };
        out_status.to_result()?;
        dst.clear();
        dst.extend_from_slice(src);
        Ok(())
    }
}
