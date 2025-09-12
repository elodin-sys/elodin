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

   #pragma clang diagnostic ignored "-Wundefined-var-template"
   namespace ffi = xla::ffi;

   extern "C" {

   jax::TriMatrixEquationSolver<ffi::DataType::F32>::FnType strsm_;
   jax::TriMatrixEquationSolver<ffi::DataType::F64>::FnType dtrsm_;
   jax::TriMatrixEquationSolver<ffi::DataType::C64>::FnType ctrsm_;
   jax::TriMatrixEquationSolver<ffi::DataType::C128>::FnType ztrsm_;

   jax::LuDecomposition<ffi::DataType::F32>::FnType sgetrf_;
   jax::LuDecomposition<ffi::DataType::F64>::FnType dgetrf_;
   jax::LuDecomposition<ffi::DataType::C64>::FnType cgetrf_;
   jax::LuDecomposition<ffi::DataType::C128>::FnType zgetrf_;

   jax::QrFactorization<ffi::DataType::F32>::FnType sgeqrf_;
   jax::QrFactorization<ffi::DataType::F64>::FnType dgeqrf_;
   jax::QrFactorization<ffi::DataType::C64>::FnType cgeqrf_;
   jax::QrFactorization<ffi::DataType::C128>::FnType zgeqrf_;

   jax::PivotingQrFactorization<ffi::DataType::F32>::FnType sgeqp3_;
   jax::PivotingQrFactorization<ffi::DataType::F64>::FnType dgeqp3_;
   jax::PivotingQrFactorization<ffi::DataType::C64>::FnType cgeqp3_;
   jax::PivotingQrFactorization<ffi::DataType::C128>::FnType zgeqp3_;

   jax::OrthogonalQr<ffi::DataType::F32>::FnType sorgqr_;
   jax::OrthogonalQr<ffi::DataType::F64>::FnType dorgqr_;
   jax::OrthogonalQr<ffi::DataType::C64>::FnType cungqr_;
   jax::OrthogonalQr<ffi::DataType::C128>::FnType zungqr_;

   jax::CholeskyFactorization<ffi::DataType::F32>::FnType spotrf_;
   jax::CholeskyFactorization<ffi::DataType::F64>::FnType dpotrf_;
   jax::CholeskyFactorization<ffi::DataType::C64>::FnType cpotrf_;
   jax::CholeskyFactorization<ffi::DataType::C128>::FnType zpotrf_;

   jax::SingularValueDecomposition<ffi::DataType::F32>::FnType sgesdd_;
   jax::SingularValueDecomposition<ffi::DataType::F64>::FnType dgesdd_;
   jax::SingularValueDecompositionComplex<ffi::DataType::C64>::FnType cgesdd_;
   jax::SingularValueDecompositionComplex<ffi::DataType::C128>::FnType zgesdd_;

   jax::SingularValueDecompositionQR<ffi::DataType::F32>::FnType sgesvd_;
   jax::SingularValueDecompositionQR<ffi::DataType::F64>::FnType dgesvd_;
   jax::SingularValueDecompositionQRComplex<ffi::DataType::C64>::FnType cgesvd_;
   jax::SingularValueDecompositionQRComplex<ffi::DataType::C128>::FnType zgesvd_;

   jax::EigenvalueDecompositionSymmetric<ffi::DataType::F32>::FnType ssyevd_;
   jax::EigenvalueDecompositionSymmetric<ffi::DataType::F64>::FnType dsyevd_;
   jax::EigenvalueDecompositionHermitian<ffi::DataType::C64>::FnType cheevd_;
   jax::EigenvalueDecompositionHermitian<ffi::DataType::C128>::FnType zheevd_;

   jax::EigenvalueDecomposition<ffi::DataType::F32>::FnType sgeev_;
   jax::EigenvalueDecomposition<ffi::DataType::F64>::FnType dgeev_;
   jax::EigenvalueDecompositionComplex<ffi::DataType::C64>::FnType cgeev_;
   jax::EigenvalueDecompositionComplex<ffi::DataType::C128>::FnType zgeev_;

   jax::SchurDecomposition<ffi::DataType::F32>::FnType sgees_;
   jax::SchurDecomposition<ffi::DataType::F64>::FnType dgees_;
   jax::SchurDecompositionComplex<ffi::DataType::C64>::FnType cgees_;
   jax::SchurDecompositionComplex<ffi::DataType::C128>::FnType zgees_;

   jax::HessenbergDecomposition<ffi::DataType::F32>::FnType sgehrd_;
   jax::HessenbergDecomposition<ffi::DataType::F64>::FnType dgehrd_;
   jax::HessenbergDecomposition<ffi::DataType::C64>::FnType cgehrd_;
   jax::HessenbergDecomposition<ffi::DataType::C128>::FnType zgehrd_;

   jax::TridiagonalReduction<ffi::DataType::F32>::FnType ssytrd_;
   jax::TridiagonalReduction<ffi::DataType::F64>::FnType dsytrd_;
   jax::TridiagonalReduction<ffi::DataType::C64>::FnType chetrd_;
   jax::TridiagonalReduction<ffi::DataType::C128>::FnType zhetrd_;

   jax::TridiagonalSolver<ffi::DataType::F32>::FnType sgtsv_;
   jax::TridiagonalSolver<ffi::DataType::F64>::FnType dgtsv_;
   jax::TridiagonalSolver<ffi::DataType::C64>::FnType cgtsv_;
   jax::TridiagonalSolver<ffi::DataType::C128>::FnType zgtsv_;

   }  // extern "C"

   namespace jax {

   static auto init = []() -> int {
     AssignKernelFn<TriMatrixEquationSolver<ffi::DataType::F32>>(strsm_);
     AssignKernelFn<TriMatrixEquationSolver<ffi::DataType::F64>>(dtrsm_);
     AssignKernelFn<TriMatrixEquationSolver<ffi::DataType::C64>>(ctrsm_);
     AssignKernelFn<TriMatrixEquationSolver<ffi::DataType::C128>>(ztrsm_);

     AssignKernelFn<LuDecomposition<ffi::DataType::F32>>(sgetrf_);
     AssignKernelFn<LuDecomposition<ffi::DataType::F64>>(dgetrf_);
     AssignKernelFn<LuDecomposition<ffi::DataType::C64>>(cgetrf_);
     AssignKernelFn<LuDecomposition<ffi::DataType::C128>>(zgetrf_);

     AssignKernelFn<QrFactorization<ffi::DataType::F32>>(sgeqrf_);
     AssignKernelFn<QrFactorization<ffi::DataType::F64>>(dgeqrf_);
     AssignKernelFn<QrFactorization<ffi::DataType::C64>>(cgeqrf_);
     AssignKernelFn<QrFactorization<ffi::DataType::C128>>(zgeqrf_);

     AssignKernelFn<PivotingQrFactorization<ffi::DataType::F32>>(sgeqp3_);
     AssignKernelFn<PivotingQrFactorization<ffi::DataType::F64>>(dgeqp3_);
     AssignKernelFn<PivotingQrFactorization<ffi::DataType::C64>>(cgeqp3_);
     AssignKernelFn<PivotingQrFactorization<ffi::DataType::C128>>(zgeqp3_);

     AssignKernelFn<OrthogonalQr<ffi::DataType::F32>>(sorgqr_);
     AssignKernelFn<OrthogonalQr<ffi::DataType::F64>>(dorgqr_);
     AssignKernelFn<OrthogonalQr<ffi::DataType::C64>>(cungqr_);
     AssignKernelFn<OrthogonalQr<ffi::DataType::C128>>(zungqr_);

     AssignKernelFn<CholeskyFactorization<ffi::DataType::F32>>(spotrf_);
     AssignKernelFn<CholeskyFactorization<ffi::DataType::F64>>(dpotrf_);
     AssignKernelFn<CholeskyFactorization<ffi::DataType::C64>>(cpotrf_);
     AssignKernelFn<CholeskyFactorization<ffi::DataType::C128>>(zpotrf_);

     AssignKernelFn<SingularValueDecomposition<ffi::DataType::F32>>(sgesdd_);
     AssignKernelFn<SingularValueDecomposition<ffi::DataType::F64>>(dgesdd_);
     AssignKernelFn<SingularValueDecompositionComplex<ffi::DataType::C64>>(
         cgesdd_);
     AssignKernelFn<SingularValueDecompositionComplex<ffi::DataType::C128>>(
         zgesdd_);

     AssignKernelFn<SingularValueDecompositionQR<ffi::DataType::F32>>(sgesvd_);
     AssignKernelFn<SingularValueDecompositionQR<ffi::DataType::F64>>(dgesvd_);
     AssignKernelFn<SingularValueDecompositionQRComplex<ffi::DataType::C64>>(
         cgesvd_);
     AssignKernelFn<SingularValueDecompositionQRComplex<ffi::DataType::C128>>(
         zgesvd_);

     AssignKernelFn<EigenvalueDecompositionSymmetric<ffi::DataType::F32>>(ssyevd_);
     AssignKernelFn<EigenvalueDecompositionSymmetric<ffi::DataType::F64>>(dsyevd_);
     AssignKernelFn<EigenvalueDecompositionHermitian<ffi::DataType::C64>>(cheevd_);
     AssignKernelFn<EigenvalueDecompositionHermitian<ffi::DataType::C128>>(
         zheevd_);

     AssignKernelFn<EigenvalueDecomposition<ffi::DataType::F32>>(sgeev_);
     AssignKernelFn<EigenvalueDecomposition<ffi::DataType::F64>>(dgeev_);
     AssignKernelFn<EigenvalueDecompositionComplex<ffi::DataType::C64>>(cgeev_);
     AssignKernelFn<EigenvalueDecompositionComplex<ffi::DataType::C128>>(zgeev_);

     AssignKernelFn<TridiagonalReduction<ffi::DataType::F32>>(ssytrd_);
     AssignKernelFn<TridiagonalReduction<ffi::DataType::F64>>(dsytrd_);
     AssignKernelFn<TridiagonalReduction<ffi::DataType::C64>>(chetrd_);
     AssignKernelFn<TridiagonalReduction<ffi::DataType::C128>>(zhetrd_);

     AssignKernelFn<SchurDecomposition<ffi::DataType::F32>>(sgees_);
     AssignKernelFn<SchurDecomposition<ffi::DataType::F64>>(dgees_);
     AssignKernelFn<SchurDecompositionComplex<ffi::DataType::C64>>(cgees_);
     AssignKernelFn<SchurDecompositionComplex<ffi::DataType::C128>>(zgees_);

     AssignKernelFn<HessenbergDecomposition<ffi::DataType::F32>>(sgehrd_);
     AssignKernelFn<HessenbergDecomposition<ffi::DataType::F64>>(dgehrd_);
     AssignKernelFn<HessenbergDecomposition<ffi::DataType::C64>>(cgehrd_);
     AssignKernelFn<HessenbergDecomposition<ffi::DataType::C128>>(zgehrd_);

     AssignKernelFn<TridiagonalSolver<ffi::DataType::F32>>(sgtsv_);
     AssignKernelFn<TridiagonalSolver<ffi::DataType::F64>>(dgtsv_);
     AssignKernelFn<TridiagonalSolver<ffi::DataType::C64>>(cgtsv_);
     AssignKernelFn<TridiagonalSolver<ffi::DataType::C128>>(zgtsv_);

     return 0;
   }();
   }  // namespace jax
}}

cpp_class!(pub unsafe struct PjRtClient as "std::shared_ptr<PjRtClient>");

fn init_cpu_lapack() {
    cpp! {{
       namespace jax {
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "blas_strsm",
           jax::TriMatrixEquationSolver<ffi::DataType::F32>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "blas_dtrsm",
           jax::TriMatrixEquationSolver<ffi::DataType::F64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "blas_ctrsm",
           jax::TriMatrixEquationSolver<ffi::DataType::C64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "blas_ztrsm",
           jax::TriMatrixEquationSolver<ffi::DataType::C128>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_sgetrf",
           jax::LuDecomposition<ffi::DataType::F32>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
            "lapack_dgetrf",
            jax::LuDecomposition<ffi::DataType::F64>::Kernel,
            "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_cgetrf",
           jax::LuDecomposition<ffi::DataType::C64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_zgetrf",
           jax::LuDecomposition<ffi::DataType::C128>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_sgeqrf",
           jax::QrFactorization<ffi::DataType::F32>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_dgeqrf",
           jax::QrFactorization<ffi::DataType::F64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_cgeqrf",
           jax::QrFactorization<ffi::DataType::C64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_zgeqrf",
           jax::QrFactorization<ffi::DataType::C128>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_sgeqp3",
           jax::PivotingQrFactorization<ffi::DataType::F32>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_dgeqp3",
           jax::PivotingQrFactorization<ffi::DataType::F64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_cgeqp3",
           jax::PivotingQrFactorization<ffi::DataType::C64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_zgeqp3",
           jax::PivotingQrFactorization<ffi::DataType::C128>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_sorgqr",
           jax::OrthogonalQr<ffi::DataType::F32>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_dorgqr",
           jax::OrthogonalQr<ffi::DataType::F64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_cungqr",
           jax::OrthogonalQr<ffi::DataType::C64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_zungqr",
           jax::OrthogonalQr<ffi::DataType::C128>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_spotrf",
           jax::CholeskyFactorization<ffi::DataType::F32>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_dpotrf",
           jax::CholeskyFactorization<ffi::DataType::F64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_cpotrf",
           jax::CholeskyFactorization<ffi::DataType::C64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_zpotrf",
           jax::CholeskyFactorization<ffi::DataType::C128>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_sgesdd",
           jax::SingularValueDecomposition<ffi::DataType::F32>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_dgesdd",
           jax::SingularValueDecomposition<ffi::DataType::F64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_cgesdd",
           jax::SingularValueDecompositionComplex<ffi::DataType::C64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_zgesdd",
           jax::SingularValueDecompositionComplex<ffi::DataType::C128>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_sgesvd",
           jax::SingularValueDecompositionQR<ffi::DataType::F32>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_dgesvd",
           jax::SingularValueDecompositionQR<ffi::DataType::F64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_cgesvd",
           jax::SingularValueDecompositionQRComplex<ffi::DataType::C64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_zgesvd",
           jax::SingularValueDecompositionQRComplex<ffi::DataType::C128>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_ssyevd",
           jax::EigenvalueDecompositionSymmetric<ffi::DataType::F32>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_dsyevd",
           jax::EigenvalueDecompositionSymmetric<ffi::DataType::F64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_cheevd",
           jax::EigenvalueDecompositionHermitian<ffi::DataType::C64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_zheevd",
           jax::EigenvalueDecompositionHermitian<ffi::DataType::C128>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_sgeev",
           jax::EigenvalueDecomposition<ffi::DataType::F32>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_dgeev",
           jax::EigenvalueDecomposition<ffi::DataType::F64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_cgeev",
           jax::EigenvalueDecompositionComplex<ffi::DataType::C64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_zgeev",
           jax::EigenvalueDecompositionComplex<ffi::DataType::C128>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_sgees",
           jax::SchurDecomposition<ffi::DataType::F32>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_dgees",
           jax::SchurDecomposition<ffi::DataType::F64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_cgees",
           jax::SchurDecompositionComplex<ffi::DataType::C64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_zgees",
           jax::SchurDecompositionComplex<ffi::DataType::C128>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_sgehrd",
           jax::HessenbergDecomposition<ffi::DataType::F32>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_dgehrd",
           jax::HessenbergDecomposition<ffi::DataType::F64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_cgehrd",
           jax::HessenbergDecomposition<ffi::DataType::C64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_zgehrd",
           jax::HessenbergDecomposition<ffi::DataType::C128>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_ssytrd",
           jax::TridiagonalReduction<ffi::DataType::F32>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_dsytrd",
           jax::TridiagonalReduction<ffi::DataType::F64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_csytrd",
           jax::TridiagonalReduction<ffi::DataType::C64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_zsytrd",
           jax::TridiagonalReduction<ffi::DataType::C128>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_sgtsv",
           jax::TridiagonalSolver<ffi::DataType::F32>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_dgtsv",
           jax::TridiagonalSolver<ffi::DataType::F64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_cgtsv",
           jax::TridiagonalSolver<ffi::DataType::C64>::Kernel,
           "Host"
       );
       XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
           "lapack_zgtsv",
           jax::TridiagonalSolver<ffi::DataType::C128>::Kernel,
           "Host"
       );
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
            cpp!([out_status as "absl::Status*"] -> PjRtClient as "std::shared_ptr<PjRtClient>" {
                auto status = GetXlaPjrtCpuClient(CpuClientOptions());
                if (status.ok()) {
                    return std::shared_ptr(std::move(status.value()));
                }else{
                    *out_status = absl::Status(status.status());
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
            cpp!([out_status as "__attribute__((unused)) absl::Status*", memory_fraction as "__attribute__((unused)) double", preallocate as "__attribute__((unused)) bool"] -> PjRtClient as "std::shared_ptr<PjRtClient>" {
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
                    *out_status = absl::Status(status.status());
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
            cpp!([self as "std::shared_ptr<PjRtClient>*", buf_ptr as "const uint8_t*", out_status as "absl::Status*", dims_ptr as "const int64_t*", dims_len as "size_t", prim_type as "int32_t"] -> PjRtBuffer as "std::unique_ptr<PjRtBuffer>" {
                auto client = *self;
                auto device = client->devices()[0];
                PjRtMemorySpace* memory_space = *device->default_memory_space();
                auto status = client->BufferFromHostBuffer(
                    buf_ptr,
                    (PrimitiveType)prim_type,
                    absl::Span(dims_ptr, dims_len), {},
                    PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
                    nullptr, memory_space, nullptr
                );
                if (status.ok()) {
                    return std::unique_ptr(std::move(status.value()));
                }else{
                    *out_status = absl::Status(status.status());
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
            cpp!([self as "std::shared_ptr<PjRtClient>*", buf_ptr as "const uint8_t*", out_status as "absl::Status*", dims_ptr as "const int64_t*", dims_len as "size_t", prim_type as "int32_t"] -> PjRtBuffer as "std::unique_ptr<PjRtBuffer>" {
                auto client = *self;
                auto device = client->devices()[0];
                PjRtMemorySpace* memory_space = *device->default_memory_space();
                auto status = client->BufferFromHostBuffer(
                    buf_ptr,
                    (PrimitiveType)prim_type,
                    absl::Span(dims_ptr, dims_len), {},
                    PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
                    nullptr, memory_space, nullptr
                );
                if (status.ok()) {
                    return std::unique_ptr(std::move(status.value()));
                }else{
                    *out_status = absl::Status(status.status());
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
            cpp!([self as "std::shared_ptr<PjRtClient>*", literal as "const std::shared_ptr<Literal>*", out_status as "absl::Status*"] -> PjRtBuffer as "std::unique_ptr<PjRtBuffer>" {
                auto client = *self;
                auto device = client->devices()[0];
                PjRtMemorySpace* memory_space = *device->default_memory_space();
                auto status = client->BufferFromHostLiteral(*literal->get(), memory_space);
                if (status.ok()) {
                    return std::unique_ptr(std::move(status.value()));
                }else{
                    *out_status = absl::Status(status.status());
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
            cpp!([self as "std::shared_ptr<PjRtClient>*", mut options as "CompileOptions", comp as "const XlaComputation*", out_status as "absl::Status*"] -> PjRtLoadedExecutable as "std::shared_ptr<PjRtLoadedExecutable>" {
                auto client = *self;
                auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(tsl::Env::Default(), "", tsl::port::MaxParallelism());
                options.executable_build_options.set_compile_thread_pool(thread_pool.get());
                auto status = client->CompileAndLoad(*comp, options);
                if (status.ok()) {
                    return std::shared_ptr(std::move(status.value()));
                }else{
                    *out_status = absl::Status(status.status());
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
            let src_ptr = cpp!([self as "std::shared_ptr<PjRtClient>*", buffer as "const std::unique_ptr<PjRtBuffer>*", out_status as "absl::Status*"] -> *const u8 as "std::uintptr_t" {
                auto status = (*self)->UnsafeBufferPointer(buffer->get());
                if (status.ok()) {
                    return status.value();
                } else {
                    *out_status = absl::Status(status.status());
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
