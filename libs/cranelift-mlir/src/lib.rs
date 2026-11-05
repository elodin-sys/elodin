pub mod const_cache;
pub mod const_fold;
pub mod debug;
pub mod display_kernel;
pub mod interp;
pub mod ir;
pub mod parser;
// tensor_rt exposes `pub extern "C" fn` taking raw pointers because
// Cranelift JIT dispatches them via extern "C" signature; they cannot
// be marked `unsafe fn`. needless_range_loop is suppressed because the
// N-D tensor index math is clearer with explicit indexed loops.
#[allow(clippy::not_unsafe_ptr_arg_deref, clippy::needless_range_loop)]
pub mod tensor_rt;

#[cfg(feature = "jit")]
pub mod inliner;
#[cfg(feature = "jit")]
pub mod lower;
#[cfg(feature = "jit")]
pub mod op_sampler;
#[cfg(feature = "jit")]
pub mod profile;
#[cfg(feature = "jit")]
pub mod slot_pool;
#[cfg(feature = "jit")]
pub mod useinfo;
