pub mod const_fold;
pub mod debug;
pub mod inliner;
pub mod ir;
pub mod lower;
pub mod op_sampler;
pub mod parser;
pub mod profile;
pub mod slot_pool;
// tensor_rt exposes `pub extern "C" fn` taking raw pointers because
// Cranelift JIT dispatches them via extern "C" signature; they cannot
// be marked `unsafe fn`. needless_range_loop is suppressed because the
// N-D tensor index math is clearer with explicit indexed loops.
#[allow(clippy::not_unsafe_ptr_arg_deref, clippy::needless_range_loop)]
pub mod tensor_rt;
pub mod useinfo;
