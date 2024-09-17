//! This module provides utilities for converting `Noxpr` expressions to `Jax` operations in Python.
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{IntoPyDict, PyDict, PyTuple},
    PyObject, PyResult, Python,
};
use smallvec::SmallVec;
use std::{collections::HashMap, ops::Deref, sync::Arc};
use xla::{ArrayElement, ElementType, Literal};

use crate::{BinaryOp, CompFn, Error, Noxpr, NoxprComp, NoxprFn, NoxprId, NoxprNode, ReprMonad};

impl Noxpr {
    /// Converts a `Noxpr` expression to a `Jax` operation using a tracer.
    pub fn to_jax(&self) -> Result<PyObject, Error> {
        let mut tracer = JaxTracer::new();
        tracer.visit(self)
    }
}

/// Traces `Noxpr` expressions and generates corresponding `Jax` operations.
#[derive(Clone)]
pub struct JaxTracer {
    lax: PyObject,
    jnp: PyObject,
    linalg: PyObject,
    cache: HashMap<NoxprId, PyObject>,
}

impl JaxTracer {
    /// Initializes a new tracer with references to the Jax libraries.
    pub fn new() -> Self {
        Python::with_gil(|py| {
            let lax = py.import_bound("jax.lax").unwrap().into();
            let jnp = py.import_bound("jax.numpy").unwrap().into();
            let linalg = py.import_bound("jax.numpy.linalg").unwrap().into();
            Self {
                lax,
                jnp,
                linalg,
                cache: HashMap::new(),
            }
        })
    }

    /// Visits a `Noxpr` expression and recursively translates it into a `Jax` operation.
    pub fn visit(&mut self, expr: &Noxpr) -> Result<PyObject, Error> {
        let id = expr.id();
        if let Some(op) = self.cache.get(&id) {
            return Ok(op.clone());
        }
        let op: PyObject = match expr.deref() {
            NoxprNode::Constant(c) => match c.ty.element_type {
                xla::ElementType::S8 => literal_to_arr::<i8>(&c.data, &c.ty.shape, &self.jnp)?,
                xla::ElementType::S16 => literal_to_arr::<i16>(&c.data, &c.ty.shape, &self.jnp)?,
                xla::ElementType::S32 => literal_to_arr::<i32>(&c.data, &c.ty.shape, &self.jnp)?,
                xla::ElementType::S64 => literal_to_arr::<i64>(&c.data, &c.ty.shape, &self.jnp)?,
                xla::ElementType::U8 => literal_to_arr::<u8>(&c.data, &c.ty.shape, &self.jnp)?,
                xla::ElementType::U16 => literal_to_arr::<u16>(&c.data, &c.ty.shape, &self.jnp)?,
                xla::ElementType::U32 => literal_to_arr::<u32>(&c.data, &c.ty.shape, &self.jnp)?,
                xla::ElementType::U64 => literal_to_arr::<u64>(&c.data, &c.ty.shape, &self.jnp)?,
                xla::ElementType::F32 => literal_to_arr::<f32>(&c.data, &c.ty.shape, &self.jnp)?,
                // xla::ElementType::F16 => literal_to_arr::<F16>(&c.data, &c.ty.shape, &self.jnp)?,
                // xla::ElementType::Bf16 => literal_to_arr::<Bf16>(&c.data, &c.ty.shape, &self.jnp)?,
                xla::ElementType::F64 => literal_to_arr::<f64>(&c.data, &c.ty.shape, &self.jnp)?,
                xla::ElementType::Pred => {
                    todo!()
                }
                xla::ElementType::C64 => todo!(),
                xla::ElementType::C128 => todo!(),
                xla::ElementType::F16 => todo!(),
                xla::ElementType::Bf16 => todo!(),
            },
            NoxprNode::Param(_) => {
                unimplemented!("param found")
            }
            NoxprNode::Tuple(args) => {
                let elems = args
                    .iter()
                    .map(|x| self.visit(x))
                    .collect::<Result<Vec<_>, _>>()?;
                Python::with_gil(|py| PyTuple::new_bound(py, elems).into_py(py))
            }
            NoxprNode::Iota(i) => Python::with_gil(|py| {
                let dtype = dtype(&i.shape.element_type)?;
                self.lax
                    .call_method1(
                        py,
                        "broadcasted_iota",
                        (dtype, i.shape.shape.to_vec(), i.dim),
                    )
                    .map_err(Error::PyO3)
            })?,
            NoxprNode::Add(op) => self.visit_binary_lax(op, "add")?,
            NoxprNode::Sub(op) => self.visit_binary_lax(op, "sub")?,
            NoxprNode::Mul(op) => self.visit_binary_lax(op, "mul")?,
            NoxprNode::Div(op) => self.visit_binary_lax(op, "div")?,
            NoxprNode::And(op) => self.visit_binary_lax(op, "bitwise_and")?,
            NoxprNode::Or(op) => self.visit_binary_lax(op, "bitwise_or")?,
            NoxprNode::Dot(op) => self.visit_binary_lax(op, "dot")?,
            NoxprNode::GreaterOrEqual(op) => self.visit_binary_lax(op, "ge")?,
            NoxprNode::Equal(op) => self.visit_binary_lax(op, "eq")?,
            NoxprNode::Atan2(op) => self.visit_binary_lax(op, "atan2")?,
            NoxprNode::LessOrEqual(op) => self.visit_binary_lax(op, "le")?,
            NoxprNode::Less(op) => self.visit_binary_lax(op, "lt")?,
            NoxprNode::DotGeneral(d) => {
                let lhs = self.visit(&d.lhs)?;
                let rhs = self.visit(&d.rhs)?;
                let contracting = (
                    d.dimensions.lhs_contracting_dimensions.to_vec(),
                    d.dimensions.rhs_contracting_dimensions.to_vec(),
                );
                let batch_dims = (
                    d.dimensions.lhs_batch_dimensions.to_vec(),
                    d.dimensions.rhs_batch_dimensions.to_vec(),
                );
                let dims = (contracting, batch_dims);
                Python::with_gil(|py| self.lax.call_method1(py, "dot_general", (lhs, rhs, dims)))?
            }
            NoxprNode::Sqrt(op) => self.visit_unary_lax(op, "sqrt")?,
            NoxprNode::Neg(op) => self.visit_unary_lax(op, "neg")?,
            NoxprNode::Log(op) => self.visit_unary_lax(op, "log")?,
            NoxprNode::Sin(op) => self.visit_unary_lax(op, "sin")?,
            NoxprNode::Cos(op) => self.visit_unary_lax(op, "cos")?,
            NoxprNode::Abs(op) => self.visit_unary_lax(op, "abs")?,
            NoxprNode::Concat(c) => {
                let nodes = c
                    .nodes
                    .iter()
                    .map(|x| self.visit(x))
                    .collect::<Result<Vec<_>, _>>()?;
                Python::with_gil(|py| {
                    self.lax
                        .call_method1(py, "concatenate", (nodes, c.dimension))
                })?
            }
            NoxprNode::Reshape(r) => {
                let expr = self.visit(&r.expr)?;
                let sizes = r.new_sizes.to_vec();
                Python::with_gil(|py| self.lax.call_method1(py, "reshape", (expr, sizes)))?
            }
            NoxprNode::Broadcast(b) => {
                let expr = self.visit(&b.expr)?;
                let sizes = b.sizes.to_vec();
                Python::with_gil(|py| self.lax.call_method1(py, "broadcast", (expr, sizes)))?
            }
            NoxprNode::BroadcastInDim(b) => {
                let expr = self.visit(&b.expr)?;
                let sizes = b.sizes.to_vec();
                let dims = b.broadcast_dims.to_vec();
                Python::with_gil(|py| {
                    self.lax
                        .call_method1(py, "broadcast_in_dim", (expr, sizes, dims))
                })?
            }
            NoxprNode::Transpose(t) => {
                let expr = self.visit(&t.expr)?;
                let permutation = t.permutation.to_vec();
                Python::with_gil(|py| self.lax.call_method1(py, "transpose", (expr, permutation)))?
            }
            NoxprNode::Gather(g) => {
                let expr = self.visit(&g.expr)?;
                let start_indices = self.visit(&g.indices)?;
                let gather_dims = Python::with_gil(|py| {
                    self.lax.call_method1(
                        py,
                        "GatherDimensionNumbers",
                        (
                            g.offset_dims.to_vec(),
                            g.collapsed_slice_dims.to_vec(),
                            g.start_index_map.to_vec(),
                        ),
                    )
                })?;
                Python::with_gil(|py| {
                    self.lax.call_method1(
                        py,
                        "gather",
                        (expr, start_indices, gather_dims, g.slice_sizes.to_vec()),
                    )
                })?
            }
            NoxprNode::Slice(s) => {
                let expr = self.visit(&s.expr)?;
                Python::with_gil(|py| {
                    self.lax
                        .call_method1(
                            py,
                            "slice",
                            (expr, s.start_indices.to_vec(), s.stop_indices.to_vec()),
                        )
                        .map_err(Error::PyO3)
                })?
            }
            NoxprNode::DynamicSlice(s) => {
                let expr = self.visit(&s.expr)?;
                let start_indices = s
                    .start_indices
                    .iter()
                    .map(|e| self.visit(e))
                    .collect::<Result<Vec<_>, Error>>()?;
                Python::with_gil(|py| {
                    self.lax
                        .call_method1(
                            py,
                            "dynamic_slice",
                            (expr, start_indices, s.size_indices.to_vec()),
                        )
                        .map_err(Error::PyO3)
                })?
            }
            NoxprNode::DynamicUpdateSlice(u) => {
                let expr = self.visit(&u.expr)?;
                let update = self.visit(&u.update)?;
                let start_indices = u
                    .start_indicies
                    .iter()
                    .map(|e| self.visit(e))
                    .collect::<Result<Vec<_>, Error>>()?;
                Python::with_gil(|py| {
                    self.lax
                        .call_method1(py, "dynamic_update_slice", (expr, update, start_indices))
                        .map_err(Error::PyO3)
                })?
            }
            NoxprNode::GetTupleElement(g) => match g.expr.deref() {
                NoxprNode::Tuple(elems) => {
                    let elem = elems.get(g.index).ok_or(Error::OutOfBoundsAccess)?;
                    self.visit(elem)?
                }
                _ => {
                    let jax = self.visit(&g.expr)?;
                    Python::with_gil(|py| {
                        jax.call_method1(py, "__getitem__", (g.index,))
                            .map(|x| x.into_py(py))
                    })
                    .unwrap()
                }
            },
            NoxprNode::Scan(s) => {
                let initial_state = self.visit(&s.initial_state)?;
                let inputs = s
                    .inputs
                    .iter()
                    .map(|x| self.visit(x))
                    .collect::<Result<Vec<_>, _>>()?;
                let scan_fn = self.visit_fn(&s.scan_fn);
                Python::with_gil(|py| {
                    self.lax
                        .call_method1(py, "scan", (scan_fn, initial_state, inputs))
                        .map_err(Error::PyO3)
                })?
            }
            NoxprNode::Jax(o) => o.clone(),
            NoxprNode::Convert(conv) => {
                let expr = self.visit(&conv.arg)?;
                let dtype = dtype(&conv.ty)?;
                Python::with_gil(|py| {
                    self.lax
                        .call_method1(py, "convert_element_type", (expr, dtype))
                        .map_err(Error::PyO3)
                })?
            }
            NoxprNode::Select(s) => {
                let pred = self.visit(&s.cond)?;
                let on_true = self.visit(&s.on_true)?;
                let on_false = self.visit(&s.on_false)?;
                Python::with_gil(|py| {
                    self.lax
                        .call_method1(py, "select", (pred, on_true, on_false))
                        .map_err(Error::PyO3)
                })?
            }
            NoxprNode::Call(c) => {
                let args = c
                    .args
                    .iter()
                    .map(|x| self.visit(x))
                    .collect::<Result<Vec<_>, _>>()?;
                let call_fn = self.visit_comp(&c.comp);
                Python::with_gil(|py| {
                    let call_fn = call_fn.into_py(py);
                    let tuple = PyTuple::new_bound(py, args.into_iter());
                    call_fn.call1(py, tuple)
                })?
            }
            NoxprNode::Cholesky(c) => {
                let expr = self.visit(&c.arg)?;
                Python::with_gil(|py| {
                    let kwargs = PyDict::new_bound(py);
                    kwargs.set_item("upper", c.upper)?;
                    self.linalg
                        .call_method_bound(py, "cholesky", (expr,), Some(&kwargs))
                })?
            }
            NoxprNode::LuInverse(lu) => {
                let expr = self.visit(&lu.arg)?;
                Python::with_gil(|py| self.linalg.call_method1(py, "inv", (expr,)))?
            }
        };
        self.cache.insert(id, op.clone());
        Ok(op)
    }

    fn visit_comp(&mut self, comp: &NoxprComp) -> PyObject {
        Python::with_gil(|py| {
            if let NoxprNode::Jax(obj) = &*comp.func.inner.node {
                if obj.getattr(py, "__call__").is_ok() {
                    return obj.clone();
                }
            }
            JaxNoxprFn {
                tracer: JaxTracer::default(),
                inner: comp.func.clone(),
            }
            .into_py(py)
        })
    }

    /// Helper function to retrieve operands for a binary operation.
    #[inline]
    fn visit_binary_op(&mut self, op: &BinaryOp) -> Result<(PyObject, PyObject), Error> {
        Ok((self.visit(&op.lhs)?, self.visit(&op.rhs)?))
    }

    /// Helper function to visit binary operations and apply the corresponding Jax method.
    #[inline]
    fn visit_binary_lax(&mut self, op: &BinaryOp, method: &str) -> Result<PyObject, Error> {
        let (lhs, rhs) = self.visit_binary_op(op)?;
        Python::with_gil(|py| self.lax.call_method1(py, method, (lhs, rhs))).map_err(Error::PyO3)
    }

    /// Helper function to visit unary operations and apply the corresponding Jax method.
    #[inline]
    fn visit_unary_lax(&mut self, op: &Noxpr, method: &str) -> Result<PyObject, Error> {
        let inner = self.visit(op)?;
        Python::with_gil(|py| self.lax.call_method1(py, method, (inner,))).map_err(Error::PyO3)
    }

    /// Wraps a `NoxprFn` into a callable Jax representation.
    fn visit_fn(&mut self, op: &NoxprFn) -> JaxNoxprFn {
        JaxNoxprFn {
            tracer: self.clone(),
            inner: Arc::new(op.clone()),
        }
    }
}

/// A Python callable wrapper for `NoxprFn` to be used in `Jax`.
#[pyo3::prelude::pyclass(weakref)]
#[derive(Clone)]
pub struct JaxNoxprFn {
    pub tracer: JaxTracer,
    pub inner: Arc<NoxprFn>,
}

#[pyo3::prelude::pymethods]
impl JaxNoxprFn {
    /// Executes the function by translating it to a `Jax` operation.
    #[allow(unused_variables)]
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(
        &mut self,
        py: Python<'_>,
        args: Bound<PyTuple>,
        kwargs: Option<Bound<PyDict>>,
    ) -> PyResult<PyObject> {
        for (i, arg) in self.inner.args.iter().enumerate() {
            let a = args.get_item(i)?;
            self.tracer.cache.insert(arg.id(), a.into());
        }

        let out = self
            .tracer
            .visit(&self.inner.inner)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(out)
    }

    #[allow(unused_variables)]
    #[pyo3(signature = (*args, **kwargs))]
    fn lower<'py>(
        &mut self,
        py: Python<'py>,
        args: Bound<PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let jax = py.import_bound("jax").unwrap();
        let jit_args = [("keep_unused", true)].into_py_dict_bound(py);
        let jit = jax.call_method("jit", (self.clone(),), Some(&jit_args))?;

        jit.call_method("lower", args, kwargs)
    }
}

impl Default for JaxTracer {
    fn default() -> Self {
        Self::new()
    }
}

/// Converts a `Literal` to a Jax numpy array.
fn literal_to_arr<T: ArrayElement + numpy::Element + bytemuck::Pod>(
    data: &Literal,
    shape: &SmallVec<[i64; 4]>,
    jnp: &PyObject,
) -> Result<PyObject, Error> {
    let buf = data.typed_buf::<T>()?;
    let shape = shape
        .iter()
        .map(|x| *x as usize)
        .collect::<SmallVec<[usize; 4]>>();
    Python::with_gil(|py| {
        let arr = numpy::PyArray::from_slice_bound(py, buf);
        let arr = arr.into_gil_ref().reshape(&shape[..]).unwrap();
        let arr = jnp.call_method1(py, "array", (&arr,))?;
        Ok(arr)
    })
}

/// Maps `ElementType` to Python `dtype` strings.
pub fn dtype(elem: &ElementType) -> Result<&'static str, Error> {
    match elem {
        ElementType::S8 => Ok("int8"),
        ElementType::S16 => Ok("int16"),
        ElementType::S32 => Ok("int32"),
        ElementType::S64 => Ok("int64"),
        ElementType::U8 => Ok("uint8"),
        ElementType::U16 => Ok("uint16"),
        ElementType::U32 => Ok("uint32"),
        ElementType::U64 => Ok("uint64"),
        ElementType::F32 => Ok("float32"),
        ElementType::F64 => Ok("float64"),
        ElementType::Pred => Ok("bool"),
        ElementType::C64 => todo!(),
        ElementType::C128 => todo!(),
        ElementType::F16 => todo!(),
        ElementType::Bf16 => todo!(),
    }
}

/// Executes a computational function and converts the result to a `Jax` operation.
pub fn call_comp_fn<T, R: ReprMonad<crate::Op>>(
    func: impl CompFn<T, R>,
    args: &[PyObject],
) -> Result<PyObject, Error> {
    let func = func.build_expr()?;
    let mut tracer = JaxTracer::new();
    for (py_arg, arg_expr) in args.iter().zip(func.args) {
        let arg_id = arg_expr.id();
        tracer.cache.insert(arg_id, py_arg.clone());
    }
    tracer.visit(&func.inner)
}

#[cfg(test)]
mod tests {
    use numpy::PyArrayLike0;
    use std::sync::Once;

    use crate::Scalar;

    use super::*;

    use pyo3::ffi;
    use std::ffi::{CStr, CString};

    // Like `pyo3::prepare_freethreaded_python` but with venv support.
    fn prepare_python() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            if std::env::var("VIRTUAL_ENV").unwrap_or_default().is_empty()
                || !std::env::var("PYTHONHOME").unwrap_or_default().is_empty()
            {
                // no virtual environment found
                pyo3::prepare_freethreaded_python();
                return;
            }

            let stdlib_path = env!("PYTHON_STDLIB_PATH");
            let purelib_path = env!("PYTHON_PURELIB_PATH");

            unsafe {
                if ffi::Py_IsInitialized() == 0 {
                    let mut config: ffi::PyConfig = std::mem::zeroed();
                    ffi::PyConfig_InitPythonConfig(&mut config);

                    config.module_search_paths_set = 2;
                    for path in [stdlib_path, purelib_path] {
                        let path_cstring = CString::new(path).unwrap();
                        let path_wchar =
                            ffi::Py_DecodeLocale(path_cstring.as_ptr(), std::ptr::null_mut());
                        let append_status = ffi::PyWideStringList_Append(
                            &mut config.module_search_paths,
                            path_wchar,
                        );
                        if append_status._type == ffi::_PyStatus_TYPE::_PyStatus_TYPE_ERROR {
                            panic!(
                                "Failed to append path to module search paths: {}",
                                CStr::from_ptr(append_status.err_msg).to_string_lossy()
                            );
                        }
                    }

                    // Initialize Python with the config
                    let init_status = ffi::Py_InitializeFromConfig(&mut config);
                    if init_status._type == ffi::_PyStatus_TYPE::_PyStatus_TYPE_ERROR {
                        panic!(
                            "Failed to initialize Python: {}",
                            CStr::from_ptr(init_status.err_msg).to_string_lossy()
                        );
                    }
                }
            }
        });
    }

    #[test]
    fn test_add() {
        prepare_python();
        let a: Scalar<f32> = 1.0f32.into();
        let b: Scalar<f32> = 2.0f32.into();
        let c = a + b;
        let mut tracer = JaxTracer::new();
        let o = tracer.visit(&c.inner).unwrap();
        Python::with_gil(|py| {
            let arr = o.extract::<PyArrayLike0<f32>>(py).unwrap();
            assert_eq!(arr.as_slice().unwrap(), &[3.0]);
        })
    }
}
