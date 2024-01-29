use std::{collections::HashMap, ops::Deref};

use pyo3::{PyObject, Python};
use smallvec::SmallVec;
use xla::{ArrayElement, ElementType, Literal};

use crate::{
    BinaryOp, CompFn, Error, Field, IntoOp, Noxpr, NoxprId, NoxprNode, Scalar, Tensor, TensorItem,
};

pub struct JaxTracer {
    lax: PyObject,
    jnp: PyObject,
    cache: HashMap<NoxprId, PyObject>,
}

impl JaxTracer {
    pub fn new() -> Self {
        Python::with_gil(|py| {
            let lax = py.import("jax.lax").unwrap().into();
            let jnp = py.import("jax.numpy").unwrap().into();
            Self {
                lax,
                jnp,
                cache: HashMap::new(),
            }
        })
    }

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
            NoxprNode::Param(_) => unimplemented!(),
            NoxprNode::Tuple(_) => todo!(),
            NoxprNode::Iota(i) => Python::with_gil(|py| {
                let size = i.shape.shape[i.dim];
                let dtype = dtype(&i.shape.element_type)?;
                self.lax
                    .call_method1(py, "iota", (dtype, size))
                    .map_err(Error::PyO3)
            })?,
            NoxprNode::Add(op) => self.visit_binary_lax(op, "add")?,
            NoxprNode::Sub(op) => self.visit_binary_lax(op, "sub")?,
            NoxprNode::Mul(op) => self.visit_binary_lax(op, "mul")?,
            NoxprNode::Div(op) => self.visit_binary_lax(op, "div")?,
            NoxprNode::And(op) => self.visit_binary_lax(op, "bitwise_and")?,
            NoxprNode::Or(op) => self.visit_binary_lax(op, "bitwise_or")?,
            NoxprNode::Dot(op) => self.visit_binary_lax(op, "dot")?,
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
            NoxprNode::Concat(c) => {
                let nodes = c
                    .nodes
                    .iter()
                    .map(|x| self.visit(x))
                    .collect::<Result<Vec<_>, _>>()?;
                Python::with_gil(|py| self.lax.call_method1(py, "concat", (nodes, c.dimension)))?
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
                Python::with_gil(|py| self.lax.call_method1(py, "broadcast", (expr, sizes, dims)))?
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
                        (
                            expr,
                            start_indices,
                            gather_dims,
                            g.slice_sizes.to_vec(),
                            true,
                            true,
                            "clip",
                        ),
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
                            (
                                expr,
                                s.start_indices.to_vec(),
                                s.stop_indices.to_vec(),
                                s.strides.to_vec(),
                            ),
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
            NoxprNode::Jax(o) => o.clone(),
        };
        self.cache.insert(id, op.clone());
        Ok(op)
    }

    #[inline]
    fn visit_binary_op(&mut self, op: &BinaryOp) -> Result<(PyObject, PyObject), Error> {
        Ok((self.visit(&op.lhs)?, self.visit(&op.rhs)?))
    }

    #[inline]
    fn visit_binary_lax(&mut self, op: &BinaryOp, method: &str) -> Result<PyObject, Error> {
        let (lhs, rhs) = self.visit_binary_op(op)?;
        Python::with_gil(|py| self.lax.call_method1(py, method, (lhs, rhs))).map_err(Error::PyO3)
    }

    #[inline]
    fn visit_unary_lax(&mut self, op: &Noxpr, method: &str) -> Result<PyObject, Error> {
        let inner = self.visit(op)?;
        Python::with_gil(|py| self.lax.call_method1(py, method, (inner,))).map_err(Error::PyO3)
    }
}

impl Default for JaxTracer {
    fn default() -> Self {
        Self::new()
    }
}

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
        let arr = numpy::PyArray::from_slice(py, buf);
        let arr = arr.reshape(&shape[..]).unwrap();
        let arr = jnp.call_method1(py, "array", (&arr,))?;
        Ok(arr)
    })
}

fn dtype(elem: &ElementType) -> Result<&'static str, Error> {
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

pub fn call_comp_fn<T, R: IntoOp>(
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

pub struct JaxDynField;

impl TensorItem for JaxDynField {
    type Item = Scalar<JaxDynField>;

    type Tensor<D> = Tensor<JaxDynField, D>
    where
        D: crate::TensorDim;

    type Dim = ();

    const ELEM: ElementType = ElementType::F32;

    fn from_op(op: Noxpr) -> Self::Item {
        Scalar::from_op(op)
    }
}

impl Field for JaxDynField {
    fn zero() -> Scalar<Self>
    where
        Self: Sized,
    {
        scalar(0.0, "float32")
    }

    fn one() -> Scalar<Self>
    where
        Self: Sized,
    {
        scalar(1.0, "float32")
    }

    fn two() -> Scalar<Self>
    where
        Self: Sized,
    {
        scalar(2.0, "float32")
    }
}

fn scalar(f: f32, dtype: &str) -> Scalar<JaxDynField> {
    let inner = Python::with_gil(|py| {
        let jax = py.import("jax").unwrap();
        let obj = jax.call_method1("numpy.array", (f, dtype)).unwrap().into();
        Noxpr::jax(obj)
    });
    crate::Scalar {
        inner,
        phantom: std::marker::PhantomData,
    }
}

#[cfg(test)]
mod tests {
    use numpy::PyArrayLike0;

    use crate::ConstantExt;

    use super::*;

    #[test]
    fn test_add() {
        pyo3::prepare_freethreaded_python();
        let a = 1.0f32.constant();
        let b = 2.0f32.constant();
        let c = a + b;
        let mut tracer = JaxTracer::new();
        let o = tracer.visit(&c.inner).unwrap();
        Python::with_gil(|py| {
            let arr = o.extract::<PyArrayLike0<f32>>(py).unwrap();
            assert_eq!(arr.as_slice().unwrap(), &[3.0]);
        })
    }
}
