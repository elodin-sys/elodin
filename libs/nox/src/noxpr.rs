use std::{
    collections::HashMap,
    iter,
    ops::{Add, Deref, Div, Mul, Neg, Sub},
    sync::Arc,
};

use itertools::Itertools;
use smallvec::{smallvec, SmallVec};
use xla::{ArrayElement, ElementType, NativeType, XlaBuilder, XlaOp, XlaOpRef};

use crate::{
    CompFn, DefaultMap, DefaultMappedDim, Error, IntoOp, MapDim, Tensor, TensorDim, TensorItem,
};

#[derive(Debug)]
pub enum NoxprNode {
    // Params / Variables
    Param(ParamExpr),

    // Tuples
    Tuple(Vec<Noxpr>),
    GetTupleElement(GetTupleElement),

    // Constants
    Constant(Constant),
    Iota(Iota),

    // Element Wise Binary Ops
    Add(BinaryOp),
    Sub(BinaryOp),
    Mul(BinaryOp),
    Div(BinaryOp),
    And(BinaryOp),
    Or(BinaryOp),

    // Matrix Multiplication
    Dot(BinaryOp),
    DotGeneral(DotGeneral),

    // Unary Ops
    Sqrt(Noxpr),
    Neg(Noxpr),
    Log(Noxpr),

    // Nary ops
    Concat(Concat),

    // Reshape
    Reshape(Reshape),
    Broadcast(Broadcast),
    BroadcastInDim(BroadcastInDim),
    Transpose(Transpose),

    // Slice
    Gather(Gather),
    Slice(Slice),
    DynamicSlice(DynamicSlice),
    DynamicUpdateSlice(DynamicUpdateSlice),

    #[cfg(feature = "jax")]
    Jax(pyo3::PyObject),
}

pub struct Constant {
    pub data: xla::Literal, // NOTE: it might make more sense to use the xla independent store below
    // pub data: SmallVec<[u8; size_of::<f64>()]>,
    pub ty: ArrayTy,
}

impl std::fmt::Debug for Constant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Constant").field("ty", &self.ty).finish()
    }
}

pub enum NoxprTy {
    Tuple(Vec<NoxprTy>),
    ArrayTy(ArrayTy),
}

#[derive(Debug, Clone)]
pub struct ArrayTy {
    pub element_type: ElementType,
    pub shape: SmallVec<[i64; 4]>,
}

impl ArrayTy {
    pub fn new(element_type: ElementType, shape: SmallVec<[i64; 4]>) -> Self {
        Self {
            element_type,
            shape,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Iota {
    pub shape: ArrayTy,
    pub dim: usize,
}

#[derive(Debug)]
pub struct BinaryOp {
    pub lhs: Noxpr,
    pub rhs: Noxpr,
}

impl BinaryOp {
    fn shape(&self) -> Option<SmallVec<[i64; 4]>> {
        let lhs_shape = self.lhs.shape()?;
        let rhs_shape = self.rhs.shape()?;
        broadcast_dims(&lhs_shape, &rhs_shape)
    }
}

pub(crate) fn broadcast_dims(lhs: &[i64], rhs: &[i64]) -> Option<SmallVec<[i64; 4]>> {
    // logic from https://numpy.org/doc/stable/user/basics.broadcasting.html extended with extra rule for dynamic arrays
    let lhs = lhs.iter().rev().copied();
    let rhs = rhs.iter().rev().copied();
    lhs.zip_longest(rhs)
        .map(|x| match x {
            itertools::EitherOrBoth::Both(lhs, rhs) if lhs == rhs && lhs == -1 => None,
            itertools::EitherOrBoth::Both(lhs, rhs) if lhs == rhs => Some(lhs),
            itertools::EitherOrBoth::Both(lhs, rhs) if lhs == 1 || rhs == 1 => Some(lhs.max(rhs)),
            itertools::EitherOrBoth::Both(_, _) => None,
            itertools::EitherOrBoth::Left(lhs) => Some(lhs),
            itertools::EitherOrBoth::Right(rhs) => Some(rhs),
        })
        .rev()
        .collect()
}

#[derive(Debug)]
pub struct DotGeneral {
    pub lhs: Noxpr,
    pub rhs: Noxpr,
    pub dimensions: DotDimensionNums,
}

impl DotGeneral {
    fn shape(&self) -> Option<SmallVec<[i64; 4]>> {
        let lhs = self.lhs.shape()?;
        let rhs = self.lhs.shape()?;
        let DotDimensionNums {
            lhs_contracting_dimensions: lhs_contracting,
            rhs_contracting_dimensions: rhs_contracting,
            lhs_batch_dimensions: lhs_batch,
            rhs_batch_dimensions: rhs_batch,
        } = self.dimensions.clone();
        let batch_shape = lhs_batch
            .iter()
            .map(|i| lhs[*i as usize])
            .collect::<SmallVec<[i64; 4]>>();
        let lhs_contract_or_batch = lhs_contracting
            .iter()
            .chain(lhs_batch.iter())
            .cloned()
            .collect::<SmallVec<[i64; 4]>>();
        let lhs_tensored_shape = lhs
            .iter()
            .enumerate()
            .filter(|(i, _)| !lhs_contract_or_batch.contains(&(*i as i64)))
            .map(|(_, v)| *v)
            .collect::<SmallVec<[i64; 4]>>();
        let rhs_contract_or_batch = rhs_contracting
            .iter()
            .chain(rhs_batch.iter())
            .cloned()
            .collect::<SmallVec<[i64; 4]>>();
        let rhs_tensored_shape = rhs
            .iter()
            .enumerate()
            .filter(|(i, _)| !rhs_contract_or_batch.contains(&(*i as i64)))
            .map(|(_, v)| *v)
            .collect::<SmallVec<[i64; 4]>>();
        let result = batch_shape
            .into_iter()
            .chain(lhs_tensored_shape)
            .chain(rhs_tensored_shape)
            .collect::<SmallVec<[i64; 4]>>();
        Some(result)
    }
}

#[derive(Debug, Clone, Default)]
pub struct DotDimensionNums {
    pub lhs_contracting_dimensions: SmallVec<[i64; 2]>,
    pub rhs_contracting_dimensions: SmallVec<[i64; 2]>,
    pub lhs_batch_dimensions: SmallVec<[i64; 2]>,
    pub rhs_batch_dimensions: SmallVec<[i64; 2]>,
}

impl DotDimensionNums {
    fn to_xla(&self) -> xla::DotDimensionNumbers {
        let DotDimensionNums {
            lhs_contracting_dimensions,
            rhs_contracting_dimensions,
            lhs_batch_dimensions,
            rhs_batch_dimensions,
        } = self;
        let mut dims = xla::DotDimensionNumbers::default();
        lhs_contracting_dimensions
            .iter()
            .cloned()
            .for_each(|d| dims.add_lhs_contracting_dimensions(d));
        rhs_contracting_dimensions
            .iter()
            .cloned()
            .for_each(|d| dims.add_rhs_contracting_dimensions(d));
        lhs_batch_dimensions
            .iter()
            .cloned()
            .for_each(|d| dims.add_lhs_batch_dimensions(d));
        rhs_batch_dimensions
            .iter()
            .cloned()
            .for_each(|d| dims.add_rhs_batch_dimensions(d));
        dims
    }
}

#[derive(Debug)]
pub struct Concat {
    pub nodes: Vec<Noxpr>,
    pub dimension: usize,
}

#[derive(Debug)]
pub struct Slice {
    pub expr: Noxpr,
    pub start_indices: SmallVec<[i64; 4]>,
    pub stop_indices: SmallVec<[i64; 4]>,
    pub strides: SmallVec<[i64; 4]>,
}

#[derive(Debug)]
pub struct DynamicSlice {
    pub expr: Noxpr,
    pub start_indices: Vec<Noxpr>,
    pub size_indices: SmallVec<[i64; 4]>,
}

#[derive(Debug)]
pub struct Reshape {
    pub expr: Noxpr,
    pub new_sizes: SmallVec<[i64; 4]>,
}

#[derive(Debug)]
pub struct Broadcast {
    pub expr: Noxpr,
    pub sizes: SmallVec<[i64; 4]>,
}

#[derive(Debug)]
pub struct BroadcastInDim {
    pub expr: Noxpr,
    pub sizes: SmallVec<[i64; 4]>,
    pub broadcast_dims: SmallVec<[i64; 4]>,
}

#[derive(Debug)]
pub struct Transpose {
    pub expr: Noxpr,
    pub permutation: SmallVec<[i64; 4]>,
}

#[derive(Debug)]
pub struct Gather {
    pub expr: Noxpr,
    pub indices: Noxpr,
    pub offset_dims: SmallVec<[i64; 4]>,
    pub collapsed_slice_dims: SmallVec<[i64; 4]>,
    pub start_index_map: SmallVec<[i64; 4]>,
    pub slice_sizes: SmallVec<[i64; 4]>,
    pub index_vector_dim: i64,
}

#[derive(Debug)]
pub struct DynamicUpdateSlice {
    pub expr: Noxpr,
    pub start_indicies: Vec<Noxpr>,
    pub update: Noxpr,
}

#[derive(Debug)]
pub struct GetTupleElement {
    pub expr: Noxpr,
    pub index: usize,
}

#[derive(Debug, Clone)]
pub struct Noxpr {
    pub node: Arc<NoxprNode>,
    pub backtrace: Arc<std::backtrace::Backtrace>,
}

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub struct NoxprId(usize);

impl Noxpr {
    pub fn new(node: NoxprNode) -> Self {
        Self {
            backtrace: Arc::new(std::backtrace::Backtrace::capture()),
            node: Arc::new(node),
        }
    }

    pub fn parameter(number: i64, ty: ArrayTy, name: String) -> Self {
        Self::new(NoxprNode::Param(ParamExpr { ty, number, name }))
    }

    pub fn dot(self, rhs: &Noxpr) -> Self {
        Self::new(NoxprNode::Dot(BinaryOp {
            lhs: self,
            rhs: rhs.clone(),
        }))
    }

    fn dot_general(self, rhs: Noxpr, dimensions: DotDimensionNums) -> Self {
        Self::new(NoxprNode::DotGeneral(DotGeneral {
            lhs: self,
            rhs,
            dimensions,
        }))
    }

    pub fn log(self) -> Self {
        Self::new(NoxprNode::Log(self))
    }

    pub fn sqrt(self) -> Self {
        Self::new(NoxprNode::Sqrt(self))
    }

    pub fn constant(data: xla::Literal, ty: ArrayTy) -> Self {
        Self::new(NoxprNode::Constant(Constant { data, ty }))
    }

    pub fn tuple(nodes: Vec<Noxpr>) -> Self {
        Self::new(NoxprNode::Tuple(nodes))
    }

    pub fn concat_in_dim(nodes: Vec<Noxpr>, dimension: usize) -> Self {
        Self::new(NoxprNode::Concat(Concat { nodes, dimension }))
    }

    pub fn slice(
        self,
        start_indices: SmallVec<[i64; 4]>,
        stop_indices: SmallVec<[i64; 4]>,
        strides: SmallVec<[i64; 4]>,
    ) -> Self {
        Self::new(NoxprNode::Slice(Slice {
            expr: self,
            start_indices,
            stop_indices,
            strides,
        }))
    }

    pub fn broadcast_in_dim(
        self,
        sizes: SmallVec<[i64; 4]>,
        broadcast_dims: SmallVec<[i64; 4]>,
    ) -> Self {
        Self::new(NoxprNode::BroadcastInDim(BroadcastInDim {
            expr: self,
            sizes,
            broadcast_dims,
        }))
    }

    pub fn transpose(self, permuation: SmallVec<[i64; 4]>) -> Self {
        Self::new(NoxprNode::Transpose(Transpose {
            expr: self,
            permutation: permuation,
        }))
    }

    pub fn or(self, rhs: Noxpr) -> Self {
        Self::new(NoxprNode::Or(BinaryOp { lhs: self, rhs }))
    }

    pub fn and(self, rhs: Noxpr) -> Self {
        Self::new(NoxprNode::And(BinaryOp { lhs: self, rhs }))
    }

    pub fn reshape(self, new_sizes: SmallVec<[i64; 4]>) -> Self {
        Self::new(NoxprNode::Reshape(Reshape {
            expr: self,
            new_sizes,
        }))
    }

    pub fn gather(
        self,
        indices: Noxpr,
        offset_dims: SmallVec<[i64; 4]>,
        collapsed_slice_dims: SmallVec<[i64; 4]>,
        start_index_map: SmallVec<[i64; 4]>,
        slice_sizes: SmallVec<[i64; 4]>,
        index_vector_dim: i64,
    ) -> Self {
        Self::new(NoxprNode::Gather(Gather {
            expr: self,
            indices,
            offset_dims,
            collapsed_slice_dims,
            start_index_map,
            slice_sizes,
            index_vector_dim,
        }))
    }

    pub fn iota(shape: ArrayTy, dim: usize) -> Self {
        Self::new(NoxprNode::Iota(Iota { shape, dim }))
    }

    pub fn get_tuple_element(&self, index: usize) -> Self {
        Self::new(NoxprNode::GetTupleElement(GetTupleElement {
            expr: self.clone(),
            index,
        }))
    }

    pub fn shape(&self) -> Option<SmallVec<[i64; 4]>> {
        match self.deref() {
            NoxprNode::Constant(c) => Some(c.ty.shape.clone()),
            NoxprNode::Param(p) => Some(p.ty.shape.clone()),
            NoxprNode::Add(ref b)
            | NoxprNode::Sub(ref b)
            | NoxprNode::Div(ref b)
            | NoxprNode::Mul(ref b)
            | NoxprNode::And(ref b)
            | NoxprNode::Or(ref b) => b.shape(),

            NoxprNode::Dot(b) => {
                let lhs_shape = b.lhs.shape()?;
                let rhs_shape = b.rhs.shape()?;
                match (lhs_shape.len(), rhs_shape.len()) {
                    (1, 1) => Some(SmallVec::new()),
                    (2, 1) => Some(smallvec![lhs_shape[0]]),
                    (2, 2) => Some(smallvec![lhs_shape[0], rhs_shape[1]]),
                    _ => None,
                }
            }
            NoxprNode::DotGeneral(s) => s.shape(),
            NoxprNode::Sqrt(expr) | NoxprNode::Neg(expr) => expr.shape(),

            NoxprNode::Concat(concat) => {
                let shapes = concat
                    .nodes
                    .iter()
                    .map(Noxpr::shape)
                    .collect::<Option<Vec<_>>>()?;
                let mut shape = shapes.first()?.clone();
                let dim = concat.dimension;
                // TODO(sphw): ensure that all shapes are the same, except for a particular dim
                if shapes.iter().any(|s| s.len() != dim) {
                    return None;
                }
                if concat.dimension >= dim {
                    return None;
                }
                let new_len = shapes.iter().map(|s| s[concat.dimension]).fold(0, |xs, x| {
                    if x == -1 || xs == -1 {
                        return -1;
                    }
                    xs + x
                });
                shape[dim] = new_len;
                Some(shape)
            }
            NoxprNode::Slice(slice) => {
                if slice.start_indices.len() != slice.stop_indices.len() {
                    return None;
                }
                slice
                    .start_indices
                    .iter()
                    .zip(slice.stop_indices.iter())
                    .zip(slice.strides.iter())
                    .map(|((start, stop), stride)| {
                        stop.checked_sub(*start)
                            .and_then(|l| l.checked_div(*stride))
                    })
                    .collect()
            }
            NoxprNode::DynamicSlice(dynamic_slice) => Some(dynamic_slice.size_indices.clone()),
            NoxprNode::Reshape(reshape) => Some(reshape.new_sizes.clone()),
            NoxprNode::Tuple(_) => None,
            NoxprNode::Log(l) => l.shape(),
            NoxprNode::Broadcast(b) => Some(b.sizes.clone()),
            NoxprNode::BroadcastInDim(b) => Some(b.sizes.clone()),
            NoxprNode::Transpose(t) => {
                let shape = self.shape()?;
                let new_shape = t
                    .permutation
                    .iter()
                    .map(|dim| shape[*dim as usize])
                    .collect();
                Some(new_shape)
            }
            NoxprNode::Gather(gather) => {
                let indices_shape = gather.indices.shape()?;
                let output_rank = gather.offset_dims.len() + indices_shape.len() - 1;
                let index_vector_dim = indices_shape.len() - 1;
                let mut expanded_indices_shape = indices_shape.clone();
                expanded_indices_shape.remove(index_vector_dim);
                let mut expanded_indices_shape_iter = expanded_indices_shape.iter().copied();
                let mut adjusted_slice_sizes = gather
                    .slice_sizes
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !gather.collapsed_slice_dims.contains(&(*i as i64)))
                    .map(|(_, x)| *x);
                Some(
                    (0..output_rank)
                        .filter_map(|i| {
                            if gather.offset_dims.contains(&(i as i64)) {
                                adjusted_slice_sizes.next()
                            } else {
                                expanded_indices_shape_iter.next()
                            }
                        })
                        .collect(),
                )
            }
            NoxprNode::Iota(i) => Some(i.shape.shape.clone()),
            NoxprNode::DynamicUpdateSlice(d) => d.expr.shape(),
            NoxprNode::GetTupleElement(g) => {
                let NoxprNode::Tuple(elems) = g.expr.deref() else {
                    return None;
                };
                elems.get(g.index)?.shape()
            }
            #[cfg(feature = "jax")]
            NoxprNode::Jax(o) => pyo3::Python::with_gil(|py| {
                let shape = o.getattr(py, "shape").ok()?.extract::<Vec<i64>>(py).ok()?;
                Some(SmallVec::from_vec(shape))
            }),
        }
    }

    pub fn dynamic_update_slice(&self, start_indicies: Vec<Noxpr>, update: Noxpr) -> Noxpr {
        Noxpr::new(NoxprNode::DynamicUpdateSlice(DynamicUpdateSlice {
            expr: self.clone(),
            start_indicies,
            update,
        }))
    }

    #[cfg(feature = "jax")]
    pub fn jax(py: pyo3::PyObject) -> Noxpr {
        Noxpr::new(NoxprNode::Jax(py))
    }

    pub fn id(&self) -> NoxprId {
        NoxprId(Arc::as_ptr(&self.node) as usize)
    }

    pub fn name(&self) -> &'static str {
        match self.deref() {
            NoxprNode::Param(_) => "Param",
            NoxprNode::Tuple(_) => "Tuple",
            NoxprNode::GetTupleElement(_) => "GetTupleElement",
            NoxprNode::Constant(_) => "Constant",
            NoxprNode::Iota(_) => todo!(),
            NoxprNode::Add(_) => "Add",
            NoxprNode::Sub(_) => "Sub",
            NoxprNode::Mul(_) => "Mul",
            NoxprNode::Div(_) => "Div",
            NoxprNode::And(_) => "And",
            NoxprNode::Or(_) => "Or",
            NoxprNode::Dot(_) => "Dot",
            NoxprNode::DotGeneral(_) => "DotGeneral",
            NoxprNode::Sqrt(_) => "Sqrt",
            NoxprNode::Neg(_) => "Neg",
            NoxprNode::Log(_) => "Log",
            NoxprNode::Concat(_) => "Concat",
            NoxprNode::Reshape(_) => "Reshape",
            NoxprNode::Broadcast(_) => "Broadcast",
            NoxprNode::BroadcastInDim(_) => "BroadcastInDim",
            NoxprNode::Transpose(_) => "Transpose",
            NoxprNode::Gather(_) => "Gather",
            NoxprNode::Slice(_) => "Slice",
            NoxprNode::DynamicSlice(_) => "DynamicSlice",
            NoxprNode::DynamicUpdateSlice(_) => "DynamicUpdateSlice",
            NoxprNode::Jax(_) => "Jax",
        }
    }
}

impl Deref for Noxpr {
    type Target = NoxprNode;

    fn deref(&self) -> &Self::Target {
        &self.node
    }
}

#[derive(Debug)]
pub struct ParamExpr {
    pub number: i64,
    pub name: String,
    pub ty: ArrayTy,
}

impl Neg for Noxpr {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(NoxprNode::Neg(self))
    }
}

macro_rules! impl_binary_op {
    ($trait:tt, $trait_fn:tt, $variant:tt) => {
        impl $trait for Noxpr {
            type Output = Noxpr;

            fn $trait_fn(self, rhs: Self) -> Self::Output {
                Noxpr::new(NoxprNode::$variant(BinaryOp { lhs: self, rhs }))
            }
        }
    };
}

impl_binary_op!(Add, add, Add);
impl_binary_op!(Mul, mul, Mul);
impl_binary_op!(Div, div, Div);
impl_binary_op!(Sub, sub, Sub);

pub struct XlaTracer {
    builder: XlaBuilder,
    cache: HashMap<NoxprId, XlaOp>,
}

impl XlaTracer {
    pub fn new(name: &str) -> Self {
        Self {
            builder: XlaBuilder::new(name),
            cache: HashMap::new(),
        }
    }

    pub fn visit(&mut self, expr: &Noxpr) -> Result<XlaOp, Error> {
        let id = expr.id();
        if let Some(op) = self.cache.get(&id) {
            return Ok(op.clone());
        }

        let op = match expr.deref() {
            NoxprNode::Constant(c) => self.builder.constant_literal(&c.data)?.reshape(&c.ty.shape),
            NoxprNode::Param(p) => {
                self.builder
                    .parameter(p.number, p.ty.element_type, &p.ty.shape, &p.name)?
            }
            NoxprNode::Add(b) => {
                let (lhs, rhs) = self.visit_binary_op(b)?;
                lhs + rhs
            }
            NoxprNode::Sub(b) => {
                let (lhs, rhs) = self.visit_binary_op(b)?;
                lhs - rhs
            }
            NoxprNode::Div(b) => {
                let (lhs, rhs) = self.visit_binary_op(b)?;
                lhs / rhs
            }
            NoxprNode::Mul(b) => {
                let (lhs, rhs) = self.visit_binary_op(b)?;
                lhs.mul(rhs)
            }
            NoxprNode::Dot(b) => {
                let (lhs, rhs) = self.visit_binary_op(b)?;
                lhs.dot(&rhs)
            }

            NoxprNode::DotGeneral(DotGeneral {
                lhs,
                rhs,
                dimensions,
            }) => {
                let lhs = self.visit(lhs)?;
                let rhs = self.visit(rhs)?;
                lhs.dot_general(&rhs, dimensions.to_xla())
            }
            NoxprNode::And(b) => {
                let (lhs, rhs) = self.visit_binary_op(b)?;
                lhs.and(&rhs)
            }
            NoxprNode::Or(b) => {
                let (lhs, rhs) = self.visit_binary_op(b)?;
                lhs.or(&rhs)
            }
            NoxprNode::Sqrt(expr) => {
                let expr = self.visit(expr)?;
                expr.sqrt()
            }
            NoxprNode::Log(expr) => {
                let expr = self.visit(expr)?;
                expr.log()
            }
            NoxprNode::Neg(expr) => {
                let expr = self.visit(expr)?;
                expr.neg()
            }
            NoxprNode::Concat(concat) => {
                let ops = concat
                    .nodes
                    .iter()
                    .map(|n| self.visit(n))
                    .collect::<Result<Vec<_>, _>>()?;
                let ops = ops.iter().map(XlaOp::as_ref).collect::<Vec<_>>();
                self.builder.concat_in_dim(&ops, concat.dimension as i64)
            }
            NoxprNode::Slice(slice) => {
                let op = self.visit(&slice.expr)?;
                op.slice(&slice.start_indices, &slice.stop_indices, &slice.strides)
            }
            NoxprNode::DynamicSlice(dynamic) => {
                let op = self.visit(&dynamic.expr)?;
                let start_indices = dynamic
                    .start_indices
                    .iter()
                    .map(|expr| self.visit(expr))
                    .collect::<Result<Vec<_>, _>>()?;
                let start_indices = start_indices.iter().map(XlaOp::as_ref).collect::<Vec<_>>();
                op.dynamic_slice(&start_indices, &dynamic.size_indices)
            }
            NoxprNode::Reshape(reshape) => {
                let op = self.visit(&reshape.expr)?;
                op.reshape(&reshape.new_sizes)
            }
            NoxprNode::Tuple(tuple) => {
                let ops = tuple
                    .iter()
                    .map(|n| self.visit(n))
                    .collect::<Result<Vec<_>, _>>()?;
                let ops = ops.iter().map(XlaOp::as_ref).collect::<Vec<_>>();
                self.builder.tuple(&ops)
            }
            NoxprNode::GetTupleElement(g) => {
                let op = self.visit(&g.expr)?;
                op.get_tuple_element(g.index as i64)
            }
            NoxprNode::Broadcast(b) => {
                let op = self.visit(&b.expr)?;
                op.broadcast(&b.sizes)
            }
            NoxprNode::BroadcastInDim(b) => {
                let op = self.visit(&b.expr)?;
                op.broadcast_in_dim(&b.sizes, &b.broadcast_dims)
            }
            NoxprNode::Transpose(t) => {
                let op = self.visit(&t.expr)?;
                op.transpose(&t.permutation)
            }
            NoxprNode::Gather(g) => {
                let op = self.visit(&g.expr)?;
                let indices = self.visit(&g.indices)?;
                op.gather(
                    &indices,
                    &g.offset_dims,
                    &g.collapsed_slice_dims,
                    &g.start_index_map,
                    &g.slice_sizes,
                    g.index_vector_dim,
                )
            }
            NoxprNode::Iota(i) => {
                self.builder
                    .iota(&i.shape.shape, i.shape.element_type, i.dim as i64)
            }
            NoxprNode::DynamicUpdateSlice(d) => {
                let inner = self.visit(&d.expr)?;
                let update = self.visit(&d.update)?;
                let start = d
                    .start_indicies
                    .iter()
                    .map(|expr| self.visit(expr))
                    .collect::<Result<Vec<_>, Error>>()?;
                let start = start
                    .iter()
                    .map(|op| op.as_ref())
                    .collect::<SmallVec<[XlaOpRef<'_>; 4]>>();
                inner.dynamic_update_slice(&update, &start)
            }
            NoxprNode::Jax(_) => {
                unimplemented!()
            }
        };
        self.cache.insert(id, op.clone());
        Ok(op)
    }

    #[inline]
    fn visit_binary_op(&mut self, op: &BinaryOp) -> Result<(XlaOp, XlaOp), Error> {
        Ok((self.visit(&op.lhs)?, self.visit(&op.rhs)?))
    }
}

pub struct NoxprFn {
    pub args: Vec<Noxpr>,
    pub inner: Noxpr,
}

impl NoxprFn {
    pub fn new(args: Vec<Noxpr>, inner: Noxpr) -> Self {
        Self { args, inner }
    }

    pub fn build(&self, name: &str) -> Result<XlaOp, Error> {
        let mut tracer = XlaTracer::new(name);
        for a in self.args.iter() {
            tracer.visit(a)?;
        }
        tracer.visit(&self.inner)
    }
}

impl Noxpr {
    pub fn vmap_with_axis(
        func: NoxprFn,
        in_axis: &[usize],
        args: &[Noxpr],
    ) -> Result<Noxpr, Error> {
        if in_axis.len() != args.len() {
            dbg!(in_axis);
            dbg!(args);
            return Err(Error::VmapInAxisMismatch);
        }
        let shape = args
            .first()
            .ok_or(Error::VmapArgsEmpty)?
            .shape()
            .ok_or(Error::UnbatchableArgument)?;
        let mut tracer = BatchTracer::new(BatchAxis::Mapped {
            index: in_axis[0],
            size: shape[in_axis[0]] as usize,
        });
        for ((arg, axis), arg_expr) in args.iter().zip(in_axis).zip(func.args) {
            let arg_id = arg_expr.id();
            let shape = arg.shape().ok_or(Error::UnbatchableArgument)?;
            let batch_axis = BatchAxis::Mapped {
                index: *axis,
                size: shape[*axis] as usize,
            };
            tracer.cache.insert(
                arg_id,
                BatchedExpr {
                    inner: arg.clone(),
                    batch_axis,
                },
            );
        }
        let expr = tracer.visit(&func.inner)?;
        let expr = expr
            .move_batch_axis(tracer.out_axis)
            .ok_or(Error::UnbatchableArgument)?;
        Ok(expr.inner)
    }
}

impl<T: TensorItem, D: TensorDim + DefaultMap> Tensor<T, D, crate::Op> {
    pub fn vmap<O: TensorItem + IntoOp>(
        &self,
        func: impl CompFn<(T::Tensor<<D::DefaultMapDim as MapDim<D>>::Item>,), O>,
    ) -> Result<Tensor<O, DefaultMappedDim<D>, crate::Op>, Error> {
        self.vmap_with_dim::<D::DefaultMapDim, O>(func)
    }

    pub fn vmap_with_dim<MDim: MapDim<D>, O: TensorItem + IntoOp>(
        &self,
        func: impl CompFn<(T::Tensor<MDim::Item>,), O>,
    ) -> Result<Tensor<O, MDim::MappedDim, crate::Op>, Error> {
        let func = func.build_expr()?;
        let inner =
            Noxpr::vmap_with_axis(func, &[MDim::MAPPED_DIM], std::slice::from_ref(&self.inner))?;
        Ok(Tensor {
            inner,
            phantom: std::marker::PhantomData,
        })
    }
}

pub struct BatchTracer {
    cache: HashMap<NoxprId, BatchedExpr>,
    out_axis: BatchAxis,
}

#[derive(Clone, Debug)]
pub struct BatchedExpr {
    inner: Noxpr,
    batch_axis: BatchAxis,
}

impl BatchedExpr {
    fn map_expr(self, func: impl FnOnce(Noxpr) -> Noxpr) -> Self {
        BatchedExpr {
            inner: func(self.inner),
            batch_axis: self.batch_axis,
        }
    }

    fn move_batch_axis(self, dest: BatchAxis) -> Option<Self> {
        let BatchAxis::Mapped {
            index: dest_axis,
            size: dest_size,
        } = dest
        else {
            return Some(self);
        };
        match self.batch_axis {
            BatchAxis::NotMapped => {
                let mut shape = self.inner.shape()?;
                let broadcast_dims = if shape.is_empty() {
                    smallvec![]
                } else {
                    smallvec![dest_axis as i64]
                };
                shape.insert(dest_axis, dest_size as i64);
                let inner = self.inner.broadcast_in_dim(shape, broadcast_dims);
                Some(Self {
                    inner,
                    batch_axis: dest,
                })
            }
            BatchAxis::Mapped { index, .. } if index == dest_axis => self.into(),
            BatchAxis::Mapped { index, .. } => {
                let shape = self.inner.shape()?;
                let mut perm = (0..shape.len())
                    .filter(|i| *i != index)
                    .map(|x| x as i64)
                    .collect::<SmallVec<[i64; 4]>>();
                perm.insert(dest_axis, index as i64);
                let inner = self.inner.transpose(perm);
                Some(Self {
                    inner,
                    batch_axis: dest,
                })
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum BatchAxis {
    NotMapped,
    Mapped { index: usize, size: usize },
}

impl BatchTracer {
    pub fn new(out_axis: BatchAxis) -> Self {
        Self {
            cache: HashMap::default(),
            out_axis,
        }
    }

    pub fn visit(&mut self, expr: &Noxpr) -> Result<BatchedExpr, Error> {
        let id = expr.id();
        if let Some(op) = self.cache.get(&id) {
            return Ok(op.clone());
        }
        let op = match expr.deref() {
            NoxprNode::Constant(_) | NoxprNode::Param(_) => BatchedExpr {
                inner: expr.clone(),
                batch_axis: BatchAxis::NotMapped,
            },
            NoxprNode::Tuple(inner) => {
                let mut exprs = Vec::with_capacity(inner.len());
                let mut batch_axis = BatchAxis::NotMapped;
                for e in inner {
                    let mapped = self.visit(e)?;
                    exprs.push(mapped.inner);
                    batch_axis = mapped.batch_axis;
                }
                let inner = Noxpr::tuple(exprs);
                BatchedExpr { inner, batch_axis }
            }
            NoxprNode::Add(b) => self.visit_binary_op(b, Noxpr::add)?,
            NoxprNode::Sub(b) => self.visit_binary_op(b, Noxpr::sub)?,
            NoxprNode::Mul(b) => self.visit_binary_op(b, Noxpr::mul)?,
            NoxprNode::Div(b) => self.visit_binary_op(b, Noxpr::div)?,
            NoxprNode::And(b) => self.visit_binary_op(b, Noxpr::and)?,
            NoxprNode::Or(b) => self.visit_binary_op(b, Noxpr::or)?,
            NoxprNode::Sqrt(e) => self.visit_unary_op(e, Noxpr::sqrt)?,
            NoxprNode::Neg(e) => self.visit_unary_op(e, Noxpr::neg)?,
            NoxprNode::Log(e) => self.visit_unary_op(e, Noxpr::log)?,
            NoxprNode::Concat(c) => {
                let nodes = c
                    .nodes
                    .iter()
                    .map(|n| self.visit(n))
                    .collect::<Result<Vec<_>, Error>>()?;
                let size = nodes
                    .iter()
                    .find_map(|n| match n.batch_axis {
                        BatchAxis::NotMapped => None,
                        BatchAxis::Mapped { size, .. } => Some(size),
                    })
                    .ok_or(Error::UnbatchableArgument)?;
                let nodes = nodes
                    .into_iter()
                    .map(|n| {
                        Ok(n.move_batch_axis(BatchAxis::Mapped { index: 0, size })
                            .ok_or(Error::UnbatchableArgument)?
                            .inner)
                    })
                    .collect::<Result<_, Error>>()?;

                BatchedExpr {
                    inner: Noxpr::concat_in_dim(nodes, c.dimension + 1),
                    batch_axis: BatchAxis::Mapped { index: 0, size },
                }
            }
            NoxprNode::DotGeneral(d) => {
                self.visit_dot_general(&d.lhs, &d.rhs, d.dimensions.clone())?
            }
            NoxprNode::Dot(d) => self.visit_dot_general(
                &d.lhs,
                &d.rhs,
                DotDimensionNums {
                    lhs_contracting_dimensions: smallvec![1],
                    rhs_contracting_dimensions: smallvec![0],
                    ..Default::default()
                },
            )?,
            NoxprNode::Slice(s) => {
                let expr = self.visit(&s.expr)?;
                match expr.batch_axis {
                    BatchAxis::NotMapped => BatchedExpr {
                        inner: expr.inner.slice(
                            s.start_indices.clone(),
                            s.stop_indices.clone(),
                            s.strides.clone(),
                        ),
                        batch_axis: BatchAxis::NotMapped,
                    },
                    BatchAxis::Mapped { index, size } => {
                        let mut start_indices = s.start_indices.clone();
                        let mut stop_indices = s.stop_indices.clone();
                        let mut strides = s.strides.clone();
                        start_indices.insert(index, 0);
                        stop_indices.insert(index, size as i64);
                        strides.insert(index, 1);
                        BatchedExpr {
                            inner: expr.inner.slice(start_indices, stop_indices, strides),
                            batch_axis: BatchAxis::Mapped { index, size },
                        }
                    }
                }
            }
            NoxprNode::DynamicSlice(_) => {
                todo!("add when we add gather, as dynamic slice is a special case of gather")
            }
            NoxprNode::Reshape(r) => {
                let expr = self.visit(&r.expr)?;
                let BatchAxis::Mapped { size, .. } = self.out_axis else {
                    return Err(Error::UnbatchableArgument);
                };
                match &expr.batch_axis {
                    BatchAxis::NotMapped => BatchedExpr {
                        inner: expr.inner.reshape(r.new_sizes.clone()),
                        batch_axis: BatchAxis::NotMapped,
                    }
                    .move_batch_axis(self.out_axis.clone())
                    .ok_or(Error::UnbatchableArgument)?,
                    BatchAxis::Mapped {
                        size: ref batch_size,
                        ..
                    } => {
                        let batch_size = *batch_size;
                        let expr = expr
                            .move_batch_axis(BatchAxis::Mapped { index: 0, size })
                            .ok_or(Error::UnbatchableArgument)?;
                        let shape = expr.inner.shape().ok_or(Error::UnbatchableArgument)?;
                        //let x = shape.first().ok_or(Error::UnbatchableArgument)?;
                        let new_sizes = shape
                            .first()
                            .cloned()
                            .into_iter()
                            .chain(r.new_sizes.iter().cloned())
                            .collect();
                        BatchedExpr {
                            inner: expr.inner.reshape(new_sizes),
                            batch_axis: BatchAxis::Mapped {
                                index: 0,
                                size: batch_size, // NOTE: not sure this is the correct size
                            },
                        }
                    }
                }
            }
            NoxprNode::Broadcast(b) => {
                let shape = b.expr.shape().ok_or(Error::UnbatchableArgument)?;
                let broadcast_dims = (0..shape.len() as i64).collect();
                self.visit_broadcast_in_dim(&b.expr, b.sizes.clone(), broadcast_dims)?
            }
            NoxprNode::BroadcastInDim(b) => {
                self.visit_broadcast_in_dim(&b.expr, b.sizes.clone(), b.broadcast_dims.clone())?
            }
            NoxprNode::Transpose(t) => {
                let expr = self.visit(&t.expr)?;
                match expr.batch_axis {
                    BatchAxis::NotMapped => BatchedExpr {
                        inner: expr.inner.transpose(t.permutation.clone()),
                        batch_axis: BatchAxis::NotMapped,
                    }
                    .move_batch_axis(self.out_axis.clone())
                    .ok_or(Error::UnbatchableArgument)?,
                    BatchAxis::Mapped { index, size } => {
                        let mut permutation = t.permutation.clone();
                        for p in &mut permutation {
                            if *p >= index as i64 {
                                *p += 1
                            }
                        }
                        BatchedExpr {
                            inner: expr.inner.transpose(t.permutation.clone()),
                            batch_axis: BatchAxis::Mapped { index: 0, size },
                        }
                    }
                }
            }
            NoxprNode::Gather(g) => {
                let expr = self.visit(&g.expr)?;
                let indices = self.visit(&g.indices)?;
                match (&expr.batch_axis, &indices.batch_axis) {
                    (BatchAxis::Mapped { .. }, BatchAxis::NotMapped) => {
                        let expr = expr
                            .move_batch_axis(BatchAxis::Mapped {
                                index: 0,
                                size: usize::MAX,
                            })
                            .ok_or(Error::UnbatchableArgument)?;
                        let expr_shape = expr.inner.shape().ok_or(Error::UnbatchableArgument)?;
                        let slice_sizes: SmallVec<[i64; 4]> = expr_shape
                            .first()
                            .into_iter()
                            .copied()
                            .chain(g.slice_sizes.iter().cloned())
                            .collect();
                        let offset_dims: SmallVec<[i64; 4]> = std::iter::once(0)
                            .chain(g.slice_sizes.iter().map(|x| x + 1))
                            .collect();
                        let collapsed_slice_dims: SmallVec<[i64; 4]> =
                            g.collapsed_slice_dims.iter().map(|x| x + 1).collect();
                        let start_index_map: SmallVec<[i64; 4]> = std::iter::once(0)
                            .chain(g.start_index_map.iter().map(|x| x + 1))
                            .collect();

                        BatchedExpr {
                            inner: expr.inner.gather(
                                indices.inner,
                                offset_dims,
                                collapsed_slice_dims,
                                start_index_map,
                                slice_sizes,
                                g.index_vector_dim,
                            ),
                            batch_axis: expr.batch_axis,
                        }
                    }
                    (BatchAxis::NotMapped, BatchAxis::Mapped { .. }) => {
                        let indices = indices
                            .move_batch_axis(BatchAxis::Mapped {
                                index: 0,
                                size: usize::MAX,
                            })
                            .ok_or(Error::UnbatchableArgument)?;
                        let offset_dims = g.offset_dims.iter().map(|x| x + 1).collect();
                        BatchedExpr {
                            inner: expr.inner.gather(
                                indices.inner,
                                offset_dims,
                                g.collapsed_slice_dims.clone(),
                                g.start_index_map.clone(),
                                g.slice_sizes.clone(),
                                g.index_vector_dim,
                            ),
                            batch_axis: indices.batch_axis,
                        }
                    }
                    (
                        BatchAxis::Mapped {
                            size: expr_size, ..
                        },
                        BatchAxis::Mapped {
                            size: indices_size, ..
                        },
                    ) => {
                        let expr_size = *expr_size;
                        let expr = expr
                            .move_batch_axis(BatchAxis::Mapped {
                                index: 0,
                                size: expr_size,
                            })
                            .ok_or(Error::UnbatchableArgument)?;
                        let indices_size = *indices_size;
                        let indices = indices
                            .move_batch_axis(BatchAxis::Mapped {
                                index: 0,
                                size: indices_size,
                            })
                            .ok_or(Error::UnbatchableArgument)?;

                        // TODO: handle scalars

                        let mut count_shape =
                            indices.inner.shape().ok_or(Error::UnbatchableArgument)?;
                        if let Some(last) = count_shape.last_mut() {
                            *last = -1;
                        }
                        let count_shape_len = count_shape.len();
                        let counts = Noxpr::iota(
                            ArrayTy {
                                element_type: ElementType::S32, // TODO: calculate this type
                                shape: count_shape,
                            },
                            0,
                        );
                        let indices =
                            Noxpr::concat_in_dim(vec![counts, indices.inner], count_shape_len - 1);

                        let slice_sizes: SmallVec<[i64; 4]> = std::iter::once(1)
                            .chain(g.slice_sizes.iter().cloned())
                            .collect();
                        let collapsed_slice_dims: SmallVec<[i64; 4]> = std::iter::once(0)
                            .chain(g.collapsed_slice_dims.iter().map(|x| x + 1))
                            .collect();

                        let offset_dims: SmallVec<[i64; 4]> =
                            g.slice_sizes.iter().map(|x| x + 1).collect();
                        let start_index_map: SmallVec<[i64; 4]> = std::iter::once(0)
                            .chain(g.start_index_map.iter().map(|x| x + 1))
                            .collect();

                        let BatchAxis::Mapped { size, .. } = self.out_axis else {
                            return Err(Error::UnbatchableArgument);
                        };
                        BatchedExpr {
                            batch_axis: BatchAxis::Mapped { index: 0, size },
                            inner: expr.inner.gather(
                                indices,
                                offset_dims,
                                collapsed_slice_dims,
                                start_index_map,
                                slice_sizes,
                                g.index_vector_dim,
                            ),
                        }
                    }
                    (BatchAxis::NotMapped, BatchAxis::NotMapped) => {
                        let inner = expr.inner.gather(
                            indices.inner,
                            g.offset_dims.clone(),
                            g.collapsed_slice_dims.clone(),
                            g.start_index_map.clone(),
                            g.slice_sizes.clone(),
                            g.index_vector_dim,
                        );
                        BatchedExpr {
                            inner,
                            batch_axis: BatchAxis::NotMapped,
                        }
                        .move_batch_axis(self.out_axis.clone())
                        .ok_or(Error::UnbatchableArgument)?
                    }
                }
            }
            NoxprNode::Iota(iota) => BatchedExpr {
                inner: Noxpr::new(NoxprNode::Iota(iota.clone())),
                batch_axis: BatchAxis::NotMapped,
            }
            .move_batch_axis(self.out_axis.clone())
            .ok_or(Error::UnbatchableArgument)?,
            NoxprNode::DynamicUpdateSlice(_) => {
                // TODO: dynamic update slice is a special case of scatter, add this when we add scatter
                todo!()
            }
            NoxprNode::Jax(_) => {
                unimplemented!()
            }
            NoxprNode::GetTupleElement(g) => {
                let NoxprNode::Tuple(elems) = g.expr.deref() else {
                    return Err(Error::UnbatchableArgument);
                };
                let expr = elems.get(g.index).ok_or(Error::UnbatchableArgument)?;
                self.visit(expr)?
            }
        };
        self.cache.insert(id, op.clone());
        Ok(op)
    }

    fn visit_dot_general(
        &mut self,
        lhs: &Noxpr,
        rhs: &Noxpr,
        dims: DotDimensionNums,
    ) -> Result<BatchedExpr, Error> {
        let lhs = self.visit(lhs)?;
        let rhs = self.visit(rhs)?;
        fn bump_dims(dims: &[i64], batch_dim: i64) -> impl Iterator<Item = i64> + '_ {
            dims.iter()
                .cloned()
                .map(move |dim| dim + (dim >= batch_dim) as i64)
        }
        match (lhs.batch_axis, rhs.batch_axis) {
            (
                BatchAxis::Mapped {
                    index: lhs_index, ..
                },
                BatchAxis::Mapped {
                    index: rhs_index, ..
                },
            ) => {
                let lhs_batch_dimensions = iter::once(lhs_index as i64)
                    .chain(bump_dims(&dims.lhs_batch_dimensions, lhs_index as i64))
                    .collect();
                let rhs_batch_dimensions = iter::once(rhs_index as i64)
                    .chain(bump_dims(&dims.rhs_batch_dimensions, rhs_index as i64))
                    .collect();
                let lhs_contracting_dimensions =
                    bump_dims(&dims.lhs_contracting_dimensions, lhs_index as i64).collect();
                let rhs_contracting_dimensions =
                    bump_dims(&dims.rhs_contracting_dimensions, rhs_index as i64).collect();
                let dims = DotDimensionNums {
                    lhs_contracting_dimensions,
                    rhs_contracting_dimensions,
                    lhs_batch_dimensions,
                    rhs_batch_dimensions,
                };
                let inner = lhs.inner.dot_general(rhs.inner, dims);
                let shape = inner.shape().ok_or(Error::UnbatchableArgument)?;
                let size = shape.first().cloned().ok_or(Error::UnbatchableArgument)?;
                Ok(BatchedExpr {
                    inner,
                    batch_axis: BatchAxis::Mapped {
                        index: 0,
                        size: size as usize,
                    },
                })
            }
            (BatchAxis::Mapped { index, .. }, BatchAxis::NotMapped) => {
                let shape = lhs.inner.shape().ok_or(Error::UnbatchableArgument)?;
                let lhs_tensor: SmallVec<[i64; 4]> = (0..shape.len() as i64)
                    .filter(|&d| {
                        !dims.lhs_batch_dimensions.contains(&d)
                            && !dims.lhs_contracting_dimensions.contains(&d)
                    })
                    .collect();

                let batch_dim_index = dims.lhs_batch_dimensions.len()
                    + lhs_tensor.iter().filter(|&&d| d < index as i64).count();
                let lhs_batch_dimensions =
                    bump_dims(&dims.lhs_batch_dimensions, index as i64).collect();
                let lhs_contracting_dimensions =
                    bump_dims(&dims.lhs_contracting_dimensions, index as i64).collect();
                let dims = DotDimensionNums {
                    lhs_contracting_dimensions,
                    lhs_batch_dimensions,
                    ..dims.clone()
                };
                let inner = lhs.inner.dot_general(rhs.inner, dims);
                let shape = inner.shape().ok_or(Error::UnbatchableArgument)?;
                let size = shape.first().cloned().ok_or(Error::UnbatchableArgument)?;
                Ok(BatchedExpr {
                    inner,
                    batch_axis: BatchAxis::Mapped {
                        index: batch_dim_index,
                        size: size as usize,
                    },
                })
            }
            (BatchAxis::NotMapped, BatchAxis::Mapped { index, .. }) => {
                let shape = lhs.inner.shape().ok_or(Error::UnbatchableArgument)?;
                let rhs_tensor: SmallVec<[i64; 4]> = (0..shape.len() as i64)
                    .filter(|&d| {
                        !dims.rhs_batch_dimensions.contains(&d)
                            && !dims.rhs_contracting_dimensions.contains(&d)
                    })
                    .collect();

                let batch_dim_index = dims.rhs_batch_dimensions.len()
                    + rhs_tensor.iter().filter(|&&d| d < index as i64).count();
                let rhs_batch_dimensions =
                    bump_dims(&dims.rhs_batch_dimensions, index as i64).collect();
                let rhs_contracting_dimensions =
                    bump_dims(&dims.rhs_contracting_dimensions, index as i64).collect();
                let dims = DotDimensionNums {
                    rhs_contracting_dimensions,
                    rhs_batch_dimensions,
                    ..dims.clone()
                };
                let inner = lhs.inner.dot_general(rhs.inner, dims);
                let shape = inner.shape().ok_or(Error::UnbatchableArgument)?;
                let size = shape.first().cloned().ok_or(Error::UnbatchableArgument)?;
                Ok(BatchedExpr {
                    inner,
                    batch_axis: BatchAxis::Mapped {
                        index: batch_dim_index,
                        size: size as usize,
                    },
                })
            }
            (BatchAxis::NotMapped, BatchAxis::NotMapped) => BatchedExpr {
                inner: lhs.inner.dot_general(rhs.inner, dims),
                batch_axis: BatchAxis::NotMapped,
            }
            .move_batch_axis(self.out_axis.clone())
            .ok_or(Error::UnbatchableArgument),
        }
    }

    fn visit_broadcast_in_dim(
        &mut self,
        inner: &Noxpr,
        mut sizes: SmallVec<[i64; 4]>,
        mut broadcast_dims: SmallVec<[i64; 4]>,
    ) -> Result<BatchedExpr, Error> {
        let expr = self.visit(inner)?;
        let BatchAxis::Mapped { size: out_size, .. } = self.out_axis else {
            unreachable!()
        };
        match expr.batch_axis {
            BatchAxis::NotMapped => {
                for dim in &mut broadcast_dims {
                    *dim += 1
                }
                sizes.insert(0, out_size as i64);
                let inner = expr.inner.broadcast_in_dim(sizes, broadcast_dims);
                Ok(BatchedExpr {
                    inner,
                    batch_axis: BatchAxis::Mapped {
                        index: 0,
                        size: out_size,
                    },
                })
            }
            ref batch_axis @ BatchAxis::Mapped { size, .. } => {
                for dim in &mut broadcast_dims {
                    *dim += 1
                }
                broadcast_dims.insert(0, 0);
                sizes.insert(0, size as i64);
                let inner = expr.inner.broadcast_in_dim(sizes, broadcast_dims);
                Ok(BatchedExpr {
                    inner,
                    batch_axis: batch_axis.clone(),
                })
            }
        }
    }

    fn visit_binary_op(
        &mut self,
        op: &BinaryOp,
        func: impl Fn(Noxpr, Noxpr) -> Noxpr,
    ) -> Result<BatchedExpr, Error> {
        let lhs = self.visit(&op.lhs)?;
        let rhs = self.visit(&op.rhs)?;
        match (lhs.batch_axis, rhs.batch_axis.clone()) {
            (BatchAxis::NotMapped, BatchAxis::NotMapped) => {
                let expr = BatchedExpr {
                    inner: func(lhs.inner, rhs.inner),
                    batch_axis: BatchAxis::NotMapped,
                };
                expr.move_batch_axis(self.out_axis.clone())
                    .ok_or(Error::UnbatchableArgument)
            }
            (BatchAxis::NotMapped, mapped @ BatchAxis::Mapped { .. })
            | (mapped @ BatchAxis::Mapped { .. }, BatchAxis::NotMapped) => {
                let inner = func(lhs.inner, rhs.inner);
                Ok(BatchedExpr {
                    inner,
                    batch_axis: mapped,
                })
            }
            (lhs_axis @ BatchAxis::Mapped { .. }, BatchAxis::Mapped { .. }) => {
                let rhs = rhs
                    .move_batch_axis(lhs_axis.clone())
                    .ok_or(Error::UnbatchableArgument)?;
                let inner = func(lhs.inner, rhs.inner);
                Ok(BatchedExpr {
                    inner,
                    batch_axis: lhs_axis.clone(),
                })
            }
        }
    }

    fn visit_unary_op(
        &mut self,
        expr: &Noxpr,
        func: impl Fn(Noxpr) -> Noxpr,
    ) -> Result<BatchedExpr, Error> {
        let expr = self.visit(expr)?;
        match expr.batch_axis {
            BatchAxis::NotMapped => Ok(expr
                .move_batch_axis(self.out_axis.clone())
                .ok_or(Error::UnbatchableArgument)?
                .map_expr(func)),

            BatchAxis::Mapped { .. } => Ok(expr.map_expr(func)),
        }
    }
}

pub trait NoxprScalarExt {
    fn constant(self) -> Noxpr;
}

impl<T: NativeType + ArrayElement> NoxprScalarExt for T {
    fn constant(self) -> Noxpr {
        Noxpr::constant(
            self.literal(),
            ArrayTy {
                element_type: T::TY,
                shape: smallvec![],
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::{Client, Collapse, CompFn, Dot, Matrix, Scalar, Tensor, ToHost, Vector};
    use nalgebra::{matrix, vector, Const};
    use smallvec::smallvec;

    #[test]
    fn test_scalar_add_vmap() {
        let client = Client::cpu().unwrap();
        fn add_one(mat: Vector<f32, 5>) -> Vector<f32, 5> {
            mat.vmap(|x: Scalar<f32>| x + 1.0).unwrap().collapse()
        }
        let comp = add_one.build().unwrap();
        let exec = match comp.compile(&client) {
            Ok(exec) => exec,
            Err(xla::Error::XlaError { msg, .. }) => {
                println!("{}", msg);
                panic!();
            }
            Err(e) => {
                panic!("{:?}", e);
            }
        };
        let out = exec
            .run(&client, vector![1.0f32, 2.0, 3.0, 5.0, 6.0])
            .unwrap()
            .to_host();
        assert_eq!(out, vector![2.0, 3.0, 4.0, 6.0, 7.0])
    }

    #[test]
    fn test_unary_vmap() {
        let client = Client::cpu().unwrap();
        fn add_one(mat: Vector<f32, 5>) -> Vector<f32, 5> {
            mat.vmap(|x: Scalar<f32>| x.sqrt()).unwrap().collapse()
        }
        let comp = add_one.build().unwrap();
        println!("{}", comp.to_hlo_text().unwrap());
        let exec = match comp.compile(&client) {
            Ok(exec) => exec,
            Err(xla::Error::XlaError { msg, .. }) => {
                println!("{}", msg);
                panic!();
            }
            Err(e) => {
                panic!("{:?}", e);
            }
        };
        let out = exec
            .run(&client, vector![4.0f32, 9.0, 16.0, 25.0, 36.0])
            .unwrap()
            .to_host();
        assert_eq!(out, vector![2.0, 3.0, 4.0, 5.0, 6.0])
    }

    #[test]
    fn test_broadcast_vmap() {
        let client = Client::cpu().unwrap();
        fn add_one(mat: Vector<f32, 5>) -> Vector<f32, 5> {
            use crate::ScalarExt;
            mat.vmap(|_| 1.0.constant()).unwrap().collapse()
        }
        let comp = add_one.build().unwrap();
        println!("{}", comp.to_hlo_text().unwrap());
        let exec = match comp.compile(&client) {
            Ok(exec) => exec,
            Err(xla::Error::XlaError { msg, .. }) => {
                println!("{}", msg);
                panic!();
            }
            Err(e) => {
                panic!("{:?}", e);
            }
        };
        let out = exec
            .run(&client, vector![4.0f32, 9.0, 16.0, 25.0, 36.0])
            .unwrap()
            .to_host();
        assert_eq!(out, vector![1.0, 1.0, 1.0, 1.0, 1.0])
    }

    #[test]
    fn test_matrix_vmap() {
        let client = Client::cpu().unwrap();
        fn add_one(mat: Matrix<f32, 2, 2>) -> Matrix<f32, 2, 2> {
            mat.vmap(|x: Vector<f32, 2>| x).unwrap().collapse()
        }
        let comp = add_one.build().unwrap();
        println!("{}", comp.to_hlo_text().unwrap());
        let exec = match comp.compile(&client) {
            Ok(exec) => exec,
            Err(xla::Error::XlaError { msg, .. }) => {
                println!("{}", msg);
                panic!();
            }
            Err(e) => {
                panic!("{:?}", e);
            }
        };
        let out = exec
            .run(&client, matrix![1.0, 2.0; 3.0, 4.0])
            .unwrap()
            .to_host();
        assert_eq!(out, matrix![1.0, 2.0; 3.0, 4.0])
    }

    #[test]
    fn test_dot_vmap() {
        let client = Client::cpu().unwrap();
        fn add_one(
            mat: Tensor<f32, (Const<2>, Const<2>, Const<2>)>,
        ) -> Tensor<f32, (Const<2>, Const<2>, Const<2>)> {
            mat.vmap(|x: Matrix<f32, 2, 2>| x.clone().dot(x))
                .unwrap()
                .collapse()
        }
        let comp = add_one.build().unwrap();
        let exec = match comp.compile(&client) {
            Ok(exec) => exec,
            Err(xla::Error::XlaError { msg, .. }) => {
                println!("{}", msg);
                panic!();
            }
            Err(e) => {
                panic!("{:?}", e);
            }
        };
        let out = exec
            .run(
                &client,
                [matrix![1.0, 2.0; 3.0, 4.0f32], matrix![5.0, 8.0; 9.0, 10.0]],
            )
            .unwrap();
        let lit = out.inner.to_literal_sync().unwrap();
        let buf = lit.typed_buf::<f32>().unwrap();
        assert_eq!(buf, &[7.0, 15.0, 10.0, 22.0, 97.0, 135.0, 120.0, 172.0]);
    }

    #[test]
    fn test_broadcast_dims() {
        let a = &[1, 6];
        let b = &[];
        let out = crate::broadcast_dims(a, b);
        assert_eq!(out, Some(smallvec![1, 6]))
    }
}
