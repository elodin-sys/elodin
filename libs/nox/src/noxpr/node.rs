//! Provides the data structures and operations for representing and manipulating tensor computations as expression trees (Noxpr).
use std::{
    collections::HashMap,
    fmt::Display,
    ops::{Add, Deref, Div, Mul, Neg, Sub},
    sync::Arc,
};

use crate::Error;
use itertools::Itertools;
use smallvec::{smallvec, SmallVec};
use xla::{ArrayElement, ElementType, NativeType, XlaBuilder, XlaComputation, XlaOp, XlaOpRef};

/// Represents various types of nodes in an expression tree (Noxpr) for tensor computations.
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
    GreaterOrEqual(BinaryOp),
    LessOrEqual(BinaryOp),
    Less(BinaryOp),
    Equal(BinaryOp),
    Atan2(BinaryOp),

    // Matrix Multiplication
    Dot(BinaryOp),
    DotGeneral(DotGeneral),

    // Unary Ops
    Sqrt(Noxpr),
    Neg(Noxpr),
    Log(Noxpr),
    Sin(Noxpr),
    Cos(Noxpr),
    Abs(Noxpr),

    Acos(Noxpr),
    Asin(Noxpr),

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

    // Control Flow
    Scan(Scan),
    Select(Select),

    // Cast
    Convert(Convert),

    #[cfg(feature = "jax")]
    Jax(pyo3::PyObject),

    Call(Call),

    // Triangle
    Cholesky(Cholesky),
    LuInverse(LuInverse),
}

/// Represents a constant value within the Noxpr.
#[derive(Clone)]
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

/// Represents the type of a node in the Noxpr, either a tuple or array type.
#[derive(Debug, Clone)]
pub enum NoxprTy {
    Tuple(Vec<NoxprTy>),
    ArrayTy(ArrayTy),
}

impl NoxprTy {
    fn pretty_print(&self, writer: &mut impl std::fmt::Write) -> std::fmt::Result {
        match self {
            NoxprTy::ArrayTy(a) => a.pretty_print(writer),
            NoxprTy::Tuple(t) => {
                write!(writer, "tuple(")?;
                for (i, ty) in t.iter().enumerate() {
                    if i != 0 {
                        write!(writer, ", ")?;
                    }
                    ty.pretty_print(writer)?;
                }
                write!(writer, ")")
            }
        }
    }
}

/// Represents a type of array including its element type and shape.
#[derive(Debug, Clone)]
pub struct ArrayTy {
    pub element_type: ElementType,
    pub shape: SmallVec<[i64; 4]>,
}

impl ArrayTy {
    /// Creates a new array type with specified element type and shape.
    pub fn new(element_type: ElementType, shape: SmallVec<[i64; 4]>) -> Self {
        Self {
            element_type,
            shape,
        }
    }

    /// Pretty prints the array type to the given writer.
    fn pretty_print(&self, writer: &mut dyn std::fmt::Write) -> std::fmt::Result {
        write!(writer, "{:?}{:?}", self.element_type, &self.shape)
    }
}

impl From<ArrayTy> for xla::ArrayShape {
    fn from(val: ArrayTy) -> Self {
        xla::ArrayShape::new_with_type(val.element_type, val.shape.to_vec())
    }
}

impl From<NoxprTy> for xla::Shape {
    fn from(val: NoxprTy) -> Self {
        match val {
            NoxprTy::ArrayTy(ty) => xla::Shape::Array(ty.into()),
            NoxprTy::Tuple(tys) => {
                let tys = tys.into_iter().map(Into::into).collect::<Vec<_>>();
                xla::Shape::tuple(tys)
            }
        }
    }
}

/// Represents an operation producing an array of sequential integers.
#[derive(Debug, Clone)]
pub struct Iota {
    pub shape: ArrayTy,
    pub dim: usize,
}

/// Represents a binary operation in the Noxpr.
#[derive(Debug)]
pub struct BinaryOp {
    pub lhs: Noxpr,
    pub rhs: Noxpr,
}

impl BinaryOp {
    /// Calculates the resulting shape of the binary operation.
    fn shape(&self) -> Option<SmallVec<[i64; 4]>> {
        let lhs_shape = self.lhs.shape()?;
        let rhs_shape = self.rhs.shape()?;
        broadcast_dims(&lhs_shape, &rhs_shape)
    }

    /// Determines the resulting type of the binary operation.
    fn ty(&self) -> Option<NoxprTy> {
        let NoxprTy::ArrayTy(lhs_ty) = self.lhs.ty()? else {
            return None;
        };
        let NoxprTy::ArrayTy(rhs_ty) = self.rhs.ty()? else {
            return None;
        };
        if lhs_ty.element_type != rhs_ty.element_type {
            return None;
        }
        let shape = broadcast_dims(&lhs_ty.shape, &rhs_ty.shape);
        Some(NoxprTy::ArrayTy(ArrayTy {
            element_type: lhs_ty.element_type,
            shape: shape?,
        }))
    }
}

/// Broadcasts dimensions of two shapes to determine the resulting shape.
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

/// Represents a generalized dot product operation for matrix or tensor multiplication.
#[derive(Debug)]
pub struct DotGeneral {
    pub lhs: Noxpr,
    pub rhs: Noxpr,
    pub dimensions: DotDimensionNums,
}

impl DotGeneral {
    /// Calculates the resulting shape of the dot operation considering the dimension mappings.
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

    /// Determines the resulting type of the dot operation based on input types and dimension mappings.
    fn ty(&self) -> Option<NoxprTy> {
        let NoxprTy::ArrayTy(lhs) = self.lhs.ty()? else {
            return None;
        };
        let NoxprTy::ArrayTy(rhs) = self.rhs.ty()? else {
            return None;
        };
        let DotDimensionNums {
            lhs_contracting_dimensions: lhs_contracting,
            rhs_contracting_dimensions: rhs_contracting,
            lhs_batch_dimensions: lhs_batch,
            rhs_batch_dimensions: rhs_batch,
        } = self.dimensions.clone();
        let batch_shape = lhs_batch
            .iter()
            .map(|i| lhs.shape[*i as usize])
            .collect::<SmallVec<[i64; 4]>>();
        let lhs_contract_or_batch = lhs_contracting
            .iter()
            .chain(lhs_batch.iter())
            .cloned()
            .collect::<SmallVec<[i64; 4]>>();
        let lhs_tensored_shape = lhs
            .shape
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
            .shape
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
        Some(NoxprTy::ArrayTy(ArrayTy {
            element_type: lhs.element_type,
            shape: result,
        }))
    }
}

/// Stores dimensions specifications for generalized dot product operations.
#[derive(Debug, Clone, Default)]
pub struct DotDimensionNums {
    pub lhs_contracting_dimensions: SmallVec<[i64; 2]>,
    pub rhs_contracting_dimensions: SmallVec<[i64; 2]>,
    pub lhs_batch_dimensions: SmallVec<[i64; 2]>,
    pub rhs_batch_dimensions: SmallVec<[i64; 2]>,
}

impl DotDimensionNums {
    /// Conversion of `DotDimensionNums` to XLA's internal `DotDimensionNumbers` structure.
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

/// Stores the details for concatenation operations within the Noxpr.
#[derive(Debug)]
pub struct Concat {
    pub nodes: Vec<Noxpr>,
    pub dimension: usize,
}

/// Represents slicing operations within the Noxpr.
#[derive(Debug)]
pub struct Slice {
    pub expr: Noxpr,
    pub start_indices: SmallVec<[i64; 4]>,
    pub stop_indices: SmallVec<[i64; 4]>,
    pub strides: SmallVec<[i64; 4]>,
}

/// Represents dynamic slicing operations within the Noxpr.
#[derive(Debug)]
pub struct DynamicSlice {
    pub expr: Noxpr,
    pub start_indices: Vec<Noxpr>,
    pub size_indices: SmallVec<[i64; 4]>,
}

/// Represents reshaping operations within the Noxpr.
#[derive(Debug)]
pub struct Reshape {
    pub expr: Noxpr,
    pub new_sizes: SmallVec<[i64; 4]>,
}

/// Represents a broadcast operation within the Noxpr.
#[derive(Debug)]
pub struct Broadcast {
    pub expr: Noxpr,
    pub sizes: SmallVec<[i64; 4]>,
}

/// Represents a broadcast operation with specific dimensions.
#[derive(Debug)]
pub struct BroadcastInDim {
    pub expr: Noxpr,
    pub sizes: SmallVec<[i64; 4]>,
    pub broadcast_dims: SmallVec<[i64; 4]>,
}

/// Represents a transpose operation within the Noxpr.
#[derive(Debug)]
pub struct Transpose {
    pub expr: Noxpr,
    pub permutation: SmallVec<[i64; 4]>,
}

/// Represents a gather operation, a form of advanced indexing.
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

/// Represents a dynamic update slice operation, updating slices of a tensor dynamically.
#[derive(Debug)]
pub struct DynamicUpdateSlice {
    pub expr: Noxpr,
    pub start_indices: Vec<Noxpr>,
    pub update: Noxpr,
}

/// Represents the operation to extract an element from a tuple.
#[derive(Debug)]
pub struct GetTupleElement {
    pub expr: Noxpr,
    pub index: usize,
}

/// Core structure for representing computational expressions.
#[derive(Debug, Clone)]
pub struct Noxpr {
    pub node: Arc<NoxprNode>,
    pub id: NoxprId,
    pub backtrace: Arc<std::backtrace::Backtrace>,
}

/// Represents a scan operation, a form of reduction across one dimension.
#[derive(Debug, Clone)]
pub struct Scan {
    pub inputs: Vec<Noxpr>,
    pub initial_state: Noxpr,
    pub scan_fn: NoxprFn,
}

/// Represents a scan operation, a form of reduction across one dimension.
#[derive(Debug, Clone)]
pub struct Select {
    pub cond: Noxpr,
    pub on_true: Noxpr,
    pub on_false: Noxpr,
}

/// Represents a scan operation, a form of reduction across one dimension.
#[derive(Debug, Clone)]
pub struct Convert {
    pub arg: Noxpr,
    pub ty: ElementType,
}

#[derive(Clone)]
pub struct NoxprComp {
    pub func: Arc<NoxprFn>,
    pub id: NoxprId,
    pub ty: NoxprTy,
}

impl std::fmt::Debug for NoxprComp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NoxprComp")
            .field("id", &self.id)
            .field("ty", &self.ty)
            .finish()
    }
}

impl NoxprComp {
    pub fn new(func: NoxprFn, ty: NoxprTy) -> Self {
        NoxprComp {
            func: Arc::new(func),
            id: NoxprId::default(),
            ty,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Call {
    pub comp: NoxprComp,
    pub args: Vec<Noxpr>,
}

#[derive(Debug, Clone)]
pub struct Cholesky {
    pub arg: Noxpr,
    pub upper: bool,
}

#[derive(Debug, Clone)]
pub struct LuInverse {
    pub arg: Noxpr,
}

/// A unique identifier for `Noxpr` expressions to facilitate caching and optimization.
#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub struct NoxprId(usize);

impl Default for NoxprId {
    /// Provides default generation of unique identifiers for expressions.
    fn default() -> Self {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        Self(COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

impl Noxpr {
    /// Creates a new `Noxpr` instance from a node.
    pub fn new(node: NoxprNode) -> Self {
        Self {
            backtrace: Arc::new(std::backtrace::Backtrace::capture()),
            id: NoxprId::default(),
            node: Arc::new(node),
        }
    }

    /// Creates a parameter `Noxpr` with a given index, type, and name.
    pub fn parameter(number: i64, ty: NoxprTy, name: String) -> Self {
        Self::new(NoxprNode::Param(ParamExpr { ty, number, name }))
    }

    /// Creates a dot product expression between two `Noxpr` instances.
    pub fn dot(self, rhs: &Noxpr) -> Self {
        Self::new(NoxprNode::Dot(BinaryOp {
            lhs: self,
            rhs: rhs.clone(),
        }))
    }

    /// Creates a generalized dot product expression between two `Noxpr` instances with dimension mapping.
    pub fn dot_general(self, rhs: Noxpr, dimensions: DotDimensionNums) -> Self {
        Self::new(NoxprNode::DotGeneral(DotGeneral {
            lhs: self,
            rhs,
            dimensions,
        }))
    }

    /// Creates a logarithmic transformation of the `Noxpr`.
    pub fn log(self) -> Self {
        Self::new(NoxprNode::Log(self))
    }

    /// Creates a square root transformation of the `Noxpr`.

    pub fn sqrt(self) -> Self {
        Self::new(NoxprNode::Sqrt(self))
    }

    /// Creates a sine transformation of the `Noxpr`.
    pub fn sin(self) -> Self {
        Self::new(NoxprNode::Sin(self))
    }

    /// Creates a cosine transformation of the `Noxpr`.
    pub fn cos(self) -> Self {
        Self::new(NoxprNode::Cos(self))
    }

    /// Creates an absolute value transformation of the `Noxpr`.
    pub fn abs(self) -> Self {
        Self::new(NoxprNode::Abs(self))
    }

    pub fn acos(self) -> Self {
        Self::new(NoxprNode::Acos(self))
    }

    pub fn asin(self) -> Self {
        Self::new(NoxprNode::Asin(self))
    }

    /// Creates a constant `Noxpr` from a given literal and type.
    pub fn constant(data: xla::Literal, ty: ArrayTy) -> Self {
        Self::new(NoxprNode::Constant(Constant { data, ty }))
    }

    /// Combines multiple `Noxpr` into a tuple.
    pub fn tuple(nodes: Vec<Noxpr>) -> Self {
        Self::new(NoxprNode::Tuple(nodes))
    }

    /// Concatenates a list of `Noxpr` along a specified dimension.
    pub fn concat_in_dim(mut nodes: Vec<Noxpr>, dimension: usize) -> Self {
        for node in &mut nodes {
            if node.shape().map(|s| s.is_empty()).unwrap_or_default() {
                *node = node.clone().reshape(smallvec![1]);
            }
        }
        Self::new(NoxprNode::Concat(Concat { nodes, dimension }))
    }

    /// Creates a slice from an `Noxpr`.
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

    /// Creates a broadcasted expression along specified dimensions.
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

    /// Creates a broadcasted expression.
    pub fn broadcast(self, sizes: SmallVec<[i64; 4]>) -> Self {
        Self::new(NoxprNode::Broadcast(Broadcast { expr: self, sizes }))
    }

    /// Creates a broadcasted expression.
    pub fn broadcast_to(self, shape: SmallVec<[i64; 4]>) -> Self {
        let arr_shape = self.shape().unwrap();
        let n_lead = shape.len() - arr_shape.len();
        let broadcast_dims = (n_lead..shape.len()).map(|x| x as i64).collect();
        self.broadcast_in_dim(shape, broadcast_dims)
    }

    /// Creates a transposed `Noxpr`.
    pub fn transpose(self, permutation: SmallVec<[i64; 4]>) -> Self {
        Self::new(NoxprNode::Transpose(Transpose {
            expr: self,
            permutation,
        }))
    }

    /// Logical OR between two `Noxpr`.
    pub fn or(self, rhs: Noxpr) -> Self {
        Self::new(NoxprNode::Or(BinaryOp { lhs: self, rhs }))
    }

    /// Logical AND between two `Noxpr`.
    pub fn and(self, rhs: Noxpr) -> Self {
        Self::new(NoxprNode::And(BinaryOp { lhs: self, rhs }))
    }

    /// Creates a greater-or-equal comparison between two `Noxpr`.
    pub fn greater_or_equal(self, rhs: Noxpr) -> Self {
        Self::new(NoxprNode::GreaterOrEqual(BinaryOp { lhs: self, rhs }))
    }

    /// Creates a less-or-equal comparison between two `Noxpr`.
    pub fn less_or_equal(self, rhs: Noxpr) -> Self {
        Self::new(NoxprNode::LessOrEqual(BinaryOp { lhs: self, rhs }))
    }

    /// Creates a less-than comparison between two `Noxpr`.
    pub fn less(self, rhs: Noxpr) -> Self {
        Self::new(NoxprNode::Less(BinaryOp { lhs: self, rhs }))
    }

    /// Creates a equality comparison between two `Noxpr`.
    pub fn eq(self, rhs: Noxpr) -> Self {
        Self::new(NoxprNode::Equal(BinaryOp { lhs: self, rhs }))
    }

    /// Element-wise arc tangent of two `Noxpr`.
    pub fn atan2(self, rhs: Noxpr) -> Self {
        Self::new(NoxprNode::Atan2(BinaryOp { lhs: self, rhs }))
    }

    /// Reshapes an `Noxpr` to a new size.
    pub fn reshape(self, new_sizes: SmallVec<[i64; 4]>) -> Self {
        Self::new(NoxprNode::Reshape(Reshape {
            expr: self,
            new_sizes,
        }))
    }

    /// Creates a gather operation from an `Noxpr`.
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

    /// Creates an iota operation, which generates an array of sequential integers.
    pub fn iota(shape: ArrayTy, dim: usize) -> Self {
        Self::new(NoxprNode::Iota(Iota { shape, dim }))
    }

    /// Extracts an element from a tuple `Noxpr`.
    pub fn get_tuple_element(&self, index: usize) -> Self {
        Self::new(NoxprNode::GetTupleElement(GetTupleElement {
            expr: self.clone(),
            index,
        }))
    }

    /// Creates a scan operation over a sequence of `Noxpr`.
    pub fn scan(inputs: Vec<Noxpr>, initial_state: Noxpr, scan_fn: NoxprFn) -> Self {
        Self::new(NoxprNode::Scan(Scan {
            inputs,
            initial_state,
            scan_fn,
        }))
    }

    /// Retrieves the type of the expression, which might be useful for type-checking or transformations.
    pub fn ty(&self) -> Option<NoxprTy> {
        match self.deref() {
            NoxprNode::Constant(c) => Some(NoxprTy::ArrayTy(ArrayTy {
                element_type: c.ty.element_type,
                shape: c.ty.shape.clone(),
            })),
            NoxprNode::Param(p) => Some(p.ty.clone()),
            NoxprNode::Add(ref b)
            | NoxprNode::Sub(ref b)
            | NoxprNode::Div(ref b)
            | NoxprNode::Mul(ref b)
            | NoxprNode::And(ref b)
            | NoxprNode::Or(ref b)
            | NoxprNode::GreaterOrEqual(ref b)
            | NoxprNode::LessOrEqual(ref b)
            | NoxprNode::Less(ref b)
            | NoxprNode::Equal(ref b)
            | NoxprNode::Atan2(ref b) => b.ty(),

            NoxprNode::Dot(b) => {
                let NoxprTy::ArrayTy(lhs_ty) = b.lhs.ty()? else {
                    return None;
                };
                let NoxprTy::ArrayTy(rhs_ty) = b.rhs.ty()? else {
                    return None;
                };
                if lhs_ty.element_type != rhs_ty.element_type {
                    return None;
                }
                let shape = match (lhs_ty.shape.len(), rhs_ty.shape.len()) {
                    (1, 1) => SmallVec::new(),
                    (2, 1) => smallvec![lhs_ty.shape[0]],
                    (2, 2) => smallvec![lhs_ty.shape[0], rhs_ty.shape[1]],
                    _ => return None,
                };
                Some(NoxprTy::ArrayTy(ArrayTy {
                    element_type: lhs_ty.element_type,
                    shape,
                }))
            }
            NoxprNode::DotGeneral(s) => s.ty(),
            NoxprNode::Sqrt(expr)
            | NoxprNode::Neg(expr)
            | NoxprNode::Sin(expr)
            | NoxprNode::Cos(expr)
            | NoxprNode::Abs(expr)
            | NoxprNode::Acos(expr)
            | NoxprNode::Asin(expr) => expr.ty(),

            NoxprNode::Concat(concat) => {
                let tys = concat
                    .nodes
                    .iter()
                    .map(Noxpr::ty)
                    .map(|s| match s {
                        Some(NoxprTy::ArrayTy(t)) => Some(t),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>()?;
                let ty = tys.first()?;
                let mut shape = ty.shape.clone();
                let rank = shape.len();
                let dim = concat.dimension;
                // TODO(sphw): ensure that all shapes are the same, except for a particular dim
                if tys.iter().any(|ty| ty.shape.len() != rank) {
                    return None;
                }
                if concat.dimension >= rank {
                    return None;
                }
                let new_len = tys
                    .iter()
                    .map(|ty| ty.shape[concat.dimension])
                    .fold(0, |xs, x| {
                        if x == -1 || xs == -1 {
                            return -1;
                        }
                        xs + x
                    });
                shape[dim] = new_len;
                Some(NoxprTy::ArrayTy(ArrayTy {
                    element_type: ty.element_type,
                    shape,
                }))
            }
            NoxprNode::Slice(slice) => {
                let NoxprTy::ArrayTy(expr_ty) = slice.expr.ty()? else {
                    return None;
                };
                if slice.start_indices.len() != slice.stop_indices.len() {
                    return None;
                }
                let shape = slice
                    .start_indices
                    .iter()
                    .zip(slice.stop_indices.iter())
                    .zip(slice.strides.iter())
                    .map(|((start, stop), stride)| {
                        stop.checked_sub(*start)
                            .and_then(|l| l.checked_div(*stride))
                    })
                    .collect::<Option<SmallVec<_>>>()?;
                Some(NoxprTy::ArrayTy(ArrayTy {
                    element_type: expr_ty.element_type,
                    shape,
                }))
            }
            NoxprNode::DynamicSlice(dynamic_slice) => {
                let ty = dynamic_slice.expr.ty()?;
                let NoxprTy::ArrayTy(ty) = ty else {
                    return None;
                };
                Some(NoxprTy::ArrayTy(ArrayTy {
                    element_type: ty.element_type,
                    shape: dynamic_slice.size_indices.clone(),
                }))
            }
            NoxprNode::Reshape(reshape) => {
                let NoxprTy::ArrayTy(ty) = reshape.expr.ty()? else {
                    return None;
                };
                Some(NoxprTy::ArrayTy(ArrayTy {
                    element_type: ty.element_type,
                    shape: reshape.new_sizes.clone(),
                }))
            }
            NoxprNode::Tuple(t) => {
                let tys = t.iter().map(Noxpr::ty).collect::<Option<Vec<_>>>()?;
                Some(NoxprTy::Tuple(tys))
            }
            NoxprNode::Log(l) => l.ty(),
            NoxprNode::Broadcast(b) => {
                let NoxprTy::ArrayTy(in_ty) = b.expr.ty()? else {
                    return None;
                };
                let mut out_shape = b.sizes.clone();
                out_shape.extend_from_slice(&in_ty.shape);
                Some(NoxprTy::ArrayTy(ArrayTy {
                    element_type: in_ty.element_type,
                    shape: out_shape,
                }))
            }
            NoxprNode::BroadcastInDim(b) => {
                let NoxprTy::ArrayTy(in_ty) = b.expr.ty()? else {
                    return None;
                };
                Some(NoxprTy::ArrayTy(ArrayTy {
                    element_type: in_ty.element_type,
                    shape: b.sizes.clone(),
                }))
            }
            NoxprNode::Transpose(t) => {
                let NoxprTy::ArrayTy(ty) = t.expr.ty()? else {
                    return None;
                };
                let new_shape = t
                    .permutation
                    .iter()
                    .map(|dim| ty.shape[*dim as usize])
                    .collect();
                Some(NoxprTy::ArrayTy(ArrayTy {
                    element_type: ty.element_type,
                    shape: new_shape,
                }))
            }
            NoxprNode::Gather(gather) => {
                let NoxprTy::ArrayTy(ty) = gather.expr.ty()? else {
                    return None;
                };
                let NoxprTy::ArrayTy(indices_ty) = gather.indices.ty()? else {
                    return None;
                };
                let indices_shape = indices_ty.shape;
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
                let shape = (0..output_rank)
                    .filter_map(|i| {
                        if gather.offset_dims.contains(&(i as i64)) {
                            adjusted_slice_sizes.next()
                        } else {
                            expanded_indices_shape_iter.next()
                        }
                    })
                    .collect();
                Some(NoxprTy::ArrayTy(ArrayTy {
                    element_type: ty.element_type,
                    shape,
                }))
            }
            NoxprNode::Iota(i) => Some(NoxprTy::ArrayTy(i.shape.clone())),
            NoxprNode::DynamicUpdateSlice(d) => d.expr.ty(),
            NoxprNode::GetTupleElement(g) => {
                let NoxprTy::Tuple(ty) = g.expr.ty()? else {
                    return None;
                };
                ty.get(g.index).cloned()
            }
            NoxprNode::Scan(s) => s.initial_state.ty(),
            #[cfg(feature = "jax")]
            NoxprNode::Jax(o) => pyo3::Python::with_gil(|py| {
                let shape = o.getattr(py, "shape").ok()?.extract::<Vec<i64>>(py).ok()?;
                let shape = SmallVec::from_vec(shape);
                let element_type = o
                    .getattr(py, "dtype.name")
                    .ok()?
                    .extract::<String>(py)
                    .ok()?;
                let element_type = element_type.parse().ok()?;
                Some(NoxprTy::ArrayTy(ArrayTy {
                    shape,
                    element_type,
                }))
            }),
            NoxprNode::Convert(c) => {
                let NoxprTy::ArrayTy(ty) = c.arg.ty()? else {
                    return None;
                };
                Some(NoxprTy::ArrayTy(ArrayTy {
                    element_type: c.ty,
                    shape: ty.shape,
                }))
            }
            NoxprNode::Select(select) => select.on_true.ty(),
            NoxprNode::Call(c) => Some(c.comp.ty.clone()),
            NoxprNode::Cholesky(c) => c.arg.ty(),
            NoxprNode::LuInverse(lu) => lu.arg.ty(),
        }
    }

    /// Retrieves the element type of the expression if it's a constant or a parameter with a defined type.
    pub fn element_type(&self) -> Option<ElementType> {
        match self.deref() {
            NoxprNode::Constant(c) => Some(c.ty.element_type),
            NoxprNode::Param(p) => match &p.ty {
                NoxprTy::Tuple(_) => None,
                NoxprTy::ArrayTy(a) => Some(a.element_type),
            },
            NoxprNode::Add(ref b)
            | NoxprNode::Sub(ref b)
            | NoxprNode::Div(ref b)
            | NoxprNode::Mul(ref b)
            | NoxprNode::And(ref b)
            | NoxprNode::Or(ref b)
            | NoxprNode::Atan2(ref b) => b.rhs.element_type(),
            NoxprNode::GreaterOrEqual(_)
            | NoxprNode::LessOrEqual(_)
            | NoxprNode::Less(_)
            | NoxprNode::Equal(_) => Some(ElementType::Pred),
            NoxprNode::Dot(b) => b.rhs.element_type(),
            NoxprNode::DotGeneral(s) => s.rhs.element_type(),
            NoxprNode::Sqrt(expr)
            | NoxprNode::Neg(expr)
            | NoxprNode::Log(expr)
            | NoxprNode::Sin(expr)
            | NoxprNode::Cos(expr)
            | NoxprNode::Abs(expr) => expr.element_type(),
            NoxprNode::Acos(expr) => expr.element_type(),
            NoxprNode::Asin(expr) => expr.element_type(),
            NoxprNode::Concat(concat) => concat.nodes.first()?.element_type(),
            NoxprNode::Slice(slice) => slice.expr.element_type(),
            NoxprNode::DynamicSlice(dynamic_slice) => dynamic_slice.expr.element_type(),
            NoxprNode::Reshape(r) => r.expr.element_type(),
            NoxprNode::Tuple(_) => None,
            NoxprNode::Broadcast(b) => b.expr.element_type(),
            NoxprNode::BroadcastInDim(b) => b.expr.element_type(),
            NoxprNode::Transpose(t) => t.expr.element_type(),
            NoxprNode::Gather(gather) => gather.expr.element_type(),
            NoxprNode::Iota(i) => Some(i.shape.element_type),
            NoxprNode::DynamicUpdateSlice(d) => d.expr.element_type(),
            NoxprNode::GetTupleElement(g) => match g.expr.deref() {
                NoxprNode::Tuple(elems) => elems.get(g.index)?.element_type(),
                NoxprNode::Param(p) => {
                    if let NoxprTy::Tuple(elems) = &p.ty {
                        let ty = elems.get(g.index)?;
                        if let NoxprTy::ArrayTy(a) = ty {
                            return Some(a.element_type);
                        }
                    }
                    None
                }
                _ => None,
            },
            NoxprNode::Scan(s) => s.initial_state.element_type(),
            #[cfg(feature = "jax")]
            NoxprNode::Jax(o) => pyo3::Python::with_gil(|py| {
                let element_type = o
                    .getattr(py, "dtype.name")
                    .ok()?
                    .extract::<String>(py)
                    .ok()?;
                element_type.parse().ok()
            }),
            NoxprNode::Convert(c) => Some(c.ty),
            NoxprNode::Select(c) => c.on_true.element_type(),
            NoxprNode::Call(c) => c.comp.func.inner.element_type(),
            NoxprNode::Cholesky(c) => c.arg.element_type(),
            NoxprNode::LuInverse(lu) => lu.arg.element_type(),
        }
    }

    /// Returns the shape of the expression as an array of integers representing the size of each dimension.
    pub fn shape(&self) -> Option<SmallVec<[i64; 4]>> {
        match self.deref() {
            NoxprNode::Constant(c) => Some(c.ty.shape.clone()),
            NoxprNode::Param(p) => match &p.ty {
                NoxprTy::Tuple(_) => None,
                NoxprTy::ArrayTy(a) => Some(a.shape.clone()),
            },
            NoxprNode::Add(ref b)
            | NoxprNode::Sub(ref b)
            | NoxprNode::Div(ref b)
            | NoxprNode::Mul(ref b)
            | NoxprNode::And(ref b)
            | NoxprNode::Or(ref b)
            | NoxprNode::GreaterOrEqual(ref b)
            | NoxprNode::LessOrEqual(ref b)
            | NoxprNode::Less(ref b)
            | NoxprNode::Equal(ref b)
            | NoxprNode::Atan2(ref b) => b.shape(),

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
            NoxprNode::Sqrt(expr)
            | NoxprNode::Neg(expr)
            | NoxprNode::Sin(expr)
            | NoxprNode::Cos(expr)
            | NoxprNode::Abs(expr)
            | NoxprNode::Acos(expr)
            | NoxprNode::Asin(expr) => expr.shape(),

            NoxprNode::Concat(concat) => {
                let shapes = concat
                    .nodes
                    .iter()
                    .map(Noxpr::shape)
                    .collect::<Option<Vec<_>>>()?;
                let mut shape = shapes.first()?.clone();
                let rank = shape.len();
                let dim = concat.dimension;
                // TODO(sphw): ensure that all shapes are the same, except for a particular dim
                if shapes.iter().any(|s| s.len() != rank) {
                    return None;
                }
                if concat.dimension >= rank {
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
            NoxprNode::Broadcast(b) => {
                let in_shape = b.expr.shape()?;
                let mut out_shape = b.sizes.clone();
                out_shape.extend_from_slice(&in_shape);
                Some(out_shape)
            }
            NoxprNode::BroadcastInDim(b) => Some(b.sizes.clone()),
            NoxprNode::Transpose(t) => {
                let shape = t.expr.shape()?;
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
            NoxprNode::GetTupleElement(g) => get_tuple_shape(g.index, &g.expr.node),
            NoxprNode::Scan(s) => s.initial_state.shape(),
            #[cfg(feature = "jax")]
            NoxprNode::Jax(o) => pyo3::Python::with_gil(|py| {
                use pyo3::prelude::PyAnyMethods;
                let jnp = py.import_bound("jax.numpy").unwrap();
                let o = jnp.call_method1("array", (o,)).ok()?;
                let shape = o.getattr("shape").ok()?.extract::<Vec<i64>>().ok()?;
                Some(SmallVec::from_vec(shape))
            }),
            NoxprNode::Convert(c) => c.arg.shape(),
            NoxprNode::Select(c) => c.on_true.shape(),
            NoxprNode::Call(c) => {
                // if let NoxprNode::Jax(j) = &*c.comp.func.inner.node {
                //     None
                // } else {
                //     c.comp.func.inner.shape()
                // }
                if let NoxprTy::ArrayTy(ref ty) = c.comp.ty {
                    Some(ty.shape.clone())
                } else {
                    None
                }
            }
            NoxprNode::Cholesky(c) => c.arg.shape(),
            NoxprNode::LuInverse(lu) => lu.arg.shape(),
        }
    }

    /// Constructs a `Noxpr` for a dynamic slicing operation.
    pub fn dynamic_slice(
        &self,
        start_indices: Vec<Noxpr>,
        size_indices: SmallVec<[i64; 4]>,
    ) -> Noxpr {
        Noxpr::new(NoxprNode::DynamicSlice(DynamicSlice {
            expr: self.clone(),
            start_indices,
            size_indices,
        }))
    }

    /// Constructs a `Noxpr` for dynamically updating a slice of the original data.
    pub fn dynamic_update_slice(&self, start_indices: Vec<Noxpr>, update: Noxpr) -> Noxpr {
        Noxpr::new(NoxprNode::DynamicUpdateSlice(DynamicUpdateSlice {
            expr: self.clone(),
            start_indices,
            update,
        }))
    }

    /// Constructs a new `Noxpr` from a JAX Python object.
    #[cfg(feature = "jax")]
    pub fn jax(py: pyo3::PyObject) -> Noxpr {
        Noxpr::new(NoxprNode::Jax(py))
    }

    pub fn call(comp: NoxprComp, args: Vec<Noxpr>) -> Noxpr {
        Noxpr::new(NoxprNode::Call(Call { comp, args }))
    }

    /// Retrieves the unique identifier of the `Noxpr` instance.
    pub fn id(&self) -> NoxprId {
        self.id
    }

    /// Provides a readable name for the type of node.
    pub fn name(&self) -> &'static str {
        match self.deref() {
            NoxprNode::Param(_) => "Param",
            NoxprNode::Tuple(_) => "Tuple",
            NoxprNode::GetTupleElement(_) => "GetTupleElement",
            NoxprNode::Constant(_) => "Constant",
            NoxprNode::Iota(_) => "Iota",
            NoxprNode::Add(_) => "Add",
            NoxprNode::Sub(_) => "Sub",
            NoxprNode::Mul(_) => "Mul",
            NoxprNode::Div(_) => "Div",
            NoxprNode::And(_) => "And",
            NoxprNode::Or(_) => "Or",
            NoxprNode::GreaterOrEqual(_) => "GreaterOrEqual",
            NoxprNode::LessOrEqual(_) => "LessOrEqual",
            NoxprNode::Less(_) => "Less",
            NoxprNode::Equal(_) => "Equal",
            NoxprNode::Atan2(_) => "Atan2",
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
            NoxprNode::Scan(_) => "Scan",
            #[cfg(feature = "jax")]
            NoxprNode::Jax(_) => "Jax",
            NoxprNode::Sin(_) => "Sin",
            NoxprNode::Cos(_) => "Cos",
            NoxprNode::Asin(_) => "ASin",
            NoxprNode::Acos(_) => "ACos",
            NoxprNode::Abs(_) => "Abs",
            NoxprNode::Convert(_) => "Convert",
            NoxprNode::Select(_) => "Select",
            NoxprNode::Call(_) => "Call",
            NoxprNode::Cholesky(_) => "Cholesky",
            NoxprNode::LuInverse(_) => "LuInverse",
        }
    }

    /// Expands the rank of an expression to a higher dimensionality, typically used in broadcasting scenarios.
    pub fn expand_rank(self, rank: usize) -> Option<Noxpr> {
        let in_shape = self.shape()?;
        let in_rank = in_shape.len();
        let broadcast_dims = (0..in_rank).map(|x| x as i64).collect();
        let mut out_shape = in_shape.clone();
        for _ in in_rank..rank {
            out_shape.push(1);
        }
        Some(self.broadcast_in_dim(out_shape, broadcast_dims))
    }

    pub fn convert(&self, ty: ElementType) -> Noxpr {
        Noxpr::new(NoxprNode::Convert(Convert {
            arg: self.clone(),
            ty,
        }))
    }

    pub fn select(&self, on_true: Noxpr, on_false: Noxpr) -> Noxpr {
        Noxpr::new(NoxprNode::Select(Select {
            cond: self.clone(),
            on_true,
            on_false,
        }))
    }

    pub fn cholesky(&self, upper: bool) -> Noxpr {
        Noxpr::new(NoxprNode::Cholesky(Cholesky {
            arg: self.clone(),
            upper,
        }))
    }

    pub fn lu_inverse(&self) -> Noxpr {
        Noxpr::new(NoxprNode::LuInverse(LuInverse { arg: self.clone() }))
    }
}

impl Display for Noxpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut tracer = PrettyPrintTracer::default();
        tracer.visit(self, f)?;
        Ok(())
    }
}

impl Deref for Noxpr {
    type Target = NoxprNode;

    fn deref(&self) -> &Self::Target {
        &self.node
    }
}

fn get_tuple_shape(index: usize, expr: &NoxprNode) -> Option<SmallVec<[i64; 4]>> {
    match expr {
        NoxprNode::Tuple(elems) => elems.get(index)?.shape(),
        NoxprNode::Param(p) => {
            if let NoxprTy::Tuple(elems) = &p.ty {
                let ty = elems.get(index)?;
                if let NoxprTy::ArrayTy(a) = ty {
                    return Some(a.shape.clone());
                }
            }
            None
        }
        NoxprNode::Call(c) => match &c.comp.ty {
            NoxprTy::Tuple(elems) => elems.get(index).and_then(|e| match e {
                NoxprTy::Tuple(_) => None,
                NoxprTy::ArrayTy(a) => Some(a.shape.clone()),
            }),
            _ => None,
        },
        _ => None,
    }
}

/// Represents a parameter in the Noxpr.
#[derive(Debug, Clone)]
pub struct ParamExpr {
    pub number: i64,
    pub name: String,
    pub ty: NoxprTy,
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

/// Traces and transforms `Noxpr` expressions into XLA operations.
pub struct XlaTracer {
    builder: XlaBuilder,
    cache: HashMap<NoxprId, XlaOp>,
    comp_cache: HashMap<NoxprId, Arc<XlaComputation>>,
}

impl XlaTracer {
    /// Constructs a new `XlaTracer` with a specified computation name.
    pub fn new(name: &str) -> Self {
        Self {
            builder: XlaBuilder::new(name),
            cache: HashMap::new(),
            comp_cache: HashMap::new(),
        }
    }

    /// Visits a `Noxpr`, recursively compiling it into an `XlaOp` using the XlaBuilder, with caching to prevent redundant computations.
    pub fn visit(&mut self, expr: &Noxpr) -> Result<XlaOp, Error> {
        let id = expr.id();
        if let Some(op) = self.cache.get(&id) {
            return Ok(op.clone());
        }

        let op = match expr.deref() {
            NoxprNode::Constant(c) => self.builder.constant_literal(&c.data)?.reshape(&c.ty.shape),
            NoxprNode::Param(p) => {
                self.builder
                    .parameter(p.number, p.ty.clone().into(), &p.name)?
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
            NoxprNode::GreaterOrEqual(b) => {
                let (lhs, rhs) = self.visit_binary_op(b)?;
                lhs.ge(&rhs)
            }
            NoxprNode::LessOrEqual(b) => {
                let (lhs, rhs) = self.visit_binary_op(b)?;
                lhs.le(&rhs)
            }
            NoxprNode::Less(b) => {
                let (lhs, rhs) = self.visit_binary_op(b)?;
                lhs.lt(&rhs)
            }
            NoxprNode::Equal(b) => {
                let (lhs, rhs) = self.visit_binary_op(b)?;
                lhs.eq(&rhs)
            }
            NoxprNode::Atan2(b) => {
                let (lhs, rhs) = self.visit_binary_op(b)?;
                lhs.atan2(&rhs)
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
            NoxprNode::Sin(expr) => {
                let expr = self.visit(expr)?;
                expr.sin()
            }
            NoxprNode::Cos(expr) => {
                let expr = self.visit(expr)?;
                expr.cos()
            }
            NoxprNode::Abs(expr) => {
                let expr = self.visit(expr)?;
                expr.abs()
            }

            NoxprNode::Acos(c) => {
                let arg = self.visit(c)?;
                arg.acos()
            }
            NoxprNode::Asin(c) => {
                let arg = self.visit(c)?;
                arg.asin()
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
                    .start_indices
                    .iter()
                    .map(|expr| self.visit(expr))
                    .collect::<Result<Vec<_>, Error>>()?;
                let start = start
                    .iter()
                    .map(|op| op.as_ref())
                    .collect::<SmallVec<[XlaOpRef<'_>; 4]>>();
                inner.dynamic_update_slice(&update, &start)
            }
            #[cfg(feature = "jax")]
            NoxprNode::Jax(_) => {
                unimplemented!()
            }
            NoxprNode::Scan(s) => {
                let mut xs_shape = None;
                let mut input_shape = vec![];
                for arg in &s.inputs {
                    let shape = arg.shape().unwrap();
                    let element_type = arg.element_type().unwrap();
                    input_shape.push(NoxprTy::ArrayTy(ArrayTy {
                        shape: shape.clone(),
                        element_type,
                    }));
                    if xs_shape.is_none() {
                        xs_shape = Some(shape);
                    } else if let Some(xs_shape) = xs_shape.as_mut() {
                        if xs_shape.first() != shape.first() {
                            //return Err(Error::ScanShapeMismatch);
                        }
                    }
                }
                let arg_shape = xs_shape.unwrap();
                if s.inputs.is_empty() {
                    return Err(Error::ScanMissingArg);
                }

                fn index(noxpr: &Noxpr, index: Noxpr) -> Option<Noxpr> {
                    let mut shape = noxpr.shape()?;
                    *shape.first_mut()? = 1;
                    let starts = std::iter::once(index)
                        .chain((0..(shape.len() - 1)).map(|_| 0i64.constant()))
                        .collect();
                    let out = noxpr.dynamic_slice(starts, shape.clone());
                    shape.remove(0);
                    Some(out.reshape(shape))
                }
                let mut init_tuple = input_shape.clone();
                init_tuple.insert(
                    0,
                    NoxprTy::ArrayTy(ArrayTy {
                        element_type: ElementType::S64,
                        shape: smallvec![],
                    }),
                );
                let scan_fn = {
                    let mut scan_fn = s.scan_fn.collapse_params(init_tuple)?;
                    let next = scan_fn.args[0].get_tuple_element(0).add(1i64.constant());
                    let mut tuple = vec![next.clone()];
                    for i in 0..s.inputs.len() {
                        tuple.push(scan_fn.args[0].get_tuple_element(1 + i));
                    }
                    tuple.push(scan_fn.inner);
                    for i in 0..s.inputs.len() {
                        let inner_xs_arg = scan_fn.args[0].get_tuple_element(1 + i);
                        let Some(next_x) = index(&inner_xs_arg, next.clone()) else {
                            panic!()
                        };
                        tuple.push(next_x);
                    }
                    scan_fn.inner = Noxpr::tuple(tuple);
                    scan_fn
                };
                let cond = {
                    let len = *arg_shape.first().unwrap();
                    let arg = scan_fn.args[0].clone();

                    let index = arg.get_tuple_element(0);
                    let inner = index.less(len.constant());
                    NoxprFn {
                        args: scan_fn.args.clone(),
                        inner,
                    }
                    .build("scan_cond")?
                    .build()?
                };
                let scan_fn = scan_fn.build("scan_fn")?.build()?;
                let mut initial_state = vec![0i64.constant()];
                for i in s.inputs.iter() {
                    initial_state.push(i.clone())
                }
                initial_state.push(s.initial_state.clone());
                for i in s.inputs.iter() {
                    initial_state.push(index(i, 0i64.constant()).unwrap())
                }
                let last_elem = s.inputs.len() + 1;
                let initial_state = Noxpr::tuple(initial_state);
                let initial_state = self.visit(&initial_state)?;
                let out = cond.stmt_while(&scan_fn, &initial_state);
                out.get_tuple_element(last_elem as i64)
            }
            NoxprNode::Convert(c) => {
                let arg = self.visit(&c.arg)?;
                arg.convert_element_type(c.ty.primitive_type())
            }
            NoxprNode::Select(s) => {
                let cond = self.visit(&s.cond)?;
                let on_true = self.visit(&s.on_true)?;
                let on_false = self.visit(&s.on_false)?;
                cond.select(&on_true, &on_false)
            }
            NoxprNode::Call(c) => {
                let comp = self.visit_comp(&c.comp)?;
                let args = c
                    .args
                    .iter()
                    .map(|arg| self.visit(arg))
                    .collect::<Result<Vec<_>, _>>()?;
                let args = args.iter().map(XlaOp::as_ref).collect::<Vec<_>>();
                self.builder.call(&args, &comp)
            }
            NoxprNode::Cholesky(c) => {
                let arg = self.visit(&c.arg)?;
                arg.cholesky(!c.upper)
            }
            NoxprNode::LuInverse(_) => {
                todo!() // TODO: add this when we get custom calls
            }
        };
        self.cache.insert(id, op.clone());
        Ok(op)
    }

    fn visit_comp(&mut self, comp: &NoxprComp) -> Result<Arc<XlaComputation>, Error> {
        if let Some(comp) = self.comp_cache.get(&comp.id) {
            return Ok(comp.clone());
        }
        let mut tracer = XlaTracer::new("comp");
        tracer.comp_cache.clone_from(&self.comp_cache);
        let xla_comp = Arc::new(comp.func.build_with_tracer(&mut tracer)?.build()?);
        self.comp_cache = tracer.comp_cache;
        self.comp_cache.insert(comp.id, xla_comp.clone());
        Ok(xla_comp)
    }

    /// Helps in visiting and compiling binary operations like Add, Mul into XLA binary operations.
    #[inline]
    fn visit_binary_op(&mut self, op: &BinaryOp) -> Result<(XlaOp, XlaOp), Error> {
        Ok((self.visit(&op.lhs)?, self.visit(&op.rhs)?))
    }
}

/// A function that encapsulates a `Noxpr` and its arguments.
#[derive(Debug, Clone)]
pub struct NoxprFn {
    pub args: Vec<Noxpr>,
    pub inner: Noxpr,
}

impl NoxprFn {
    /// Creates a new `NoxprFn` with specified arguments and inner expression.
    pub fn new(args: Vec<Noxpr>, inner: Noxpr) -> Self {
        Self { args, inner }
    }

    /// Builds an XLA operation based on the `NoxprFn` definition.
    pub fn build(&self, name: &str) -> Result<XlaOp, Error> {
        let mut tracer = XlaTracer::new(name);
        self.build_with_tracer(&mut tracer)
    }

    /// Builds an XLA operation based on the `NoxprFn` definition.
    pub fn build_with_tracer(&self, tracer: &mut XlaTracer) -> Result<XlaOp, Error> {
        for a in self.args.iter() {
            tracer.visit(a)?;
        }
        tracer.visit(&self.inner)
    }

    /// Collapses multiple parameters into a single tuple parameter for compact representation.
    pub fn collapse_params(&self, mut init_tuple: Vec<NoxprTy>) -> Result<Self, Error> {
        let init_offset = init_tuple.len();
        for a in self.args.iter() {
            init_tuple.push(a.ty().unwrap())
        }
        let new_param = Noxpr::parameter(
            0,
            NoxprTy::Tuple(init_tuple),
            "collapsed_params".to_string(),
        );
        let cache = self
            .args
            .iter()
            .enumerate()
            .map(|(i, arg)| {
                (
                    arg.id(),
                    Noxpr::get_tuple_element(&new_param, i + init_offset),
                )
            })
            .collect();
        let mut tracer = ReplacementTracer { cache };
        let mut final_fn = tracer.visit_fn(self);
        final_fn.args = vec![new_param];
        Ok(final_fn)
    }

    /// Pretty prints the function's structure into a formatted string.
    pub fn pretty_print(
        &self,
        parent_printer: &PrettyPrintTracer,
        mut writer: &mut dyn std::fmt::Write,
    ) -> std::fmt::Result {
        let mut printer = parent_printer.clone();
        write!(writer, "fn(")?;
        for (i, _) in self.args.iter().enumerate() {
            if i != 0 {
                write!(writer, ", ")?;
            }
            write!(writer, "var_{}", i)?;
        }
        writeln!(writer, ") {{")?;
        let mut ident_writer =
            indent_write::fmt::IndentWriter::new("  ", &mut writer as &mut dyn std::fmt::Write);
        printer.visit(&self.inner, &mut ident_writer)?;
        writeln!(writer, "}}")
    }
}

impl std::fmt::Display for NoxprFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let printer = PrettyPrintTracer::default();
        self.pretty_print(&printer, f)
    }
}

/// Used to trace and replace `Noxpr` references, mainly for transformation or optimization purposes.
#[derive(Debug, Clone, Default)]
pub struct ReplacementTracer {
    pub cache: HashMap<NoxprId, Noxpr>,
}

impl ReplacementTracer {
    /// Visits and potentially modifies a `NoxprFn` based on cached transformations.
    fn visit_fn(&mut self, func: &NoxprFn) -> NoxprFn {
        let args = func.args.iter().map(|a| self.visit(a)).collect::<Vec<_>>();
        let inner = self.visit(&func.inner);
        NoxprFn::new(args, inner)
    }

    /// Visits and potentially modifies a `Noxpr` based on cached transformations.
    pub fn visit(&mut self, expr: &Noxpr) -> Noxpr {
        let id = expr.id();
        if let Some(expr) = self.cache.get(&id) {
            return expr.clone();
        }
        let expr = match expr.deref() {
            NoxprNode::Param(p) => Noxpr::new(NoxprNode::Param(p.clone())),
            NoxprNode::Tuple(t) => Noxpr::tuple(t.iter().map(|e| self.visit(e)).collect()),
            NoxprNode::GetTupleElement(g) => {
                Noxpr::new(NoxprNode::GetTupleElement(GetTupleElement {
                    expr: self.visit(&g.expr),
                    index: g.index,
                }))
            }
            NoxprNode::Constant(c) => Noxpr::new(NoxprNode::Constant(c.clone())),
            NoxprNode::Iota(i) => Noxpr::new(NoxprNode::Iota(i.clone())),
            NoxprNode::Add(a) => Noxpr::new(NoxprNode::Add(self.visit_binary_op(a))),
            NoxprNode::Sub(s) => Noxpr::new(NoxprNode::Sub(self.visit_binary_op(s))),
            NoxprNode::Mul(x) => Noxpr::new(NoxprNode::Mul(self.visit_binary_op(x))),
            NoxprNode::Div(x) => Noxpr::new(NoxprNode::Div(self.visit_binary_op(x))),
            NoxprNode::And(x) => Noxpr::new(NoxprNode::And(self.visit_binary_op(x))),
            NoxprNode::GreaterOrEqual(x) => {
                Noxpr::new(NoxprNode::GreaterOrEqual(self.visit_binary_op(x)))
            }
            NoxprNode::LessOrEqual(x) => {
                Noxpr::new(NoxprNode::LessOrEqual(self.visit_binary_op(x)))
            }
            NoxprNode::Less(x) => Noxpr::new(NoxprNode::Less(self.visit_binary_op(x))),
            NoxprNode::Equal(x) => Noxpr::new(NoxprNode::Equal(self.visit_binary_op(x))),
            NoxprNode::Atan2(x) => Noxpr::new(NoxprNode::Atan2(self.visit_binary_op(x))),
            NoxprNode::Or(x) => Noxpr::new(NoxprNode::Or(self.visit_binary_op(x))),
            NoxprNode::Dot(x) => Noxpr::new(NoxprNode::Dot(self.visit_binary_op(x))),
            NoxprNode::DotGeneral(d) => Noxpr::new(NoxprNode::DotGeneral(DotGeneral {
                lhs: self.visit(&d.lhs),
                rhs: self.visit(&d.rhs),
                dimensions: d.dimensions.clone(),
            })),
            NoxprNode::Sqrt(s) => Noxpr::new(NoxprNode::Sqrt(self.visit(s))),
            NoxprNode::Neg(n) => Noxpr::new(NoxprNode::Neg(self.visit(n))),
            NoxprNode::Log(l) => Noxpr::new(NoxprNode::Log(self.visit(l))),
            NoxprNode::Sin(s) => Noxpr::new(NoxprNode::Sin(self.visit(s))),
            NoxprNode::Cos(c) => Noxpr::new(NoxprNode::Cos(self.visit(c))),
            NoxprNode::Abs(a) => Noxpr::new(NoxprNode::Abs(self.visit(a))),

            NoxprNode::Acos(c) => Noxpr::new(NoxprNode::Acos(self.visit(c))),
            NoxprNode::Asin(c) => Noxpr::new(NoxprNode::Asin(self.visit(c))),

            NoxprNode::Concat(c) => Noxpr::new(NoxprNode::Concat(Concat {
                nodes: c.nodes.iter().map(|n| self.visit(n)).collect(),
                dimension: c.dimension,
            })),
            NoxprNode::Reshape(r) => Noxpr::new(NoxprNode::Reshape(Reshape {
                expr: self.visit(&r.expr),
                new_sizes: r.new_sizes.clone(),
            })),
            NoxprNode::Broadcast(b) => Noxpr::new(NoxprNode::Broadcast(Broadcast {
                expr: self.visit(&b.expr),
                sizes: b.sizes.clone(),
            })),
            NoxprNode::BroadcastInDim(b) => Noxpr::new(NoxprNode::BroadcastInDim(BroadcastInDim {
                expr: self.visit(&b.expr),
                sizes: b.sizes.clone(),
                broadcast_dims: b.broadcast_dims.clone(),
            })),
            NoxprNode::Transpose(t) => Noxpr::new(NoxprNode::Transpose(Transpose {
                expr: self.visit(&t.expr),
                permutation: t.permutation.clone(),
            })),
            NoxprNode::Gather(g) => Noxpr::new(NoxprNode::Gather(Gather {
                expr: self.visit(&g.expr),
                indices: self.visit(&g.indices),
                offset_dims: g.offset_dims.clone(),
                collapsed_slice_dims: g.collapsed_slice_dims.clone(),
                start_index_map: g.start_index_map.clone(),
                slice_sizes: g.slice_sizes.clone(),
                index_vector_dim: g.index_vector_dim,
            })),
            NoxprNode::Slice(s) => Noxpr::new(NoxprNode::Slice(Slice {
                expr: self.visit(&s.expr),
                start_indices: s.start_indices.clone(),
                stop_indices: s.stop_indices.clone(),
                strides: s.strides.clone(),
            })),
            NoxprNode::DynamicSlice(d) => Noxpr::new(NoxprNode::DynamicSlice(DynamicSlice {
                expr: self.visit(&d.expr),
                start_indices: d.start_indices.iter().map(|e| self.visit(e)).collect(),
                size_indices: d.size_indices.clone(),
            })),
            NoxprNode::DynamicUpdateSlice(d) => {
                Noxpr::new(NoxprNode::DynamicUpdateSlice(DynamicUpdateSlice {
                    expr: self.visit(&d.expr),
                    start_indices: d.start_indices.iter().map(|e| self.visit(e)).collect(),
                    update: self.visit(&d.update),
                }))
            }
            NoxprNode::Scan(s) => Noxpr::new(NoxprNode::Scan(Scan {
                inputs: s.inputs.iter().map(|e| self.visit(e)).collect(),
                initial_state: self.visit(&s.initial_state),
                scan_fn: s.scan_fn.clone(),
            })),
            #[cfg(feature = "jax")]
            NoxprNode::Jax(j) => Noxpr::new(NoxprNode::Jax(j.clone())),
            NoxprNode::Convert(c) => {
                let arg = self.visit(&c.arg);
                Noxpr::new(NoxprNode::Convert(Convert { arg, ty: c.ty }))
            }
            NoxprNode::Select(s) => {
                let cond = self.visit(&s.cond);
                let on_true = self.visit(&s.on_true);
                let on_false = self.visit(&s.on_false);
                Noxpr::new(NoxprNode::Select(Select {
                    cond,
                    on_true,
                    on_false,
                }))
            }
            NoxprNode::Call(c) => {
                let args = c.args.iter().map(|a| self.visit(a)).collect();
                Noxpr::new(NoxprNode::Call(Call {
                    comp: c.comp.clone(),
                    args,
                }))
            }
            NoxprNode::Cholesky(c) => self.visit(&c.arg).cholesky(c.upper),
            NoxprNode::LuInverse(lu) => self.visit(&lu.arg).lu_inverse(),
        };
        self.cache.insert(id, expr.clone());
        expr
    }

    /// Helper method to visit and modify binary operations.
    fn visit_binary_op(&mut self, op: &BinaryOp) -> BinaryOp {
        BinaryOp {
            lhs: self.visit(&op.lhs),
            rhs: self.visit(&op.rhs),
        }
    }
}

/// Extension for scalar operations on `Noxpr`.
pub trait NoxprScalarExt {
    /// Creates a constant `Noxpr` from a scalar value.
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

/// Utility class for pretty printing expressions, useful for debugging or displaying computational graphs.
#[derive(Clone, Debug, Default)]
pub struct PrettyPrintTracer {
    printed: HashMap<NoxprId, usize>,
}

impl PrettyPrintTracer {
    /// Prints a variable label for debugging purposes.
    fn print_var_label(&self, writer: &mut dyn std::fmt::Write) -> std::fmt::Result {
        write!(writer, "var_{}", self.printed.len())
    }

    /// Prints a variable with a given ID.
    fn print_var(
        &mut self,
        id: NoxprId,
        writer: &mut dyn std::fmt::Write,
    ) -> Result<usize, std::fmt::Error> {
        write!(writer, "let ")?;
        self.print_var_label(writer)?;
        write!(writer, " = ")?;
        let num = self.printed.len();
        self.printed.insert(id, num);
        Ok(num)
    }

    /// Visits a `Noxpr` and prints it in a readable format.
    fn visit<W: std::fmt::Write>(
        &mut self,
        expr: &Noxpr,
        writer: &mut W,
    ) -> Result<usize, std::fmt::Error> {
        let id = expr.id();
        if let Some(num) = self.printed.get(&id) {
            return Ok(*num);
        }
        let var_name = match expr.deref() {
            NoxprNode::Param(p) => {
                let num = self.print_var(id, writer)?;
                write!(writer, "param(num = {}, name = {}, ty = ", p.number, p.name,)?;
                p.ty.pretty_print(writer)?;
                write!(writer, ")")?;
                Ok(num)
            }
            NoxprNode::Tuple(t) => {
                let nums: Vec<_> = t
                    .iter()
                    .map(|e| self.visit(e, writer))
                    .collect::<Result<_, _>>()?;
                let num = self.print_var(id, writer)?;
                write!(writer, "(")?;
                for num in nums {
                    write!(writer, "var_{}, ", num)?;
                }
                write!(writer, ")")?;
                Ok(num)
            }
            NoxprNode::GetTupleElement(t) => {
                self.visit(&t.expr, writer)?;
                let Some(&arg_num) = self.printed.get(&t.expr.id()) else {
                    panic!("tuple element not visited")
                };
                let num = self.print_var(id, writer)?;
                write!(writer, "var_{arg_num}.{}", t.index)?;
                Ok(num)
            }
            NoxprNode::Constant(c) => {
                let num = self.print_var(id, writer)?;
                write!(writer, "constant(")?;
                c.ty.pretty_print(writer)?;
                write!(writer, ")")?;
                Ok(num)
            }
            NoxprNode::Iota(i) => {
                let num = self.print_var(id, writer)?;
                write!(writer, "iota({:?}, ", i.dim,)?;
                i.shape.pretty_print(writer)?;
                write!(writer, ")")?;
                Ok(num)
            }
            NoxprNode::Add(a) => self.visit_binary_op(id, a, "+", writer),
            NoxprNode::Sub(s) => self.visit_binary_op(id, s, "-", writer),

            NoxprNode::Acos(c) => {
                let arg = self.visit(c, writer)?;
                let num = self.print_var(id, writer)?;
                write!(writer, "acos(var_{})", arg)?;
                Ok(num)
            }
            NoxprNode::Asin(c) => {
                let arg = self.visit(c, writer)?;
                let num = self.print_var(id, writer)?;
                write!(writer, "asin(var_{})", arg)?;
                Ok(num)
            }

            NoxprNode::Mul(m) => self.visit_binary_op(id, m, "*", writer),
            NoxprNode::Div(d) => self.visit_binary_op(id, d, "/", writer),
            NoxprNode::And(a) => self.visit_binary_op(id, a, "&&", writer),
            NoxprNode::Or(o) => self.visit_binary_op(id, o, "||", writer),
            NoxprNode::GreaterOrEqual(g) => self.visit_binary_op(id, g, ">=", writer),
            NoxprNode::LessOrEqual(le) => self.visit_binary_op(id, le, "<=", writer),
            NoxprNode::Less(l) => self.visit_binary_op(id, l, "<", writer),
            NoxprNode::Equal(l) => self.visit_binary_op(id, l, "==", writer),
            NoxprNode::Atan2(l) => {
                let lhs = self.visit(&l.lhs, writer)?;
                let rhs = self.visit(&l.rhs, writer)?;
                let num = self.print_var(id, writer)?;
                write!(writer, "atan2(var_{}, var_{})", lhs, rhs)?;
                Ok(num)
            }
            NoxprNode::Dot(d) => self.visit_binary_op(id, d, ".", writer),
            NoxprNode::DotGeneral(d) => {
                let lhs = self.visit(&d.lhs, writer)?;
                let rhs = self.visit(&d.rhs, writer)?;
                let num = self.print_var(id, writer)?;
                write!(
                    writer,
                    "dot_general(lhs = var_{}, rhs = var_{}, dims = {:?})",
                    lhs, rhs, d.dimensions
                )?;
                Ok(num)
            }
            NoxprNode::Sqrt(s) => {
                let arg = self.visit(s, writer)?;
                let num = self.print_var(id, writer)?;
                write!(writer, "sqrt(var_{})", arg)?;
                Ok(num)
            }
            NoxprNode::Neg(n) => {
                let arg = self.visit(n, writer)?;
                let num = self.print_var(id, writer)?;
                write!(writer, "-var_{}", arg)?;
                Ok(num)
            }
            NoxprNode::Log(l) => {
                let arg = self.visit(l, writer)?;
                let num = self.print_var(id, writer)?;
                write!(writer, "log(var_{})", arg)?;
                Ok(num)
            }
            NoxprNode::Sin(s) => {
                let arg = self.visit(s, writer)?;
                let num = self.print_var(id, writer)?;
                write!(writer, "sin(var_{})", arg)?;
                Ok(num)
            }
            NoxprNode::Cos(c) => {
                let arg = self.visit(c, writer)?;
                let num = self.print_var(id, writer)?;
                write!(writer, "cos(var_{})", arg)?;
                Ok(num)
            }
            NoxprNode::Abs(c) => {
                let arg = self.visit(c, writer)?;
                let num = self.print_var(id, writer)?;
                write!(writer, "|var_{}|", arg)?;
                Ok(num)
            }
            NoxprNode::Concat(c) => {
                let nums: Vec<_> = c
                    .nodes
                    .iter()
                    .map(|e| self.visit(e, writer))
                    .collect::<Result<_, _>>()?;
                let num = self.print_var(id, writer)?;
                write!(writer, "concat(")?;
                for num in nums {
                    write!(writer, "var_{}, ", num)?;
                }
                write!(writer, "dim = {})", c.dimension)?;
                Ok(num)
            }
            NoxprNode::Reshape(r) => {
                let arg = self.visit(&r.expr, writer)?;
                let num = self.print_var(id, writer)?;
                write!(writer, "reshape(var_{}, {:?})", arg, r.new_sizes)?;
                Ok(num)
            }
            NoxprNode::Broadcast(b) => {
                let arg = self.visit(&b.expr, writer)?;
                let num = self.print_var(id, writer)?;
                write!(writer, "broadcast(var_{}, {:?})", arg, b.sizes)?;
                Ok(num)
            }
            NoxprNode::BroadcastInDim(b) => {
                let arg = self.visit(&b.expr, writer)?;
                let num = self.print_var(id, writer)?;
                write!(
                    writer,
                    "broadcast_in_dim(var_{}, {:?}, {:?})",
                    arg, b.sizes, b.broadcast_dims
                )?;
                Ok(num)
            }
            NoxprNode::Transpose(t) => {
                let arg = self.visit(&t.expr, writer)?;
                let num = self.print_var(id, writer)?;
                write!(writer, "transpose(var_{}, {:?})", arg, t.permutation)?;
                Ok(num)
            }
            NoxprNode::Gather(g) => {
                let expr = self.visit(&g.expr, writer)?;
                let indices = self.visit(&g.indices, writer)?;
                let num = self.print_var(id, writer)?;
                write!(
                    writer,
                    "gather(expr = var_{}, indices = var_{}, offset_dims = {:?}, collapsed_slice_dims = {:?}, start_index_map = {:?}, slice_sizes = {:?}, index_vector_dim = {})",
                    expr, indices, g.offset_dims, g.collapsed_slice_dims, g.start_index_map, g.slice_sizes, g.index_vector_dim
                )?;
                Ok(num)
            }
            NoxprNode::Slice(s) => {
                let expr = self.visit(&s.expr, writer)?;
                let num = self.print_var(id, writer)?;
                write!(
                    writer,
                    "slice(expr = var_{}, start_indices = {:?}, stop_indices = {:?}, strides = {:?})",
                    expr, s.start_indices, s.stop_indices, s.strides
                )?;
                Ok(num)
            }
            NoxprNode::DynamicSlice(d) => {
                let expr = self.visit(&d.expr, writer)?;
                let start_indices: Vec<_> = d
                    .start_indices
                    .iter()
                    .map(|e| self.visit(e, writer))
                    .collect::<Result<_, _>>()?;
                let num = self.print_var(id, writer)?;
                write!(
                    writer,
                    "dynamic_slice(expr = var_{}, start_indices = [",
                    expr
                )?;
                for num in start_indices {
                    write!(writer, "var_{}, ", num)?;
                }
                write!(writer, "], slice_sizes = {:?})", &d.size_indices[..])?;
                Ok(num)
            }
            NoxprNode::DynamicUpdateSlice(d) => {
                let expr = self.visit(&d.expr, writer)?;
                let update = self.visit(&d.update, writer)?;
                let start_indices: Vec<_> = d
                    .start_indices
                    .iter()
                    .map(|e| self.visit(e, writer))
                    .collect::<Result<_, _>>()?;
                let num = self.print_var(id, writer)?;
                write!(
                    writer,
                    "dynamic_update_slice(expr = var_{}, update = var_{}, start_indices = [",
                    expr, update
                )?;
                for num in start_indices {
                    write!(writer, "var_{}, ", num)?;
                }
                write!(writer, "])")?;
                Ok(num)
            }
            NoxprNode::Scan(s) => {
                let inputs = s
                    .inputs
                    .iter()
                    .map(|e| self.visit(e, writer).map(|n| format!("var_{}", n)))
                    .collect::<Result<Vec<_>, _>>()?;
                let init = self.visit(&s.initial_state, writer)?;
                let num = self.print_var(id, writer)?;
                write!(
                    writer,
                    "scan(inputs = var_{:?}, xs = init_{}, fn = ",
                    &inputs, init
                )?;
                s.scan_fn.pretty_print(self, writer)?;
                write!(writer, ")")?;
                Ok(num)
            }
            #[cfg(feature = "jax")]
            NoxprNode::Jax(j) => {
                let num = self.print_var(id, writer)?;
                write!(writer, "jax({:?})", j)?;
                Ok(num)
            }
            NoxprNode::Convert(c) => self.visit(&c.arg, writer),
            NoxprNode::Select(s) => {
                let cond = self.visit(&s.cond, writer)?;
                let on_true = self.visit(&s.on_true, writer)?;
                let on_false = self.visit(&s.on_false, writer)?;
                let num = self.print_var(id, writer)?;
                write!(
                    writer,
                    "select(cond = var_{}, on_true = var_{}, on_false = var_{}",
                    cond, on_true, on_false
                )?;
                Ok(num)
            }
            NoxprNode::Call(_) => {
                let num = self.print_var(id, writer)?;
                Ok(num)
            }
            NoxprNode::Cholesky(c) => {
                let arg = self.visit(&c.arg, writer)?;
                let num = self.print_var(id, writer)?;
                write!(writer, "cholesky(var_{}, upper = {})", arg, c.upper)?;
                Ok(num)
            }
            NoxprNode::LuInverse(c) => {
                let arg = self.visit(&c.arg, writer)?;
                let num = self.print_var(id, writer)?;
                write!(writer, "lu_inverse(var_{})", arg,)?;
                Ok(num)
            }
        };
        let num = var_name;
        write!(writer, ": {:?}", expr.shape())?;
        writeln!(writer)?;
        num
    }

    /// Specifically handles printing of binary operations.
    fn visit_binary_op(
        &mut self,
        id: NoxprId,
        op: &BinaryOp,
        op_label: &str,
        writer: &mut impl std::fmt::Write,
    ) -> Result<usize, std::fmt::Error> {
        let lhs = self.visit(&op.lhs, writer)?;
        let rhs = self.visit(&op.rhs, writer)?;
        let num = self.print_var(id, writer)?;
        write!(writer, "{} {} {}", lhs, op_label, rhs)?;
        Ok(num)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor, Client, Collapse, CompFn, Const, Matrix, Scalar, Tensor, Vector};

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
                panic!("{}", msg);
            }
            Err(e) => {
                panic!("{:?}", e);
            }
        };
        let out = exec
            .run(&client, tensor![1.0f32, 2.0, 3.0, 5.0, 6.0])
            .unwrap()
            .to_host();
        assert_eq!(out, tensor![2.0, 3.0, 4.0, 6.0, 7.0])
    }

    #[test]
    fn test_unary_vmap() {
        let client = Client::cpu().unwrap();
        fn add_one(mat: Vector<f32, 5>) -> Vector<f32, 5> {
            mat.vmap(|x: Scalar<f32>| x.sqrt()).unwrap().collapse()
        }
        let comp = add_one.build().unwrap();
        let exec = match comp.compile(&client) {
            Ok(exec) => exec,
            Err(xla::Error::XlaError { msg, .. }) => {
                panic!("{}", msg);
            }
            Err(e) => {
                panic!("{:?}", e);
            }
        };
        let out = exec
            .run(&client, tensor![4.0f32, 9.0, 16.0, 25.0, 36.0])
            .unwrap()
            .to_host();
        assert_eq!(out, tensor![2.0, 3.0, 4.0, 5.0, 6.0])
    }

    #[test]
    fn test_broadcast_vmap() {
        let client = Client::cpu().unwrap();
        fn add_one(mat: Vector<f32, 5>) -> Vector<f32, 5> {
            mat.vmap(|_| Scalar::from(1.0)).unwrap().collapse()
        }
        let comp = add_one.build().unwrap();
        let exec = match comp.compile(&client) {
            Ok(exec) => exec,
            Err(xla::Error::XlaError { msg, .. }) => {
                panic!("{}", msg);
            }
            Err(e) => {
                panic!("{:?}", e);
            }
        };
        let out = exec
            .run(&client, tensor![4.0f32, 9.0, 16.0, 25.0, 36.0])
            .unwrap()
            .to_host();
        assert_eq!(out, tensor![1.0, 1.0, 1.0, 1.0, 1.0])
    }

    #[test]
    fn test_matrix_vmap() {
        let client = Client::cpu().unwrap();
        fn add_one(mat: Matrix<f32, 2, 2>) -> Matrix<f32, 2, 2> {
            mat.vmap(|x: Vector<f32, 2>| x).unwrap().collapse()
        }
        let comp = add_one.build().unwrap();
        let exec = match comp.compile(&client) {
            Ok(exec) => exec,
            Err(xla::Error::XlaError { msg, .. }) => {
                panic!("{}", msg);
            }
            Err(e) => {
                panic!("{:?}", e);
            }
        };
        let out = exec
            .run(&client, tensor![[1.0, 2.0], [3.0, 4.0]])
            .unwrap()
            .to_host();
        assert_eq!(out, tensor![[1.0, 2.0], [3.0, 4.0]])
    }

    #[test]
    fn test_dot_vmap() {
        let client = Client::cpu().unwrap();
        fn add_one(
            mat: Tensor<f32, (Const<2>, Const<2>, Const<2>)>,
        ) -> Tensor<f32, (Const<2>, Const<2>, Const<2>)> {
            mat.vmap(|x: Matrix<f32, 2, 2>| x.clone().dot(&x))
                .unwrap()
                .collapse()
        }
        let comp = add_one.build().unwrap();
        let exec = match comp.compile(&client) {
            Ok(exec) => exec,
            Err(xla::Error::XlaError { msg, .. }) => {
                panic!("{}", msg);
            }
            Err(e) => {
                panic!("{:?}", e);
            }
        };
        let out = exec
            .run(
                &client,
                tensor![[[1.0, 2.0], [3.0, 4.0f32]], [[5.0, 8.0], [9.0, 10.0]]],
            )
            .unwrap()
            .to_host();
        assert_eq!(
            out,
            tensor![
                [[7.0, 10.0], [15.0, 22.0,]],
                [[97.0, 120.0,], [135.0, 172.0]]
            ]
        );
    }

    #[test]
    fn test_broadcast_dims() {
        let a = &[1, 6];
        let b = &[];
        let out = crate::broadcast_dims(a, b);
        assert_eq!(out, Some(smallvec::smallvec![1, 6]))
    }

    #[test]
    fn test_normalize_vmap() {
        let client = Client::cpu().unwrap();
        fn norm(q: Matrix<f32, 2, 3>) -> Matrix<f32, 2, 3> {
            q.vmap(|x: Vector<f32, 3>| x.clone() / x.norm())
                .unwrap()
                .collapse()
        }

        let comp = norm.build().unwrap();
        let exec = match comp.compile(&client) {
            Ok(exec) => exec,
            Err(xla::Error::XlaError { msg, .. }) => {
                panic!("{}", msg);
            }
            Err(e) => {
                panic!("{:?}", e);
            }
        };
        let out = exec
            .run(&client, tensor![[1.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
            .unwrap()
            .to_host();
        assert_eq!(out, tensor![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    }

    #[test]
    fn test_scalar_add_scan() {
        let client = Client::cpu().unwrap();
        fn add_one(mat: Vector<f32, 5>) -> Scalar<f32> {
            mat.scan(Scalar::from(0f32), |acc, x| acc + x).unwrap()
        }
        let comp = add_one.build().unwrap();
        let exec = match comp.compile(&client) {
            Ok(exec) => exec,
            Err(xla::Error::XlaError { msg, .. }) => {
                panic!("{}", msg);
            }
            Err(e) => {
                panic!("{:?}", e);
            }
        };
        let out = exec
            .run(&client, tensor![1.0f32, 2.0, 3.0, 5.0, 6.0])
            .unwrap()
            .to_host();
        assert_eq!(out, 17.0.into())
    }

    #[test]
    fn test_vec_add_scan() {
        let client = Client::cpu().unwrap();
        fn add_one(mat: Matrix<f32, 3, 2>) -> Vector<f32, 2> {
            mat.scan(Vector::<f32, 2>::zeros(), |acc, x| acc + x)
                .unwrap()
        }
        let comp = add_one.build().unwrap();
        let exec = match comp.compile(&client) {
            Ok(exec) => exec,
            Err(xla::Error::XlaError { msg, .. }) => {
                panic!("{}", msg);
            }
            Err(e) => {
                panic!("{:?}", e);
            }
        };
        let out = exec
            .run(&client, tensor![[1.0f32, 2.0], [3.0, 5.0], [6.0, 7.0]])
            .unwrap()
            .to_host();
        assert_eq!(out, tensor![10.0, 14.0])
    }

    #[test]
    fn test_scan_order() {
        let client = Client::cpu().unwrap();
        fn acc_only(mat: Matrix<f32, 3, 2>) -> Vector<f32, 2> {
            mat.scan(Vector::<f32, 2>::zeros(), |acc, _| acc).unwrap()
        }
        let comp = acc_only.build().unwrap();
        let exec = match comp.compile(&client) {
            Ok(exec) => exec,
            Err(xla::Error::XlaError { msg, .. }) => {
                panic!("{}", msg);
            }
            Err(e) => {
                panic!("{:?}", e);
            }
        };
        let out = exec
            .run(&client, tensor![[1.0f32, 2.0], [3.0, 5.0], [6.0, 7.0]])
            .unwrap()
            .to_host();
        assert_eq!(out, tensor![0.0, 0.0])
    }

    #[test]
    fn test_vmap_add_scan() {
        let client = Client::cpu().unwrap();
        fn add_one(mat: Matrix<f32, 3, 2>) -> Vector<f32, 3> {
            mat.vmap(|vec: Vector<f32, 2>| vec.scan(Scalar::from(0f32), |acc, x| acc + x).unwrap())
                .unwrap()
                .collapse()
        }
        let comp = add_one.build().unwrap();
        let exec = match comp.compile(&client) {
            Ok(exec) => exec,
            Err(xla::Error::XlaError { msg, .. }) => {
                panic!("{}", msg);
            }
            Err(e) => {
                panic!("{:?}", e);
            }
        };
        let out = exec
            .run(&client, tensor![[1.0f32, 2.0], [3.0, 5.0], [6.0, 7.0]])
            .unwrap()
            .to_host();
        assert_eq!(out, tensor![3.0, 8.0, 13.0])
    }
}
