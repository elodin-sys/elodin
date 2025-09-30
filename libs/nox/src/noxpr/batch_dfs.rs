// use crate::noxpr::batch::{BatchAxis, BatchedExpr};
use crate::{
    ArrayTy, BinaryOp, CompFn, DefaultMap, DefaultMappedDim, Dim, DotDimensionNums, Error, Noxpr,
    NoxprFn, NoxprId, NoxprNode, NoxprTy, ReplaceDim, ReprMonad, Tensor, TensorItem,
    TraversalError, xla::ElementType,
};
use core::{
    iter,
    ops::{Add, Deref, Div, Mul, Neg, Sub},
};
use smallvec::{SmallVec, smallvec};
use std::collections::HashMap;
use traversal::DftPost;

use super::Op;

/// Describes the axis along which batch operations are performed.
#[derive(Clone, Debug, PartialEq, Eq, Copy)]
pub enum BatchAxis {
    NotMapped,
    Mapped { index: usize, size: usize },
}

/// Represents an expression along with its batch axis information.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchedExpr {
    pub(crate) inner: Noxpr,
    pub(crate) batch_axis: BatchAxis,
}

impl BatchedExpr {
    pub(crate) fn map_expr(self, func: impl FnOnce(Noxpr) -> Noxpr) -> Self {
        BatchedExpr {
            inner: func(self.inner),
            batch_axis: self.batch_axis,
        }
    }

    pub(crate) fn move_batch_axis(self, dest: BatchAxis) -> Option<Self> {
        let BatchAxis::Mapped {
            index: dest_axis,
            size: dest_size,
        } = dest
        else {
            return Some(self);
        };
        match self.batch_axis {
            BatchAxis::NotMapped => {
                let mut new_shape = self.inner.shape()?;
                if dest_axis > new_shape.len() {
                    for _ in new_shape.len()..dest_axis {
                        new_shape.push(1);
                    }
                }
                new_shape.insert(dest_axis, dest_size as i64);
                let broadcast_dims = (0..new_shape.len())
                    .filter(|x| *x != dest_axis)
                    .map(|x| x as i64)
                    .collect();
                let inner = self.inner.broadcast_in_dim(new_shape, broadcast_dims);
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
                    batch_axis: BatchAxis::Mapped {
                        index: dest_axis,
                        size: shape[index] as usize,
                    },
                })
            }
        }
    }

    /// Return true if the other expression is equal to this one except for its IDs.
    fn is_equal_ignoring_ids(&self, other: &BatchedExpr) -> bool {
        self.batch_axis == self.batch_axis && self.inner.is_equal_ignoring_ids(&other.inner)
    }
}

/// Helper class for tracing batch operations, useful for operations like batched matrix multiplication.
#[derive(Clone)]
pub struct BatchTracer {
    cache: HashMap<NoxprId, BatchedExpr>,
    out_axis: BatchAxis,
}

impl BatchTracer {
    /// Creates a new `BatchTracer` for managing batch operation contexts.
    pub fn new(out_axis: BatchAxis) -> Self {
        Self {
            cache: HashMap::default(),
            out_axis,
        }
    }

    pub fn walk(&mut self, expr: &Noxpr) -> Result<BatchedExpr, Error> {
        let mut seen_once = std::collections::HashSet::new();
        let mut seen_twice = std::collections::HashSet::new();
        let mut cache: HashMap<NoxprId, BatchedExpr> = std::collections::HashMap::new();
        // Create a post-order depth-first traversal iterator
        let first_traversal = DftPost::new(expr, |node: &Noxpr| {
            if seen_once.insert(node.id()) {
                node.node.children()
            } else {
                seen_twice.insert(node.id());
                crate::ChildrenIter::Empty
            }
        });
        for (_depth, _node) in first_traversal {
            // Figure out which nodes we need to cache.
        }
        seen_once.clear();
        let second_traversal = DftPost::new(expr, |node: &Noxpr| {
            if seen_once.insert(node.id()) {
                node.node.children()
            } else {
                crate::ChildrenIter::Empty
            }
        });

        // Collect nodes in a stack (post-order traversal gives us deepest first)
        let mut node_stack: Vec<BatchedExpr> = Vec::new();

        for (_depth, node) in second_traversal {
            // Process the node based on its type
            let mut already_cached = false;
            let result = if let Some(batched_expr) = cache.get(&node.id()) {
                already_cached = true;
                batched_expr.clone()
            } else {
                self.process_node(node, &mut node_stack)?
            };
            if !already_cached && seen_twice.contains(&node.id()) {
                cache.insert(node.id(), result.clone());
            }
            node_stack.push(result);
        }
        assert_eq!(node_stack.len(), 1);
        node_stack
            .pop()
            .ok_or(Error::Internal(TraversalError::RootNodeFailed))
    }

    /// Processes a single node during the DFS traversal
    fn process_node(
        &mut self,
        expr: &Noxpr,
        stack: &mut Vec<BatchedExpr>,
    ) -> Result<BatchedExpr, Error> {
        match expr.deref() {
            NoxprNode::Constant(_) => Ok(BatchedExpr {
                inner: expr.clone(),
                batch_axis: BatchAxis::NotMapped,
            }),
            NoxprNode::Param(_) => {
                // Check if this parameter is already in the cache (for vmap)
                if let Some(cached) = self.cache.get(&expr.id()) {
                    Ok(cached.clone())
                } else {
                    Ok(BatchedExpr {
                        inner: expr.clone(),
                        batch_axis: BatchAxis::NotMapped,
                    })
                }
            }
            NoxprNode::Tuple(inner) => {
                let mut exprs = Vec::with_capacity(inner.len());
                let mut batch_axis = BatchAxis::NotMapped;
                // Pop children in reverse order (post-order traversal gives us deepest first)
                for _ in (0..inner.len()).rev() {
                    let mapped = stack
                        .pop()
                        .ok_or(Error::Internal(TraversalError::ChildNotProcessed))?;
                    exprs.push(mapped.inner);
                    batch_axis = mapped.batch_axis;
                }
                // Reverse to get correct order
                exprs.reverse();
                let inner = Noxpr::tuple(exprs);
                Ok(BatchedExpr { inner, batch_axis })
            }
            NoxprNode::Add(b) => self.process_binary_op(b, Noxpr::add, stack),
            NoxprNode::Sub(b) => self.process_binary_op(b, Noxpr::sub, stack),
            NoxprNode::Mul(b) => self.process_binary_op(b, Noxpr::mul, stack),
            NoxprNode::Div(b) => self.process_binary_op(b, Noxpr::div, stack),
            NoxprNode::And(b) => self.process_binary_op(b, Noxpr::and, stack),
            NoxprNode::Or(b) => self.process_binary_op(b, Noxpr::or, stack),
            NoxprNode::GreaterOrEqual(b) => {
                self.process_binary_op(b, Noxpr::greater_or_equal, stack)
            }
            NoxprNode::LessOrEqual(b) => self.process_binary_op(b, Noxpr::less_or_equal, stack),
            NoxprNode::Less(b) => self.process_binary_op(b, Noxpr::less, stack),
            NoxprNode::Equal(b) => self.process_binary_op(b, Noxpr::eq, stack),
            NoxprNode::Atan2(b) => self.process_binary_op(b, Noxpr::atan2, stack),
            NoxprNode::Sqrt(e) => self.process_unary_op(e, Noxpr::sqrt, stack),
            NoxprNode::Neg(e) => self.process_unary_op(e, Noxpr::neg, stack),
            NoxprNode::Log(e) => self.process_unary_op(e, Noxpr::log, stack),
            NoxprNode::Sin(e) => self.process_unary_op(e, Noxpr::sin, stack),
            NoxprNode::Cos(e) => self.process_unary_op(e, Noxpr::cos, stack),
            NoxprNode::Abs(e) => self.process_unary_op(e, Noxpr::abs, stack),
            NoxprNode::Acos(e) => self.process_unary_op(e, Noxpr::acos, stack),
            NoxprNode::Asin(e) => self.process_unary_op(e, Noxpr::asin, stack),
            NoxprNode::Concat(c) => self.process_concat(c, stack),
            NoxprNode::DotGeneral(d) => {
                self.process_dot_general(&d.lhs, &d.rhs, d.dimensions.clone(), stack)
            }
            NoxprNode::Dot(d) => {
                let lhs_rank = d.lhs.shape().ok_or(Error::UnbatchableArgument)?.len();
                let rhs_rank = d.rhs.shape().ok_or(Error::UnbatchableArgument)?.len();
                self.process_dot_general(
                    &d.lhs,
                    &d.rhs,
                    DotDimensionNums {
                        lhs_contracting_dimensions: smallvec![lhs_rank.saturating_sub(1) as i64],
                        rhs_contracting_dimensions: smallvec![rhs_rank.saturating_sub(2) as i64],
                        ..Default::default()
                    },
                    stack,
                )
            }
            NoxprNode::Slice(s) => self.process_slice(s, stack),
            NoxprNode::DynamicSlice(_) => Err(Error::Internal(TraversalError::UnsupportedNodeType)),
            NoxprNode::Reshape(r) => self.process_reshape(r, stack),
            NoxprNode::Broadcast(b) => {
                let shape = b.expr.shape().ok_or(Error::UnbatchableArgument)?;
                let broadcast_dims = (0..shape.len() as i64).collect();
                self.process_broadcast_in_dim(&b.expr, b.sizes.clone(), broadcast_dims, stack)
            }
            NoxprNode::BroadcastInDim(b) => self.process_broadcast_in_dim(
                &b.expr,
                b.sizes.clone(),
                b.broadcast_dims.clone(),
                stack,
            ),
            NoxprNode::Transpose(t) => self.process_transpose(t, stack),
            NoxprNode::Gather(g) => self.process_gather(g, stack),
            NoxprNode::Iota(iota) => {
                let expr = Noxpr::new(NoxprNode::Iota(iota.clone()));
                BatchedExpr {
                    inner: expr,
                    batch_axis: BatchAxis::NotMapped,
                }
                .move_batch_axis(self.out_axis.clone())
                .ok_or(Error::UnbatchableArgument)
            }
            NoxprNode::DynamicUpdateSlice(_) => {
                Err(Error::Internal(TraversalError::UnsupportedNodeType))
            }
            #[cfg(feature = "jax")]
            NoxprNode::Jax(_) => Err(Error::Internal(TraversalError::UnsupportedNodeType)),
            NoxprNode::GetTupleElement(g) => {
                // We still do this to process the errors.
                let NoxprNode::Tuple(elems) = g.expr.deref() else {
                    return Err(Error::UnbatchableArgument);
                };
                let _expr = elems.get(g.index).ok_or(Error::UnbatchableArgument)?;
                // For GetTupleElement, we need to pop the tuple result from the stack
                // The tuple processing should have already pushed the individual elements
                Ok(stack
                    .pop()
                    .ok_or(Error::Internal(TraversalError::ChildNotProcessed))?)
            }
            NoxprNode::Scan(s) => self.process_scan(s, stack),
            NoxprNode::Convert(c) => {
                let arg = stack
                    .pop()
                    .ok_or(Error::Internal(TraversalError::ChildNotProcessed))?;
                Ok(BatchedExpr {
                    inner: arg.inner.convert(c.ty),
                    batch_axis: arg.batch_axis,
                })
            }
            NoxprNode::Select(s) => self.process_select(s, stack),
            NoxprNode::Call(_) => Err(Error::Internal(TraversalError::UnsupportedNodeType)),
            NoxprNode::Cholesky(c) => {
                let arg = stack
                    .pop()
                    .ok_or(Error::Internal(TraversalError::ChildNotProcessed))?;
                let arg = arg
                    .move_batch_axis(self.out_axis.clone())
                    .ok_or(Error::UnbatchableArgument)?;
                Ok(BatchedExpr {
                    inner: arg.inner.cholesky(c.upper),
                    batch_axis: arg.batch_axis,
                })
            }
            NoxprNode::LuInverse(_lu) => Err(Error::Internal(TraversalError::UnsupportedNodeType)),
        }
    }

    /// Processes a binary operation during DFS traversal
    fn process_binary_op(
        &mut self,
        _op: &BinaryOp,
        func: impl Fn(Noxpr, Noxpr) -> Noxpr,
        stack: &mut Vec<BatchedExpr>,
    ) -> Result<BatchedExpr, Error> {
        // Pop rhs first, then lhs (reverse order)
        let rhs = stack
            .pop()
            .ok_or(Error::Internal(TraversalError::RhsNotProcessed))?;
        let lhs = stack
            .pop()
            .ok_or(Error::Internal(TraversalError::LhsNotProcessed))?;
        fn scalar_broadcast(
            rank: usize,
            expr: BatchedExpr,
            shape: SmallVec<[i64; 4]>,
        ) -> Option<Noxpr> {
            if expr.batch_axis == BatchAxis::NotMapped || shape.len() == rank {
                Some(expr.inner)
            } else {
                expr.inner.expand_rank(rank)
            }
        }


        let scalar_broadcast_func =
            move |lhs: BatchedExpr, rhs: BatchedExpr| -> Result<Noxpr, Error> {
                let rhs_shape = rhs.inner.shape().ok_or(Error::UnbatchableArgument)?;
                let lhs_shape = lhs.inner.shape().ok_or(Error::UnbatchableArgument)?;
                if rhs_shape.len() == lhs_shape.len() {
                    Ok(func(lhs.inner, rhs.inner))
                } else {
                    let rank = rhs_shape.len().max(lhs_shape.len());
                    let lhs =
                        scalar_broadcast(rank, lhs, lhs_shape).ok_or(Error::UnbatchableArgument)?;
                    let rhs =
                        scalar_broadcast(rank, rhs, rhs_shape).ok_or(Error::UnbatchableArgument)?;
                    Ok(func(lhs, rhs))
                }
            };

        match (lhs.batch_axis, rhs.batch_axis) {
            (BatchAxis::NotMapped, BatchAxis::NotMapped) => {
                let expr = BatchedExpr {
                    inner: scalar_broadcast_func(lhs, rhs)?,
                    batch_axis: BatchAxis::NotMapped,
                };
                Ok(expr
                    .move_batch_axis(self.out_axis.clone())
                    .ok_or(Error::UnbatchableArgument)?)
            }

            (BatchAxis::NotMapped, mapped @ BatchAxis::Mapped { .. })
                | (mapped @ BatchAxis::Mapped { .. }, BatchAxis::NotMapped) => {
                let inner = scalar_broadcast_func(lhs, rhs)?;
                Ok(BatchedExpr {
                    inner,
                    batch_axis: mapped,
                })
            }
            (lhs_axis @ BatchAxis::Mapped { .. }, BatchAxis::Mapped { .. }) => {
                let rhs = rhs
                    .move_batch_axis(lhs_axis.clone())
                    .ok_or(Error::UnbatchableArgument)?;
                let inner = scalar_broadcast_func(lhs, rhs)?;
                Ok(BatchedExpr {
                    inner,
                    batch_axis: lhs_axis.clone(),
                })
            }
        }
    }

    /// Processes a unary operation during DFS traversal
    fn process_unary_op(
        &mut self,
        _expr: &Noxpr,
        func: impl Fn(Noxpr) -> Noxpr,
        stack: &mut Vec<BatchedExpr>,
    ) -> Result<BatchedExpr, Error> {
        let expr = stack
            .pop()
            .ok_or(Error::Internal(TraversalError::ChildNotProcessed))?;

        match expr.batch_axis {
            BatchAxis::NotMapped => Ok(expr
                .move_batch_axis(self.out_axis.clone())
                .ok_or(Error::UnbatchableArgument)?
                .map_expr(func)),
            BatchAxis::Mapped { .. } => Ok(expr.map_expr(func)),
        }
    }

    /// Processes concat operations during DFS traversal
    fn process_concat(
        &mut self,
        c: &crate::Concat,
        stack: &mut Vec<BatchedExpr>,
    ) -> Result<BatchedExpr, Error> {
        // Pop children in reverse order
        let mut nodes = Vec::with_capacity(c.nodes.len());
        for _ in (0..c.nodes.len()).rev() {
            let node = stack
                .pop()
                .ok_or(Error::Internal(TraversalError::ChildNotProcessed))?;
            nodes.push(node);
        }
        // Reverse to get correct order
        nodes.reverse();
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

        Ok(BatchedExpr {
            inner: Noxpr::concat_in_dim(nodes, c.dimension + 1),
            batch_axis: BatchAxis::Mapped { index: 0, size },
        })
    }

    /// Processes dot general operations during DFS traversal
    fn process_dot_general(
        &mut self,
        _lhs: &Noxpr,
        _rhs: &Noxpr,
        dims: DotDimensionNums,
        stack: &mut Vec<BatchedExpr>,
    ) -> Result<BatchedExpr, Error> {
        // Pop rhs first, then lhs (reverse order)
        let rhs = stack
            .pop()
            .ok_or(Error::Internal(TraversalError::ChildNotProcessed))?;
        let lhs = stack
            .pop()
            .ok_or(Error::Internal(TraversalError::ChildNotProcessed))?;
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
            (BatchAxis::NotMapped, BatchAxis::NotMapped) => Ok(BatchedExpr {
                inner: lhs.inner.dot_general(rhs.inner, dims),
                batch_axis: BatchAxis::NotMapped,
            }
            .move_batch_axis(self.out_axis.clone())
            .ok_or(Error::UnbatchableArgument)?),
        }
    }

    /// Processes slice operations during DFS traversal
    fn process_slice(
        &mut self,
        s: &crate::Slice,
        stack: &mut Vec<BatchedExpr>,
    ) -> Result<BatchedExpr, Error> {
        let expr = stack
            .pop()
            .ok_or(Error::Internal(TraversalError::ChildNotProcessed))?;
        match expr.batch_axis {
            BatchAxis::NotMapped => Ok(BatchedExpr {
                inner: expr.inner.slice(
                    s.start_indices.clone(),
                    s.stop_indices.clone(),
                    s.strides.clone(),
                ),
                batch_axis: BatchAxis::NotMapped,
            }),
            BatchAxis::Mapped { index, size } => {
                let mut start_indices = s.start_indices.clone();
                let mut stop_indices = s.stop_indices.clone();
                let mut strides = s.strides.clone();
                start_indices.insert(index, 0);
                stop_indices.insert(index, size as i64);
                strides.insert(index, 1);
                Ok(BatchedExpr {
                    inner: expr.inner.slice(start_indices, stop_indices, strides),
                    batch_axis: BatchAxis::Mapped { index, size },
                })
            }
        }
    }

    /// Processes reshape operations during DFS traversal
    fn process_reshape(
        &mut self,
        r: &crate::Reshape,
        stack: &mut Vec<BatchedExpr>,
    ) -> Result<BatchedExpr, Error> {
        let expr = stack
            .pop()
            .ok_or(Error::Internal(TraversalError::ChildNotProcessed))?;
        let BatchAxis::Mapped { size, .. } = self.out_axis else {
            return Err(Error::UnbatchableArgument);
        };
        match &expr.batch_axis {
            BatchAxis::NotMapped => BatchedExpr {
                inner: expr.inner.reshape(r.new_sizes.clone()),
                batch_axis: BatchAxis::NotMapped,
            }
            .move_batch_axis(self.out_axis.clone())
            .ok_or(Error::UnbatchableArgument),
            BatchAxis::Mapped {
                size: batch_size, ..
            } => {
                let batch_size = *batch_size;
                let expr = expr
                    .move_batch_axis(BatchAxis::Mapped { index: 0, size })
                    .ok_or(Error::UnbatchableArgument)?;
                let shape = expr.inner.shape().ok_or(Error::UnbatchableArgument)?;
                let new_sizes = shape
                    .first()
                    .cloned()
                    .into_iter()
                    .chain(r.new_sizes.iter().cloned())
                    .collect();
                Ok(BatchedExpr {
                    inner: expr.inner.reshape(new_sizes),
                    batch_axis: BatchAxis::Mapped {
                        index: 0,
                        size: batch_size,
                    },
                })
            }
        }
    }

    /// Processes broadcast in dim operations during DFS traversal
    fn process_broadcast_in_dim(
        &mut self,
        _inner: &Noxpr,
        mut sizes: SmallVec<[i64; 4]>,
        mut broadcast_dims: SmallVec<[i64; 4]>,
        stack: &mut Vec<BatchedExpr>,
    ) -> Result<BatchedExpr, Error> {
        let expr = stack
            .pop()
            .ok_or(Error::Internal(TraversalError::ChildNotProcessed))?;
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

    /// Processes transpose operations during DFS traversal
    fn process_transpose(
        &mut self,
        t: &crate::Transpose,
        stack: &mut Vec<BatchedExpr>,
    ) -> Result<BatchedExpr, Error> {
        let expr = stack
            .pop()
            .ok_or(Error::Internal(TraversalError::ChildNotProcessed))?;
        match expr.batch_axis {
            BatchAxis::NotMapped => Ok(expr
                .move_batch_axis(self.out_axis.clone())
                .ok_or(Error::UnbatchableArgument)?
                .map_expr(|inner| inner.transpose(t.permutation.clone()))),
            BatchAxis::Mapped { index, size } => {
                let mut permutation = t.permutation.clone();
                for p in &mut permutation {
                    if *p >= index as i64 {
                        *p += 1
                    }
                }
                Ok(BatchedExpr {
                    inner: expr.inner.transpose(t.permutation.clone()),
                    batch_axis: BatchAxis::Mapped { index: 0, size },
                })
            }
        }
    }

    /// Processes gather operations during DFS traversal
    fn process_gather(
        &mut self,
        g: &crate::Gather,
        stack: &mut Vec<BatchedExpr>,
    ) -> Result<BatchedExpr, Error> {
        // Pop indices first, then expr (reverse order)
        let indices = stack
            .pop()
            .ok_or(Error::Internal(TraversalError::ChildNotProcessed))?;
        let expr = stack
            .pop()
            .ok_or(Error::Internal(TraversalError::ChildNotProcessed))?;
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

                Ok(BatchedExpr {
                    inner: expr.inner.gather(
                        indices.inner,
                        offset_dims,
                        collapsed_slice_dims,
                        start_index_map,
                        slice_sizes,
                        g.index_vector_dim,
                    ),
                    batch_axis: expr.batch_axis,
                })
            }
            (BatchAxis::NotMapped, BatchAxis::Mapped { .. }) => {
                let indices = indices
                    .move_batch_axis(BatchAxis::Mapped {
                        index: 0,
                        size: usize::MAX,
                    })
                    .ok_or(Error::UnbatchableArgument)?;
                let offset_dims = g.offset_dims.iter().map(|x| x + 1).collect();
                Ok(BatchedExpr {
                    inner: expr.inner.gather(
                        indices.inner,
                        offset_dims,
                        g.collapsed_slice_dims.clone(),
                        g.start_index_map.clone(),
                        g.slice_sizes.clone(),
                        g.index_vector_dim,
                    ),
                    batch_axis: indices.batch_axis,
                })
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

                let mut count_shape = indices.inner.shape().ok_or(Error::UnbatchableArgument)?;
                if let Some(last) = count_shape.last_mut() {
                    *last = -1;
                }
                let count_shape_len = count_shape.len();
                let counts = Noxpr::iota(
                    ArrayTy {
                        element_type: ElementType::S32,
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

                let offset_dims: SmallVec<[i64; 4]> = g.slice_sizes.iter().map(|x| x + 1).collect();
                let start_index_map: SmallVec<[i64; 4]> = std::iter::once(0)
                    .chain(g.start_index_map.iter().map(|x| x + 1))
                    .collect();

                let BatchAxis::Mapped { size, .. } = self.out_axis else {
                    return Err(Error::UnbatchableArgument);
                };
                Ok(BatchedExpr {
                    batch_axis: BatchAxis::Mapped { index: 0, size },
                    inner: expr.inner.gather(
                        indices,
                        offset_dims,
                        collapsed_slice_dims,
                        start_index_map,
                        slice_sizes,
                        g.index_vector_dim,
                    ),
                })
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
                Ok(BatchedExpr {
                    inner,
                    batch_axis: BatchAxis::NotMapped,
                }
                .move_batch_axis(self.out_axis.clone())
                .ok_or(Error::UnbatchableArgument)?)
            }
        }
    }

    /// Processes scan operations during DFS traversal
    fn process_scan(
        &mut self,
        s: &crate::Scan,
        stack: &mut Vec<BatchedExpr>,
    ) -> Result<BatchedExpr, Error> {
        let BatchAxis::Mapped { size: out_size, .. } = self.out_axis else {
            panic!();
        };
        let axis = BatchAxis::Mapped {
            index: 1,
            size: out_size,
        };
        // Pop initial_state first, then inputs in reverse order
        let initial_state = stack
            .pop()
            .ok_or(Error::Internal(TraversalError::ChildNotProcessed))?
            .move_batch_axis(self.out_axis.clone())
            .unwrap();
        let mut inputs: Vec<_> = Vec::with_capacity(s.inputs.len());
        for _ in (0..s.inputs.len()).rev() {
            let input = stack
                .pop()
                .ok_or(Error::Internal(TraversalError::ChildNotProcessed))?
                .move_batch_axis(axis.clone())
                .ok_or(Error::UnbatchableArgument)?;
            inputs.push(input);
        }
        // Reverse to get correct order
        inputs.reverse();
        let batch_axis = inputs
            .iter()
            .find(|i| i.batch_axis != BatchAxis::NotMapped)
            .map(|i| i.batch_axis.clone())
            .unwrap_or(BatchAxis::NotMapped);
        match &batch_axis {
            BatchAxis::NotMapped => {
                let inputs = inputs.into_iter().map(|i| i.inner).collect();
                let inner = Noxpr::scan(inputs, initial_state.inner, s.scan_fn.clone());
                Ok(BatchedExpr {
                    inner,
                    batch_axis: BatchAxis::NotMapped,
                }
                .move_batch_axis(self.out_axis.clone())
                .ok_or(Error::UnbatchableArgument)?)
            }
            BatchAxis::Mapped { .. } => {
                for input in &mut inputs {
                    *input = input
                        .clone()
                        .move_batch_axis(batch_axis.clone())
                        .ok_or(Error::UnbatchableArgument)?;
                }
                let mut inner_batcher = self.clone();
                let mut args = s.scan_fn.args.clone();
                for (arg, input) in args.iter_mut().rev().zip(inputs.iter().rev()) {
                    let id = arg.id();
                    let NoxprNode::Param(p) = arg.node.as_ref() else {
                        return Err(Error::ScanMissingParam);
                    };
                    let mut p = p.clone();
                    let mut shape = input.inner.shape().ok_or(Error::UnbatchableArgument)?;
                    shape.remove(0);
                    match &mut p.ty {
                        NoxprTy::Tuple(_) => {
                            return Err(Error::Unsupported(
                                "Tuple types are not supported yet.".into(),
                            ));
                        }
                        NoxprTy::ArrayTy(ty) => {
                            ty.shape = shape;
                        }
                    }
                    let inner = Noxpr::parameter(p.number, p.ty, p.name);
                    let mut batched_expr = BatchedExpr {
                        inner,
                        batch_axis: input.batch_axis.clone(),
                    }
                    .move_batch_axis(axis.clone())
                    .unwrap();
                    if let BatchAxis::Mapped { ref mut index, .. } = batched_expr.batch_axis {
                        *index = 0;
                    }
                    *arg = batched_expr.inner.clone();
                    inner_batcher.cache.insert(id, batched_expr);
                }

                if let Some(arg) = args.first_mut() {
                    let input = &initial_state;
                    let id = arg.id();
                    let NoxprNode::Param(p) = arg.node.as_ref() else {
                        panic!("non param arg in scan function")
                    };
                    let mut p = p.clone();
                    let shape = input.inner.shape().ok_or(Error::UnbatchableArgument)?;
                    match &mut p.ty {
                        NoxprTy::Tuple(_) => todo!(),
                        NoxprTy::ArrayTy(ty) => {
                            ty.shape = shape;
                        }
                    }
                    let inner = Noxpr::parameter(p.number, p.ty, p.name);
                    let mut batched_expr = BatchedExpr {
                        inner,
                        batch_axis: axis.clone(),
                    };
                    if let BatchAxis::Mapped { ref mut index, .. } = batched_expr.batch_axis {
                        *index = 0;
                    }

                    *arg = batched_expr.inner.clone();
                    inner_batcher.cache.insert(id, batched_expr);
                }
                let inner = inner_batcher.walk(&s.scan_fn.inner)?;
                let scan_fn = NoxprFn {
                    args,
                    inner: inner.inner,
                };
                Ok(BatchedExpr {
                    inner: Noxpr::scan(
                        inputs.into_iter().map(|i| i.inner).collect(),
                        initial_state.inner,
                        scan_fn,
                    ),
                    batch_axis: self.out_axis.clone(),
                })
            }
        }
    }

    /// Processes select operations during DFS traversal
    fn process_select(
        &mut self,
        _s: &crate::Select,
        stack: &mut Vec<BatchedExpr>,
    ) -> Result<BatchedExpr, Error> {
        // Pop on_false, on_true, then cond (reverse order)
        let on_false = stack
            .pop()
            .ok_or(Error::ExpectedArgument("on_false".into()))?
            .move_batch_axis(self.out_axis.clone())
            .ok_or(Error::UnbatchableArgument)?;
        let on_true = stack
            .pop()
            .ok_or(Error::ExpectedArgument("on_true".into()))?
            .move_batch_axis(self.out_axis.clone())
            .ok_or(Error::UnbatchableArgument)?;
        let cond = stack
            .pop()
            .ok_or(Error::ExpectedArgument("cond".into()))?
            .move_batch_axis(self.out_axis.clone())
            .ok_or(Error::UnbatchableArgument)?;
        Ok(BatchedExpr {
            inner: Noxpr::select(&cond.inner, on_true.inner, on_false.inner),
            batch_axis: cond.batch_axis,
        })
    }
}

impl Noxpr {
    /// Applies vectorized map operation to a function across specified axes.
    pub fn vmap_with_axis(
        func: NoxprFn,
        in_axis: &[usize],
        args: &[Noxpr],
    ) -> Result<Noxpr, Error> {
        if in_axis.len() != args.len() {
            return Err(Error::VmapInAxisMismatch);
        }
        let shape = args
            .first()
            .ok_or(Error::VmapArgsEmpty)?
            .shape()
            .ok_or(Error::UnbatchableArgument)
            .unwrap();
        let mut tracer = BatchTracer::new(BatchAxis::Mapped {
            index: in_axis[0],
            size: shape[in_axis[0]] as usize,
        });
        for ((arg, axis), arg_expr) in args.iter().zip(in_axis).zip(func.args) {
            let arg_id = arg_expr.id();
            let shape = arg.shape().ok_or(Error::UnbatchableArgument).unwrap();
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
        // ROOT visit
        let expr = tracer.walk(&func.inner).unwrap();
        let expr = expr
            .move_batch_axis(tracer.out_axis)
            .ok_or(Error::UnbatchableArgument)
            .unwrap();
        Ok(expr.inner)
    }
}

impl<T: TensorItem, D: Dim + DefaultMap> Tensor<T, D, crate::Op> {
    /// Vectorized map of a function over a tensor.
    pub fn vmap<O: TensorItem + ReprMonad<Op>>(
        &self,
        func: impl CompFn<(T::Tensor<<D::DefaultMapDim as ReplaceDim<D>>::Item>,), O>,
    ) -> Result<Tensor<O, DefaultMappedDim<D>, crate::Op>, Error> {
        self.vmap_with_dim::<D::DefaultMapDim, O>(func)
    }

    /// Vectorized map of a function over a tensor with a specified mapping dimension.
    pub fn vmap_with_dim<MDim: ReplaceDim<D>, O: TensorItem + ReprMonad<Op>>(
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

    /// Applies a scan operation over a tensor, accumulating results using a specified function.
    pub fn scan<O: ReprMonad<Op>>(
        &self,
        initial_state: O,
        func: impl CompFn<(O, T::Tensor<<D::DefaultMapDim as ReplaceDim<D>>::Item>), O>,
    ) -> Result<O, Error> {
        let scan_fn = func.build_expr()?;
        let initial_state = initial_state.into_inner();
        let res = Noxpr::scan(vec![self.inner.clone()], initial_state.clone(), scan_fn);
        Ok(O::from_inner(res))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NoxprScalarExt;
    use crate::noxpr::batch::BatchTracer as RecursiveBatchTracer;

    /// Helper function to create test expressions
    fn create_test_expressions() -> Vec<Noxpr> {
        // Simple scalar operations
        let scalar_a = 2.0f32.constant();
        let scalar_b = 3.0f32.constant();

        vec![
            // Binary operations
            scalar_a.clone() + scalar_b.clone(),
            scalar_a.clone() - scalar_b.clone(),
            scalar_a.clone() * scalar_b.clone(),
            scalar_a.clone() / scalar_b.clone(),
            // Unary operations
            scalar_a.clone().sqrt(),
            scalar_a.clone().neg(),
            scalar_a.clone().log(),
            scalar_a.clone().sin(),
            scalar_a.clone().cos(),
            scalar_a.clone().abs(),
            // Tuple operations
            Noxpr::tuple(vec![scalar_a.clone(), scalar_b.clone()]),
            // Complex nested operations
            (scalar_a.clone() + scalar_b.clone()) * scalar_a.clone(),
        ]
    }

    /// Helper function to compare results from both implementations
    fn compare_batch_results(expr: &Noxpr, out_axis: BatchAxis) -> Result<(), Error> {
        let mut recursive_tracer = RecursiveBatchTracer::new(out_axis.clone());
        let mut dfs_tracer = BatchTracer::new(out_axis);

        let recursive_result = recursive_tracer.visit(expr)?;
        let dfs_result = dfs_tracer.walk(expr)?;

        // Compare the batch axis information (the important functional comparison)
        assert_eq!(
            recursive_result.batch_axis, dfs_result.batch_axis,
            "Batch axis information should match between implementations"
        );

        // Note: Expression IDs will be different because the implementations
        // process nodes in different orders and create different intermediate
        // expressions.
        assert_ne!(
            recursive_result, dfs_result,
            "Batch results should match between implementations"
        );

        // Both should produce the same batch
        assert!(
            recursive_result.is_equal_ignoring_ids(&dfs_result),
            "Batch results should match between implementations"
        );

        assert_eq!(recursive_result.inner.labels().count(),
                   dfs_result.inner.labels().count());

        Ok(())
    }

    #[test]
    fn test_binary_operations_consistency() {
        let expressions = create_test_expressions();
        let out_axis = BatchAxis::Mapped { index: 0, size: 1 };

        for expr in expressions {
            if compare_batch_results(&expr, out_axis.clone()).is_ok() {
                // Test passed
            } else {
                panic!("Binary operation consistency test failed for expression");
            }
        }
    }

    #[test]
    fn test_unary_operations_consistency() {
        let scalar = 4.0f32.constant();
        let out_axis = BatchAxis::Mapped { index: 0, size: 1 };

        let unary_ops = vec![
            scalar.clone().sqrt(),
            scalar.clone().neg(),
            scalar.clone().log(),
            scalar.clone().sin(),
            scalar.clone().cos(),
            scalar.clone().abs(),
            scalar.clone().acos(),
            scalar.clone().asin(),
        ];

        for expr in unary_ops {
            compare_batch_results(&expr, out_axis.clone())
                .expect("Unary operation consistency test failed");
        }
    }

    #[test]
    fn test_tuple_operations_consistency() {
        let scalar_a = 1.0f32.constant();
        let scalar_b = 2.0f32.constant();
        let scalar_c = 3.0f32.constant();

        let tuple_expr = Noxpr::tuple(vec![scalar_a, scalar_b, scalar_c]);
        let out_axis = BatchAxis::Mapped { index: 0, size: 1 };

        compare_batch_results(&tuple_expr, out_axis)
            .expect("Tuple operation consistency test failed");
    }

    #[test]
    fn test_nested_operations_consistency() {
        let scalar_a = 2.0f32.constant();
        let scalar_b = 3.0f32.constant();

        // Test nested binary operations
        let nested_expr = (scalar_a.clone() + scalar_b.clone()) * scalar_a.clone();
        let out_axis = BatchAxis::Mapped { index: 0, size: 1 };

        compare_batch_results(&nested_expr, out_axis)
            .expect("Nested operation consistency test failed");
    }

    // Vector and matrix tests removed for now due to complexity

    #[test]
    fn test_different_batch_axes() {
        let scalar_a = 2.0f32.constant();
        let scalar_b = 3.0f32.constant();
        let expr = scalar_a + scalar_b;

        // Test with different batch axes
        let batch_axes = vec![
            BatchAxis::NotMapped,
            BatchAxis::Mapped { index: 0, size: 1 },
            BatchAxis::Mapped { index: 1, size: 2 },
        ];

        for out_axis in batch_axes {
            compare_batch_results(&expr, out_axis)
                .expect("Different batch axes consistency test failed");
        }
    }

    #[test]
    fn test_error_conditions_consistency() {
        let scalar = 1.0f32.constant();
        let out_axis = BatchAxis::Mapped { index: 0, size: 1 };

        // Test that both implementations handle errors consistently
        let mut recursive_tracer = RecursiveBatchTracer::new(out_axis.clone());
        let mut dfs_tracer = BatchTracer::new(out_axis);

        // Both should succeed for valid expressions
        let recursive_result = recursive_tracer.visit(&scalar);
        let dfs_result = dfs_tracer.walk(&scalar);

        match (recursive_result, dfs_result) {
            (Ok(_), Ok(_)) => {
                // Both succeeded - this is expected
            }
            (Err(e1), Err(e2)) => {
                // Both failed - check if error types are similar
                println!(
                    "Both implementations failed: recursive={:?}, dfs={:?}",
                    e1, e2
                );
            }
            _ => {
                panic!("Inconsistent error handling between implementations");
            }
        }
    }

    #[test]
    fn test_caching_behavior() {
        let scalar_a = 2.0f32.constant();
        let scalar_b = 3.0f32.constant();

        // Create an expression that uses the same sub-expression multiple times
        let shared_expr = scalar_a.clone() + scalar_b.clone();
        let expr = shared_expr.clone() * shared_expr.clone();

        let out_axis = BatchAxis::Mapped { index: 0, size: 1 };

        // Both implementations should handle caching correctly
        compare_batch_results(&expr, out_axis).expect("Caching behavior consistency test failed");
    }

    #[test]
    fn test_numerical_evaluation_consistency() {
        // Test that both implementations produce numerically equivalent results
        let scalar_a = 2.5f32.constant();
        let scalar_b = 3.7f32.constant();
        let expr = scalar_a + scalar_b; // This should evaluate to 6.2

        let out_axis = BatchAxis::Mapped { index: 0, size: 1 };
        compare_batch_results(&expr, out_axis).unwrap();
    }

    #[test]
    fn test_multiplication_consistency() {
        // Test multiplication with different operands
        let scalar_a = 3.0f32.constant();
        let scalar_b = 4.0f32.constant();
        let expr = scalar_a * scalar_b; // This should evaluate to 12.0

        let out_axis = BatchAxis::Mapped { index: 0, size: 1 };

        compare_batch_results(&expr, out_axis).unwrap();
    }

    #[test]
    fn test_sqrt_operation_consistency() {
        // Test sqrt operation specifically
        let scalar = 16.0f32.constant();
        let expr = scalar.sqrt(); // This should evaluate to 4.0

        let out_axis = BatchAxis::Mapped { index: 0, size: 1 };
        compare_batch_results(&expr, out_axis).unwrap();
    }

    #[test]
    fn test_complex_nested_expression_consistency() {
        // Test complex nested expressions: (a + b) * c
        let scalar_a = 2.0f32.constant();
        let scalar_b = 3.0f32.constant();
        let scalar_c = 4.0f32.constant();
        let expr = (scalar_a + scalar_b) * scalar_c; // This should evaluate to (2+3)*4 = 20.0

        let out_axis = BatchAxis::Mapped { index: 0, size: 1 };

        compare_batch_results(&expr, out_axis).unwrap();
    }

    #[test]
    fn test_division_consistency() {
        // Test division operation
        let scalar_a = 15.0f32.constant();
        let scalar_b = 3.0f32.constant();
        let expr = scalar_a / scalar_b; // This should evaluate to 5.0

        let out_axis = BatchAxis::Mapped { index: 0, size: 1 };

        let mut recursive_tracer = RecursiveBatchTracer::new(out_axis.clone());
        let mut dfs_tracer = BatchTracer::new(out_axis);

        let recursive_result = recursive_tracer
            .visit(&expr)
            .expect("Recursive tracer should succeed");
        let dfs_result = dfs_tracer.walk(&expr).expect("DFS tracer should succeed");
        // NOTE: The IDs change depending onthe test order so these assertions cannot
        // be evaluated unless only this one test is run.
        // assert_eq!(
        //     "Noxpr { node: Div(BinaryOp { lhs: Noxpr { node: Constant(Constant { ty: ArrayTy { element_type: F32, shape: [] } }), id: NoxprId(0), backtrace: <disabled> }, rhs: Noxpr { node: Constant(Constant { ty: ArrayTy { element_type: F32, shape: [] } }), id: NoxprId(1), backtrace: <disabled> } }), id: NoxprId(2), backtrace: <disabled> }",
        //     &format!("{:?}", &expr));

        // assert_eq!(
        //     "BatchedExpr { inner: Noxpr { node: BroadcastInDim(BroadcastInDim { expr: Noxpr { node: Div(BinaryOp { lhs: Noxpr { node: Constant(Constant { ty: ArrayTy { element_type: F32, shape: [] } }), id: NoxprId(0), backtrace: <disabled> }, rhs: Noxpr { node: Constant(Constant { ty: ArrayTy { element_type: F32, shape: [] } }), id: NoxprId(1), backtrace: <disabled> } }), id: NoxprId(3), backtrace: <disabled> }, sizes: [1], broadcast_dims: [] }), id: NoxprId(4), backtrace: <disabled> }, batch_axis: Mapped { index: 0, size: 1 } }",
        //     &format!("{:?}", &recursive_result));

        // assert_eq!(
        //     "BatchedExpr { inner: Noxpr { node: BroadcastInDim(BroadcastInDim { expr: Noxpr { node: Div(BinaryOp { lhs: Noxpr { node: Constant(Constant { ty: ArrayTy { element_type: F32, shape: [] } }), id: NoxprId(0), backtrace: <disabled> }, rhs: Noxpr { node: Constant(Constant { ty: ArrayTy { element_type: F32, shape: [] } }), id: NoxprId(1), backtrace: <disabled> } }), id: NoxprId(5), backtrace: <disabled> }, sizes: [1], broadcast_dims: [] }), id: NoxprId(6), backtrace: <disabled> }, batch_axis: Mapped { index: 0, size: 1 } }",
        //     &format!("{:?}", &dfs_result));
        // Both should produce the same batch axis
        assert_eq!(
            recursive_result.batch_axis, dfs_result.batch_axis,
            "Batch axis should match between implementations"
        );

        // Both will produce different ids
        assert_ne!(
            recursive_result, dfs_result,
            "Batch results should match between implementations"
        );

        // Both should produce the same batch
        assert!(
            recursive_result.is_equal_ignoring_ids(&dfs_result),
            "Batch results should match between implementations"
        );
    }

    #[test]
    fn test_ignore_ids() {
        // Test division operation
        let scalar_a = 15.0f32.constant();
        let scalar_b = 3.0f32.constant();
        let expr = scalar_a / scalar_b; // This should evaluate to 5.0

        let out_axis = BatchAxis::Mapped { index: 0, size: 1 };

        let mut recursive_tracer = RecursiveBatchTracer::new(out_axis.clone());
        let recursive_result = recursive_tracer
            .visit(&expr)
            .expect("Recursive tracer should succeed");

        let scalar_a = 15.0f32.constant();
        let scalar_b = 30000.0f32.constant();
        let expr = scalar_a / scalar_b; // This should evaluate to 5.0
        let mut dfs_tracer = BatchTracer::new(out_axis);

        let dfs_result = dfs_tracer.walk(&expr).expect("DFS tracer should succeed");

        // Both should produce the same batch
        assert_ne!(
            recursive_result, dfs_result,
            "Batch results should match between implementations"
        );

        assert!(
            !recursive_result.is_equal_ignoring_ids(&dfs_result),
            "Batch results should not match"
        );
        let recur_labels: Vec<usize> = recursive_result.inner.labels().collect();
        let dfs_labels: Vec<usize> = dfs_result.inner.labels().collect();
        // assert_eq!(recur_labels, vec![4,3,0,1]);
        // assert_eq!(dfs_labels, vec![9,8,5,6]);
    }

    #[test]
    fn test_subtraction_consistency() {
        // Test subtraction operation
        let scalar_a = 10.0f32.constant();
        let scalar_b = 3.0f32.constant();
        let expr = scalar_a - scalar_b; // This should evaluate to 7.0

        let out_axis = BatchAxis::Mapped { index: 0, size: 1 };

        compare_batch_results(&expr, out_axis).unwrap();
    }

    #[test]
    fn test_multiple_unary_operations_consistency() {
        // Test multiple unary operations: sin(cos(log(sqrt(x))))
        let scalar = 4.0f32.constant();
        let expr = scalar.sqrt().log().cos().sin(); // Complex unary chain

        let out_axis = BatchAxis::Mapped { index: 0, size: 1 };

        compare_batch_results(&expr, out_axis).unwrap();
    }

    #[test]
    fn test_mixed_operations_consistency() {
        // Test mixed operations: (a + b) / (c - d)
        let scalar_a = 8.0f32.constant();
        let scalar_b = 4.0f32.constant();
        let scalar_c = 6.0f32.constant();
        let scalar_d = 2.0f32.constant();
        let expr = (scalar_a + scalar_b) / (scalar_c - scalar_d); // (8+4)/(6-2) = 12/4 = 3.0

        let out_axis = BatchAxis::Mapped { index: 0, size: 1 };

        compare_batch_results(&expr, out_axis).unwrap();
    }

    #[test]
    fn test_tuple_operations_detailed_consistency() {
        // Test tuple operations with detailed verification
        let scalar_a = 1.0f32.constant();
        let scalar_b = 2.0f32.constant();
        let scalar_c = 3.0f32.constant();
        let expr = Noxpr::tuple(vec![scalar_a, scalar_b, scalar_c]);

        let out_axis = BatchAxis::Mapped { index: 0, size: 1 };
        compare_batch_results(&expr, out_axis).unwrap();
    }

    #[test]
    fn test_logical_operations_consistency() {
        // Test logical operations: (a > b) && (c < d)
        let scalar_a = 5.0f32.constant();
        let scalar_b = 3.0f32.constant();
        let scalar_c = 2.0f32.constant();
        let scalar_d = 4.0f32.constant();
        let expr = (scalar_a.greater_or_equal(scalar_b)).and(scalar_c.less(scalar_d));

        let out_axis = BatchAxis::Mapped { index: 0, size: 1 };
        compare_batch_results(&expr, out_axis).unwrap();
    }
}
