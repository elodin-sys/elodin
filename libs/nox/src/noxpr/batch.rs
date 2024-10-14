use crate::{
    xla::ElementType, ArrayTy, BinaryOp, CompFn, DefaultMap, DefaultMappedDim, Dim,
    DotDimensionNums, Error, Noxpr, NoxprFn, NoxprId, NoxprNode, NoxprTy, ReplaceDim, ReprMonad,
    Tensor, TensorItem,
};
use smallvec::{smallvec, SmallVec};
use std::{
    collections::HashMap,
    iter,
    ops::{Add, Deref, Div, Mul, Neg, Sub},
};

use super::Op;

/// Helper class for tracing batch operations, useful for operations like batched matrix multiplication.
#[derive(Clone)]
pub struct BatchTracer {
    cache: HashMap<NoxprId, BatchedExpr>,
    out_axis: BatchAxis,
}

/// Represents an expression along with its batch axis information.
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
}

/// Describes the axis along which batch operations are performed.
#[derive(Clone, Debug, PartialEq)]
pub enum BatchAxis {
    NotMapped,
    Mapped { index: usize, size: usize },
}

impl BatchTracer {
    /// Creates a new `BatchTracer` for managing batch operation contexts.
    pub fn new(out_axis: BatchAxis) -> Self {
        Self {
            cache: HashMap::default(),
            out_axis,
        }
    }

    pub fn visit(&mut self, expr: &Noxpr) -> Result<BatchedExpr, Error> {
        self.visit_inner(expr)
    }

    /// Visits a `Noxpr` and applies any batch-specific transformations.
    pub fn visit_inner(&mut self, expr: &Noxpr) -> Result<BatchedExpr, Error> {
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
            NoxprNode::GreaterOrEqual(b) => self.visit_binary_op(b, Noxpr::greater_or_equal)?,
            NoxprNode::LessOrEqual(b) => self.visit_binary_op(b, Noxpr::less_or_equal)?,
            NoxprNode::Less(b) => self.visit_binary_op(b, Noxpr::less)?,
            NoxprNode::Equal(b) => self.visit_binary_op(b, Noxpr::eq)?,
            NoxprNode::Atan2(b) => self.visit_binary_op(b, Noxpr::atan2)?,
            NoxprNode::Sqrt(e) => self.visit_unary_op(e, Noxpr::sqrt)?,
            NoxprNode::Neg(e) => self.visit_unary_op(e, Noxpr::neg)?,
            NoxprNode::Log(e) => self.visit_unary_op(e, Noxpr::log)?,
            NoxprNode::Sin(e) => self.visit_unary_op(e, Noxpr::sin)?,
            NoxprNode::Cos(e) => self.visit_unary_op(e, Noxpr::cos)?,
            NoxprNode::Abs(e) => self.visit_unary_op(e, Noxpr::abs)?,
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
            NoxprNode::Dot(d) => {
                let lhs_rank = d.lhs.shape().ok_or(Error::UnbatchableArgument)?.len();
                let rhs_rank = d.rhs.shape().ok_or(Error::UnbatchableArgument)?.len();
                self.visit_dot_general(
                    &d.lhs,
                    &d.rhs,
                    DotDimensionNums {
                        lhs_contracting_dimensions: smallvec![lhs_rank.saturating_sub(1) as i64],
                        rhs_contracting_dimensions: smallvec![rhs_rank.saturating_sub(2) as i64],
                        ..Default::default()
                    },
                )?
            }
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
            #[cfg(feature = "jax")]
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
            NoxprNode::Scan(s) => {
                let BatchAxis::Mapped { size: out_size, .. } = self.out_axis else {
                    panic!();
                };
                let axis = BatchAxis::Mapped {
                    index: 1,
                    size: out_size,
                };
                let mut inputs: Vec<_> = s
                    .inputs
                    .iter()
                    .map(|e| {
                        self.visit(e).and_then(|b| {
                            let out = b
                                .move_batch_axis(axis.clone())
                                .ok_or(Error::UnbatchableArgument)?;
                            Ok(out)
                        })
                    })
                    .collect::<Result<_, Error>>()?;
                let initial_state = self
                    .visit(&s.initial_state)?
                    .move_batch_axis(self.out_axis.clone())
                    .unwrap();
                let batch_axis = inputs
                    .iter()
                    .find(|i| i.batch_axis != BatchAxis::NotMapped)
                    .map(|i| i.batch_axis.clone())
                    .unwrap_or(BatchAxis::NotMapped);
                match &batch_axis {
                    BatchAxis::NotMapped => {
                        let inputs = inputs.into_iter().map(|i| i.inner).collect();
                        let inner = Noxpr::scan(inputs, initial_state.inner, s.scan_fn.clone());
                        BatchedExpr {
                            inner,
                            batch_axis: BatchAxis::NotMapped,
                        }
                        .move_batch_axis(self.out_axis.clone())
                        .ok_or(Error::UnbatchableArgument)?
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
                                panic!("non param arg in scan function")
                            };
                            let mut p = p.clone();
                            let mut shape =
                                input.inner.shape().ok_or(Error::UnbatchableArgument)?;
                            shape.remove(0);
                            match &mut p.ty {
                                NoxprTy::Tuple(_) => todo!(),
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
                            if let BatchAxis::Mapped { index, .. } = &mut batched_expr.batch_axis {
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
                            if let BatchAxis::Mapped { index, .. } = &mut batched_expr.batch_axis {
                                *index = 0;
                            }

                            *arg = batched_expr.inner.clone();
                            inner_batcher.cache.insert(id, batched_expr);
                        }
                        let inner = inner_batcher.visit(&s.scan_fn.inner)?;
                        let scan_fn = NoxprFn {
                            args,
                            inner: inner.inner,
                        };
                        BatchedExpr {
                            inner: Noxpr::scan(
                                inputs.into_iter().map(|i| i.inner).collect(),
                                initial_state.inner,
                                scan_fn,
                            ),
                            batch_axis: self.out_axis.clone(),
                        }
                        // .move_batch_axis(self.out_axis.clone())
                        // .unwrap()
                    }
                }
            }
            NoxprNode::Convert(c) => {
                let arg = self.visit(&c.arg)?;
                BatchedExpr {
                    inner: arg.inner.convert(c.ty),
                    batch_axis: arg.batch_axis,
                }
            }
            NoxprNode::Select(s) => {
                let cond = self
                    .visit(&s.cond)?
                    .move_batch_axis(self.out_axis.clone())
                    .ok_or(Error::UnbatchableArgument)?;
                let on_true = self
                    .visit(&s.on_true)?
                    .move_batch_axis(self.out_axis.clone())
                    .ok_or(Error::UnbatchableArgument)?;
                let on_false = self
                    .visit(&s.on_false)?
                    .move_batch_axis(self.out_axis.clone())
                    .ok_or(Error::UnbatchableArgument)?;
                // TODO: handle conflating batch axes
                BatchedExpr {
                    inner: Noxpr::select(&cond.inner, on_true.inner, on_false.inner),
                    batch_axis: cond.batch_axis,
                }
            }
            NoxprNode::Call(_) => {
                // TODO(sphw): we have to figure out if we can batch calls at all
                todo!()
            }
            NoxprNode::Cholesky(c) => {
                let arg = self
                    .visit(&c.arg)?
                    .move_batch_axis(self.out_axis.clone())
                    .ok_or(Error::UnbatchableArgument)?;
                BatchedExpr {
                    inner: arg.inner.cholesky(c.upper),
                    batch_axis: arg.batch_axis,
                }
            }
            NoxprNode::LuInverse(_lu) => {
                todo!()
            }
        };
        self.cache.insert(id, op.clone());
        Ok(op)
    }

    /// Specifically handles the dot-general operation in a batched context.
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

    /// Handles the broadcasting of expressions within a batched context.
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

    /// Manages binary operations in a batched context, considering batch axes.
    fn visit_binary_op(
        &mut self,
        op: &BinaryOp,
        func: impl Fn(Noxpr, Noxpr) -> Noxpr,
    ) -> Result<BatchedExpr, Error> {
        let lhs = self.visit(&op.lhs)?;
        let rhs = self.visit(&op.rhs)?;
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
        match (lhs.batch_axis.clone(), rhs.batch_axis.clone()) {
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

    /// Manages unary operations in a batched context.
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
        let expr = tracer.visit(&func.inner).unwrap();
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
