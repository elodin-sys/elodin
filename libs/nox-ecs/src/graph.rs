use bytes::Buf;
use conduit::{ComponentType, ComponentValue, EntityId};
use nox::{xla::Literal, ArrayTy, CompFn, FromBuilder, IntoOp, Noxpr, NoxprFn, NoxprTy};
use std::{collections::BTreeMap, marker::PhantomData};

use crate::{Component, ComponentArray, ComponentGroup, PipelineBuilder, Query, SystemParam};

#[derive(Clone)]
pub struct GraphQuery<E> {
    pub edges: Vec<Edge>,
    pub phantom_data: PhantomData<E>,
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub from: EntityId,
    pub to: EntityId,
}

impl Edge {
    pub fn new(from: impl Into<EntityId>, to: impl Into<EntityId>) -> Self {
        let from = from.into();
        let to = to.into();
        Self { from, to }
    }

    pub fn reverse(&self) -> Self {
        Self {
            from: self.to,
            to: self.from,
        }
    }
}

impl IntoOp for Edge {
    fn into_op(self) -> Noxpr {
        Noxpr::constant(
            Literal::vector(&[self.from.0, self.to.0]),
            ArrayTy {
                element_type: nox::xla::ElementType::U64,
                shape: smallvec::smallvec![2],
            },
        )
    }
}

impl FromBuilder for Edge {
    type Item<'a> = Self;

    fn from_builder(_builder: &nox::Builder) -> Self::Item<'_> {
        todo!()
    }
}

impl conduit::Component for Edge {
    const NAME: &'static str = "edge";

    fn component_type() -> conduit::ComponentType {
        ComponentType {
            primitive_ty: conduit::PrimitiveTy::U64,
            shape: smallvec::smallvec![2],
        }
    }
}

impl Component for Edge {}

pub trait EdgeComponent: Component {
    fn to_edge(&self) -> Edge;
    fn from_value(value: ComponentValue<'_>) -> Option<Self>
    where
        Self: Sized;
}

impl EdgeComponent for Edge {
    fn to_edge(&self) -> Edge {
        self.clone()
    }

    fn from_value(value: ComponentValue<'_>) -> Option<Self>
    where
        Self: Sized,
    {
        let ComponentValue::U64(val) = value else {
            return None;
        };
        let from = EntityId(*val.get(0)?);
        let to = EntityId(*val.get(1)?);
        Some(Edge { from, to })
    }
}

pub struct TotalEdge;

impl<E: EdgeComponent + 'static> SystemParam<PipelineBuilder> for GraphQuery<E> {
    type Item = Self;

    fn init(_: &mut crate::PipelineBuilder) -> Result<(), crate::Error> {
        Ok(())
    }

    fn from_builder(builder: &crate::PipelineBuilder) -> Self::Item {
        let col = builder.world.column::<E>().unwrap();
        let ty = &col.metadata.component_type;
        let buf = &mut &col.column[..];
        let edges = (0..col.len())
            .map(move |_| {
                let (size, value) = ty.parse_value(buf).unwrap();
                buf.advance(size);
                E::from_value(value).map(|e| e.to_edge())
            })
            .collect::<Option<Vec<_>>>()
            .unwrap();
        GraphQuery {
            edges,
            phantom_data: PhantomData,
        }
    }

    fn insert_into_builder(self, _builder: &mut crate::PipelineBuilder) {}
}

impl SystemParam<PipelineBuilder> for GraphQuery<TotalEdge> {
    type Item = Self;

    fn init(_: &mut crate::PipelineBuilder) -> Result<(), crate::Error> {
        Ok(())
    }

    fn from_builder(builder: &crate::PipelineBuilder) -> Self::Item {
        let edges = (0..builder.world.entity_len)
            .flat_map(|from| {
                (0..builder.world.entity_len)
                    .filter(move |to| from != *to)
                    .map(move |to| Edge::new(from, to))
            })
            .collect::<Vec<_>>();
        GraphQuery {
            edges,
            phantom_data: PhantomData,
        }
    }

    fn insert_into_builder(self, _builder: &mut crate::PipelineBuilder) {}
}

pub fn exprs_from_edges_queries<A, B>(
    edges: &[Edge],
    f_query: Query<A>,
    t_query: Query<B>,
) -> BTreeMap<usize, (Query<A>, Query<B>)> {
    let mut from_map = BTreeMap::new();
    for edge in edges {
        let vec: &mut Vec<EntityId> = from_map.entry(edge.from).or_default();
        vec.push(edge.to);
    }
    let mut exprs: BTreeMap<usize, (Query<A>, Query<B>)> = BTreeMap::new();
    for (from_id, ids) in from_map {
        let from = f_query.filter(&[from_id]);
        if from.len == 0 {
            continue;
        }
        let mut to = t_query.filter(&ids);
        match exprs.entry(ids.len()) {
            std::collections::btree_map::Entry::Occupied(mut o) => {
                let (o_from, o_to) = o.get_mut();
                for (a, b) in o_from.exprs.iter_mut().zip(from.exprs.iter()) {
                    *a = Noxpr::concat_in_dim(vec![a.clone(), b.clone()], 0);
                }
                o_from.len += from.len;
                for (a, b) in o_to.exprs.iter_mut().zip(to.exprs.iter()) {
                    let mut b_shape = b.shape().unwrap();
                    b_shape.insert(0, 1);
                    let b = b.clone().reshape(b_shape);
                    *a = Noxpr::concat_in_dim(vec![a.clone(), b], 0);
                }
                o_from.entity_map.insert(from_id, o_from.entity_map.len());
                o_to.entity_map.extend(to.entity_map);
                o_to.len += to.len;
            }
            std::collections::btree_map::Entry::Vacant(s) => {
                for b in to.exprs.iter_mut() {
                    let mut b_shape = b.shape().unwrap();
                    b_shape.insert(0, 1);

                    *b = b.clone().reshape(b_shape);
                }
                //from.entity_map = std::iter::once((from_id, 0)).collect();
                s.insert((from, to));
            }
        }
    }
    exprs
}

impl<E> GraphQuery<E> {
    /// Folds over each edge of a graph, using the `from` EntityId
    /// to form th resulting `ComponentArray`
    pub fn edge_fold<I: ComponentGroup + IntoOp, F: ComponentGroup, T: ComponentGroup>(
        &self,
        from_query: &Query<F>,
        to_query: &Query<T>,
        init: I,
        func: impl CompFn<(I, (F::Params, T::Params)), I>,
    ) -> ComponentArray<I> {
        let scan_func = func.build_expr().unwrap();
        let init_op = init.into_op();
        let mut output_array: Option<ComponentArray<I>> = None;
        let exprs = exprs_from_edges_queries(&self.edges, from_query.clone(), to_query.clone());
        for (len, (from, to)) in exprs.iter() {
            let axis: Vec<usize> = std::iter::repeat(0)
                .take(from.exprs.len() + to.exprs.len())
                .collect::<Vec<_>>();
            let vmap_args = from
                .exprs
                .iter()
                .cloned()
                .chain(to.exprs.iter().cloned())
                .collect::<Vec<_>>();
            let vmap_fn_args = scan_func.args[1..]
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    let shape = a.shape().unwrap();
                    let element_type = a.element_type().unwrap();
                    Noxpr::parameter(
                        i as i64,
                        NoxprTy::ArrayTy(ArrayTy {
                            element_type,
                            shape: shape.clone(),
                        }),
                        format!("vmap_{}", i),
                    )
                })
                .collect::<Vec<_>>();
            let scan_args = vmap_fn_args[..from.exprs.len()]
                .iter()
                .map(|e| {
                    let mut shape = e.shape().unwrap();
                    shape.insert(0, *len as i64);
                    let broadcast_dims = (1..shape.len()).map(|x| x as i64).collect();
                    e.clone().broadcast_in_dim(shape, broadcast_dims)
                })
                .chain(vmap_fn_args[from.exprs.len()..].iter().cloned())
                .collect::<Vec<_>>();
            let vmap_fn = NoxprFn {
                args: vmap_fn_args.clone(),
                inner: Noxpr::scan(scan_args, init_op.clone(), scan_func.clone()),
            };
            let buffer = Noxpr::vmap_with_axis(vmap_fn, &axis, &vmap_args).unwrap();
            if let Some(output_array) = output_array.as_mut() {
                output_array.buffer =
                    Noxpr::concat_in_dim(vec![output_array.buffer.clone(), buffer], 0);
                for (id, index) in from.entity_map.iter() {
                    output_array
                        .entity_map
                        .insert(*id, index + output_array.len);
                }
                output_array.len += from.len;
            } else {
                output_array = Some(ComponentArray {
                    buffer,
                    entity_map: from.entity_map.clone(),
                    len: from.len,
                    phantom_data: PhantomData,
                });
            }
        }
        output_array.unwrap()
    }
}

#[cfg(test)]
mod tests {

    use nox::{tensor, Matrix, Scalar, Vector};

    use crate::IntoSystemExt;

    use super::*;

    #[test]
    fn test_fold_graph() {
        #[derive(Component)]
        struct A(Scalar<f64>);

        fn fold_system(g: GraphQuery<Edge>, a: Query<A>) -> ComponentArray<A> {
            g.edge_fold(&a, &a, A(5.0.into()), |acc: A, (_, b): (A, A)| {
                A(acc.0 + b.0)
            })
        }

        let mut world = fold_system.world();
        let a = world.spawn(A(10.0.into())).id();
        let b = world.spawn(A(100.0.into())).id();
        let c = world.spawn(A(1000.0.into())).id();
        world.spawn(A(10000.0.into()));
        //world.spawn(Edge { from: c, to: d });
        world.spawn(Edge::new(a, b));
        world.spawn(Edge::new(a, c));
        world.spawn(Edge::new(b, a));
        world.spawn(Edge::new(b, c));

        let world = world.run();
        let c = world.column::<A>().unwrap();
        assert_eq!(
            c.typed_buf::<f64>().unwrap(),
            &[1105.0, 1015.0, 1000.0, 10000.0],
        );
    }

    #[test]
    fn test_single_graph() {
        #[derive(Component)]
        struct A(Scalar<f64>);

        fn fold_system(query: GraphQuery<Edge>, a: Query<A>) -> ComponentArray<A> {
            query.edge_fold(&a, &a, A(5.0.into()), |acc: A, (_, b): (A, A)| {
                A(acc.0 + b.0)
            })
        }

        let mut world = fold_system.world();
        let a = world.spawn(A(10.0.into())).id();
        let b = world.spawn(A(100.0.into())).id();
        world.spawn(A(1000.0.into()));
        world.spawn(A(10000.0.into()));
        world.spawn(Edge::new(a, b));

        let world = world.run();
        let c = world.column::<A>().unwrap();
        assert_eq!(
            c.typed_buf::<f64>().unwrap(),
            &[105.0, 100.0, 1000.0, 10000.0],
        );
    }

    #[test]
    fn test_complex_graph_compile() {
        #[derive(Component)]
        struct A(Scalar<f64>);

        fn fold_system(query: GraphQuery<Edge>, a: Query<A>) -> ComponentArray<A> {
            query.edge_fold(&a, &a, A(5.0.into()), |acc: A, (_, b): (A, A)| {
                A(acc.0 + b.0)
            })
        }
        // NOTE: this loops 5000 times because that was the only way to repro
        // a very specific failure case caused by an old impl of `NoxprId`
        for _ in 0..5000 {
            let mut world = fold_system.world();
            let c = world.spawn(A(1000.0.into())).id();
            let a = world.spawn(A(10.0.into())).id();
            let b = world.spawn(A(100.0.into())).id();
            let d = world.spawn(A(10000.0.into())).id();
            world.spawn(Edge::new(c, d));
            world.spawn(Edge::new(a, b));
            world.spawn(Edge::new(a, c));
            world.spawn(Edge::new(b, a));
            world.spawn(Edge::new(b, c));

            world.build().unwrap();
        }
    }

    #[test]
    fn test_total_fold_graph() {
        #[derive(Component)]
        struct A(Scalar<f64>);

        fn fold_system(g: GraphQuery<TotalEdge>, a: Query<A>) -> ComponentArray<A> {
            g.edge_fold(&a, &a, A(5.0.into()), |acc: A, (_, b): (A, A)| {
                A(acc.0 + b.0)
            })
        }

        let mut world = fold_system.world();
        world.spawn(A(10.0.into())).id();
        world.spawn(A(100.0.into())).id();
        world.spawn(A(1000.0.into())).id();

        let world = world.run();
        let c = world.column::<A>().unwrap();
        assert_eq!(c.typed_buf::<f64>().unwrap(), &[1105.0, 1015.0, 115.0,],);
    }

    #[test]
    fn test_total_fold_vector_graph() {
        #[derive(Component)]
        struct A(Vector<f64, 2>);

        fn fold_system(g: GraphQuery<TotalEdge>, a: Query<A>) -> ComponentArray<A> {
            g.edge_fold(
                &a,
                &a,
                A(tensor![5.0, 0.0,].into()),
                |acc: A, (_, b): (A, A)| A(acc.0 + b.0),
            )
        }

        let mut world = fold_system.world();
        world.spawn(A(tensor![10.0, 0.0].into())).id();
        world.spawn(A(tensor![100.0, 0.0].into())).id();
        world.spawn(A(tensor![1000.0, 0.0].into())).id();
        world.spawn(A(tensor![10000.0, 0.0].into())).id();
        world.spawn(A(tensor![100000.0, 0.0].into())).id();

        let world = world.run();
        let c = world.column::<A>().unwrap();
        assert_eq!(
            c.typed_buf::<f64>().unwrap(),
            &[111105.0, 0.0, 111015.0, 0.0, 110115.0, 0.0, 101115.0, 0.0, 11115.0, 0.0],
        );
    }

    #[test]
    fn test_total_fold_matrix_graph() {
        #[derive(Component)]
        struct A(Matrix<f64, 2, 2>);

        fn fold_system(g: GraphQuery<TotalEdge>, a: Query<A>) -> ComponentArray<A> {
            g.edge_fold(
                &a,
                &a,
                A(tensor![[5.0, 0.0,], [0.0, 0.0]].into()),
                |acc: A, (_, b): (A, A)| A(acc.0 + b.0),
            )
        }

        let mut world = fold_system.world();
        world.spawn(A(tensor![[10.0, 0.0], [0.0, 0.0]].into())).id();
        world.spawn(A(tensor![[100.0, 0.0], [0., 0.]].into())).id();
        world.spawn(A(tensor![[1000.0, 0.0], [0., 0.]].into())).id();
        world
            .spawn(A(tensor![[10000.0, 0.0], [0., 0.]].into()))
            .id();
        world
            .spawn(A(tensor![[100000.0, 0.0], [0., 0.]].into()))
            .id();

        let world = world.run();
        let c = world.column::<A>().unwrap();
        assert_eq!(
            c.typed_buf::<f64>().unwrap(),
            &[
                111105.0, 0.0, 0.0, 0.0, 111015.0, 0.0, 0.0, 0.0, 110115.0, 0.0, 0.0, 0.0,
                101115.0, 0.0, 0.0, 0.0, 11115.0, 0.0, 0.0, 0.0,
            ],
        );
    }
}
