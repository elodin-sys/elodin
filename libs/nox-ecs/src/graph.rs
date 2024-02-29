use bytes::Buf;
use conduit::{ComponentId, ComponentType, ComponentValue, EntityId};
use nox::{xla::Literal, ArrayTy, CompFn, FromBuilder, IntoOp, Noxpr, NoxprFn, NoxprTy};
use std::{collections::BTreeMap, marker::PhantomData};

use crate::{Component, ComponentArray, ComponentGroup, Query, SystemParam};

pub struct GraphQuery<E: EdgeComponent, G: ComponentGroup> {
    pub edges: Vec<Edge>,
    exprs: BTreeMap<usize, (Query<G>, Query<G>)>,
    phantom_data: PhantomData<(E, G)>,
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub from: EntityId,
    pub to: EntityId,
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

impl Component for Edge {
    type Inner = Self;

    type HostTy = Self;

    fn host(val: Self::HostTy) -> Self {
        val
    }

    fn component_id() -> conduit::ComponentId {
        ComponentId::new("edge")
    }

    fn component_type() -> conduit::ComponentType {
        ComponentType {
            primitive_ty: conduit::PrimitiveTy::U64,
            shape: smallvec::smallvec![2],
        }
    }
}

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

impl<E: EdgeComponent + 'static, G: ComponentGroup> SystemParam for GraphQuery<E, G> {
    type Item = Self;

    fn init(builder: &mut crate::PipelineBuilder) -> Result<(), crate::Error> {
        Query::<G>::init(builder)
    }

    fn from_builder(builder: &crate::PipelineBuilder) -> Self::Item {
        let col = builder.world.column::<E>().unwrap();
        let ty = &col.column.buffer.component_type;
        let buf = &mut &col.column.buffer.buf[..];
        let len = col.column.buffer.len;
        let edges = (0..len)
            .map(move |_| {
                let (size, value) = ty.parse_value(buf).unwrap();
                buf.advance(size);
                dbg!(E::from_value(value).map(|e| e.to_edge()))
            })
            .collect::<Option<Vec<_>>>()
            .unwrap();
        let mut from_map = BTreeMap::new();
        for edge in &edges {
            let vec: &mut Vec<EntityId> = from_map.entry(edge.from).or_default();
            vec.push(edge.to);
        }
        let g_query = Query::<G>::from_builder(builder);
        let mut exprs: BTreeMap<usize, (Query<G>, Query<G>)> = BTreeMap::new();
        for (from_id, ids) in from_map {
            let from = g_query.filter(&[from_id]);
            let mut to = g_query.filter(&ids);
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
        GraphQuery {
            edges,
            phantom_data: PhantomData,
            exprs,
        }
    }

    fn insert_into_builder(self, _builder: &mut crate::PipelineBuilder) {}
}

impl<E: EdgeComponent, G: ComponentGroup> GraphQuery<E, G> {
    /// Folds over each edge of a graph, using the `from` EntityId
    /// to form th resulting `ComponentArray`
    pub fn edge_fold<I: ComponentGroup + IntoOp>(
        &self,
        init: I,
        func: impl CompFn<(I, (G::Params, G::Params)), I>,
    ) -> ComponentArray<I> {
        let scan_func = func.build_expr().unwrap();
        let init_op = init.into_op();
        let mut output_array: Option<ComponentArray<I>> = None;
        for (len, (from, to)) in self.exprs.iter() {
            let axis = std::iter::repeat(0)
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
                    shape.push(*len as i64);
                    e.clone().broadcast(shape)
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

    use nox::{Scalar, ScalarExt};

    use crate::IntoSystem;

    use super::*;

    #[test]
    fn test_fold_graph() {
        #[derive(Component)]
        struct A(Scalar<f64>);

        fn fold_system(a: GraphQuery<Edge, A>) -> ComponentArray<A> {
            a.edge_fold(A(5.0.constant()), |acc: A, (_, b): (A, A)| A(acc.0 + b.0))
        }

        let mut world = fold_system.world();
        let a = world.spawn(A::host(10.0));
        let b = world.spawn(A::host(100.0));
        let c = world.spawn(A::host(1000.0));
        world.spawn(A::host(10000.0));
        //world.spawn(Edge { from: c, to: d });
        world.spawn(Edge { from: a, to: b });
        world.spawn(Edge { from: a, to: c });
        world.spawn(Edge { from: b, to: a });
        world.spawn(Edge { from: b, to: c });

        let client = nox::Client::cpu().unwrap();
        let mut exec = world.build().unwrap();
        exec.run(&client).unwrap();
        let c = exec.column(A::component_id()).unwrap();
        assert_eq!(
            c.typed_buf::<f64>().unwrap(),
            &[1105.0, 1015.0, 1000.0, 10000.0],
        );
    }

    #[test]
    fn test_single_graph() {
        #[derive(Component)]
        struct A(Scalar<f64>);

        fn fold_system(a: GraphQuery<Edge, A>) -> ComponentArray<A> {
            a.edge_fold(A(5.0.constant()), |acc: A, (_, b): (A, A)| A(acc.0 + b.0))
        }

        let mut world = fold_system.world();
        let a = world.spawn(A::host(10.0));
        let b = world.spawn(A::host(100.0));
        world.spawn(A::host(1000.0));
        world.spawn(A::host(10000.0));
        world.spawn(Edge { from: a, to: b });

        let client = nox::Client::cpu().unwrap();
        let mut exec = world.build().unwrap();
        exec.run(&client).unwrap();
        let c = exec.column(A::component_id()).unwrap();
        assert_eq!(
            c.typed_buf::<f64>().unwrap(),
            &[105.0, 100.0, 1000.0, 10000.0],
        );
    }

    // FIXME(sphw): this example crashes some of the time, it is not clear
    // why at the moment. It is seemingly random, so likely some unknown source of
    // non determinism is to blame
    // #[test]
    // fn test_fold_complex_graph() {
    //     #[derive(Component)]
    //     struct A(Scalar<f64>);

    //     fn fold_system(a: GraphQuery<Edge, A>) -> ComponentArray<A> {
    //         a.edge_fold(A(5.0.constant()), |acc: A, (_, b): (A, A)| A(acc.0 + b.0))
    //     }

    //     let mut world = fold_system.world();
    //     let c = world.spawn(A::host(1000.0));
    //     let a = world.spawn(A::host(10.0));
    //     println!("a {:?}", a);
    //     let b = world.spawn(A::host(100.0));
    //     println!("b {:?}", b);
    //     let d = world.spawn(A::host(10000.0));
    //     world.spawn(Edge { from: c, to: d });
    //     world.spawn(Edge { from: a, to: b });
    //     world.spawn(Edge { from: a, to: c });
    //     world.spawn(Edge { from: b, to: a });
    //     world.spawn(Edge { from: b, to: c });

    //     let client = nox::Client::cpu().unwrap();
    //     let mut exec = world.build().unwrap();
    //     exec.run(&client).unwrap();
    //     let c = exec.column(A::component_id()).unwrap();
    //     assert_eq!(
    //         c.typed_buf::<f64>().unwrap(),
    //         &[10005.0, 1105.0, 1015.0, 10000.0] //&[1105., 1015., 10005., 10000.0]
    //     );
    // }
}
