use impeller2::{
    component::Component,
    types::{ComponentId, ComponentView, EntityId},
};

use bytes::Buf;
use nox::{ArrayTy, Builder, CompFn, Const, Noxpr, NoxprFn, NoxprTy, Op, ReprMonad, xla::Literal};
use std::{collections::BTreeMap, marker::PhantomData};

use crate::ecs::{Query, component_array::ComponentArray, query::ComponentGroup};

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

impl ReprMonad<Op> for Edge {
    type Elem = u64;

    type Dim = Const<2>;

    type Map<T: nox::OwnedRepr> = Self;

    fn map<N: nox::OwnedRepr>(
        self,
        _func: impl Fn(Noxpr) -> N::Inner<Self::Elem, Self::Dim>,
    ) -> Self::Map<N> {
        // Edge doesn't transform, just return self
        self
    }

    fn into_inner(self) -> Noxpr {
        Noxpr::constant(
            Literal::vector(&[self.from.0, self.to.0]),
            ArrayTy {
                element_type: nox::xla::ElementType::U64,
                shape: smallvec::smallvec![2],
            },
        )
    }

    fn inner(&self) -> &Noxpr {
        // Edge computes its Noxpr on demand, cannot return a reference
        panic!("Edge::inner() is not supported - use into_inner() instead")
    }

    fn from_inner(_inner: Noxpr) -> Self {
        // Edge is constructed from entity IDs, not from Noxpr
        panic!("Edge::from_inner() is not supported - use Edge::new() instead")
    }
}

impl crate::ecs::Component for Edge {}

impl Component for Edge {
    const NAME: &'static str = "edge";

    fn schema() -> impeller2::schema::Schema<Vec<u64>> {
        impeller2::schema::Schema::new(impeller2::types::PrimType::U64, [2usize]).unwrap()
    }
}

pub trait EdgeComponent: Component {
    fn to_edge(&self) -> Edge;
    fn from_value(value: ComponentView<'_>) -> Option<Self>
    where
        Self: Sized;
}

impl EdgeComponent for Edge {
    fn to_edge(&self) -> Edge {
        self.clone()
    }

    fn from_value(value: ComponentView<'_>) -> Option<Self>
    where
        Self: Sized,
    {
        let ComponentView::U64(val) = value else {
            return None;
        };
        if val.buf.len() < 2 {
            return None;
        }
        let from = EntityId(val.buf[0]);
        let to = EntityId(val.buf[1]);
        Some(Edge { from, to })
    }
}

pub struct TotalEdge;

impl<E: EdgeComponent + crate::ecs::Component + 'static> crate::ecs::system::SystemParam
    for GraphQuery<E>
{
    type Item = Self;

    fn init(_builder: &mut crate::ecs::system::SystemBuilder) -> Result<(), crate::Error> {
        Ok(())
    }

    fn param(builder: &crate::ecs::system::SystemBuilder) -> Result<Self::Item, crate::Error> {
        let col = builder.world.column::<E>().unwrap();
        let ty = &col.schema;
        let buf = &mut &col.column[..];
        let edges = (0..col.len())
            .map(move |_| {
                let (size, value) = ty.parse_value(buf).unwrap();
                buf.advance(size);
                E::from_value(value).map(|e| e.to_edge())
            })
            .collect::<Option<Vec<_>>>()
            .unwrap();
        Ok(GraphQuery {
            edges,
            phantom_data: PhantomData,
        })
    }

    fn component_ids() -> impl Iterator<Item = ComponentId> {
        std::iter::empty()
    }

    fn output(
        &self,
        _builder: &mut crate::ecs::system::SystemBuilder,
    ) -> Result<Noxpr, crate::Error> {
        unimplemented!()
    }
}

impl crate::ecs::system::SystemParam for GraphQuery<TotalEdge> {
    type Item = Self;

    fn init(_builder: &mut crate::ecs::system::SystemBuilder) -> Result<(), crate::Error> {
        Ok(())
    }

    fn param(builder: &crate::ecs::system::SystemBuilder) -> Result<Self::Item, crate::Error> {
        let edges = (0..builder.world.entity_len())
            .flat_map(|from| {
                (0..builder.world.entity_len())
                    .filter(move |to| from != *to)
                    .map(move |to| Edge::new(from, to))
            })
            .collect::<Vec<_>>();
        Ok(GraphQuery {
            edges,
            phantom_data: PhantomData,
        })
    }

    fn component_ids() -> impl Iterator<Item = ComponentId> {
        std::iter::empty()
    }

    fn output(
        &self,
        _builder: &mut crate::ecs::system::SystemBuilder,
    ) -> Result<Noxpr, crate::Error> {
        unimplemented!()
    }
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
        match exprs.entry(to.len) {
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
                s.insert((from, to));
            }
        }
    }
    exprs
}

impl<E> GraphQuery<E> {
    pub fn edge_fold<I: ComponentGroup, F: ComponentGroup, T: ComponentGroup>(
        &self,
        from_query: &Query<F>,
        to_query: &Query<T>,
        init: I,
        func: impl CompFn<(I, (F::Params, T::Params)), I>,
    ) -> ComponentArray<I> {
        let mut builder = Builder::new();
        let res = func.compute(&mut builder);
        let inner = if !builder.mut_params.is_empty() {
            let mut tuple = Vec::with_capacity(builder.mut_params.count() + 1);
            let res_op = res.into_noxpr();
            tuple.push(res_op);
            for o in builder.mut_params.into_iter() {
                tuple.insert(1, o.into_inner().into_inner());
            }
            Noxpr::tuple(tuple)
        } else {
            res.into_noxpr()
        };
        let scan_func = NoxprFn {
            inner,
            args: builder.params.into_inner(),
        };

        let init_op = init.into_noxpr();
        let mut output_array: Option<ComponentArray<I>> = None;
        let exprs = exprs_from_edges_queries(&self.edges, from_query.clone(), to_query.clone());
        for (len, (from, to)) in exprs.iter() {
            let axis: Vec<usize> =
                std::iter::repeat_n(0, from.exprs.len() + to.exprs.len()).collect::<Vec<_>>();
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
                    component_id: I::component_ids()
                        .next()
                        .expect("ComponentGroup must have at least one component"),
                });
            }
        }
        output_array.unwrap()
    }
}
