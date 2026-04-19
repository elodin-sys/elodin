use impeller2::{
    component::Component,
    types::{ComponentId, ComponentView, EntityId},
};

use bytes::Buf;
use nox::{
    ArrayTy, Builder, CompFn, Const, Literal, Noxpr, NoxprFn, NoxprTy, Op, ReprMonad,
    array::ArrayViewExt,
};
use std::{collections::BTreeMap, marker::PhantomData};

use crate::query::normalize_expr;
use crate::{ComponentArray, ComponentGroup, Query};

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
        unimplemented!()
    }

    fn into_inner(self) -> Noxpr {
        Noxpr::constant(
            Literal::vector(&[self.from.0, self.to.0]),
            ArrayTy {
                element_type: nox::ElementType::U64,
                shape: smallvec::smallvec![2],
            },
        )
    }

    fn inner(&self) -> &Noxpr {
        unimplemented!()
    }

    fn from_inner(_inner: Noxpr) -> Self {
        unimplemented!()
    }
}

impl crate::component::Component for Edge {}

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
        let from = EntityId(val.get(0));
        let to = EntityId(val.get(1));
        Some(Edge { from, to })
    }
}

pub struct TotalEdge;

impl<E: EdgeComponent + 'static> crate::system::SystemParam for GraphQuery<E> {
    type Item = Self;

    fn init(_builder: &mut crate::system::SystemBuilder) -> Result<(), crate::Error> {
        Ok(())
    }

    fn param(builder: &crate::system::SystemBuilder) -> Result<Self::Item, crate::Error> {
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

    fn output(&self, _builder: &mut crate::system::SystemBuilder) -> Result<Noxpr, crate::Error> {
        unimplemented!()
    }
}

impl crate::system::SystemParam for GraphQuery<TotalEdge> {
    type Item = Self;

    fn init(_builder: &mut crate::system::SystemBuilder) -> Result<(), crate::Error> {
        Ok(())
    }

    fn param(builder: &crate::system::SystemBuilder) -> Result<Self::Item, crate::Error> {
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

    fn output(&self, _builder: &mut crate::system::SystemBuilder) -> Result<Noxpr, crate::Error> {
        unimplemented!()
    }
}

pub fn exprs_from_edges_queries<A, B>(
    edges: &[Edge],
    f_query: Query<A>,
    t_query: Query<B>,
) -> BTreeMap<usize, (Query<A>, Query<B>)> {
    fn add_source_axis(buffer: &Noxpr) -> Noxpr {
        let mut shape = buffer.shape().unwrap();
        shape.insert(0, 1);
        buffer.clone().reshape(shape)
    }

    fn source_batched_expr(buffer: &Noxpr, batch1: bool) -> Noxpr {
        add_source_axis(&normalize_expr(buffer, batch1, false))
    }

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
        let to = t_query.filter(&ids);
        match exprs.entry(to.len) {
            std::collections::btree_map::Entry::Occupied(mut o) => {
                let (o_from, o_to) = o.get_mut();
                if o_from.batch1 {
                    for a in &mut o_from.exprs {
                        *a = normalize_expr(a, true, false);
                    }
                    for a in &mut o_to.exprs {
                        *a = source_batched_expr(a, o_to.batch1);
                    }
                    o_from.batch1 = false;
                    o_to.batch1 = false;
                }
                for (a, b) in o_from.exprs.iter_mut().zip(from.exprs.iter()) {
                    let b = normalize_expr(b, from.batch1, false);
                    *a = Noxpr::concat_in_dim(vec![a.clone(), b], 0);
                }
                o_from.len += from.len;
                for (a, b) in o_to.exprs.iter_mut().zip(to.exprs.iter()) {
                    *a =
                        Noxpr::concat_in_dim(vec![a.clone(), source_batched_expr(b, to.batch1)], 0);
                }
                o_from.entity_map.insert(from_id, o_from.entity_map.len());
                o_to.entity_map.extend(to.entity_map);
                o_to.len += to.len;
            }
            std::collections::btree_map::Entry::Vacant(s) => {
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
            let buffer = if from.batch1 {
                if *len == 1 && to.batch1 {
                    let args = std::iter::once(init_op.clone())
                        .chain(from.exprs.iter().cloned())
                        .chain(to.exprs.iter().cloned())
                        .collect::<Vec<_>>();
                    Noxpr::substitute_params(&scan_func, &args)
                } else {
                    let scan_args = from
                        .exprs
                        .iter()
                        .map(|e| {
                            let mut shape = e.shape().unwrap();
                            shape.insert(0, *len as i64);
                            let broadcast_dims = (1..shape.len()).map(|x| x as i64).collect();
                            e.clone().broadcast_in_dim(shape, broadcast_dims)
                        })
                        .chain(to.exprs.iter().map(|e| normalize_expr(e, to.batch1, false)))
                        .collect::<Vec<_>>();
                    Noxpr::scan(scan_args, init_op.clone(), scan_func.clone())
                }
            } else {
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
                Noxpr::vmap_with_axis(vmap_fn, &axis, &vmap_args).unwrap()
            };
            if let Some(existing) = output_array.take() {
                let mut entity_map = existing.entity_map.clone();
                for (id, index) in from.entity_map.iter() {
                    entity_map.insert(*id, index + existing.len);
                }
                let buffer = Noxpr::concat_in_dim(
                    vec![
                        normalize_expr(&existing.buffer, existing.batch1, false),
                        normalize_expr(&buffer, from.batch1, false),
                    ],
                    0,
                );
                output_array = Some(ComponentArray::new(
                    buffer,
                    existing.len + from.len,
                    entity_map,
                    existing.component_id,
                    false,
                ));
            } else {
                output_array = Some(ComponentArray::new(
                    buffer,
                    from.len,
                    from.entity_map.clone(),
                    ComponentId::new("foo"), // TODO(sphw): fix me
                    from.batch1,
                ));
            }
        }
        output_array.unwrap()
    }
}

// --- Python wrappers ---

use crate::{Error, PyComponent, QueryInner};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[pyclass]
#[derive(Clone)]
pub struct GraphQueryInner {
    query: GraphQuery<()>,
}

#[pymethods]
impl GraphQueryInner {
    #[staticmethod]
    fn from_builder_total_edge(
        builder: &mut crate::system::PySystemBuilder,
    ) -> Result<GraphQueryInner, Error> {
        Ok(GraphQueryInner {
            query: GraphQuery {
                edges: builder.total_edges.clone(),
                phantom_data: PhantomData,
            },
        })
    }

    #[staticmethod]
    fn from_builder(
        builder: &mut crate::system::PySystemBuilder,
        edge_name: String,
        reverse: bool,
    ) -> Result<GraphQueryInner, Error> {
        let edge_id = ComponentId::new(&edge_name);
        let edges = builder
            .edge_map
            .get(&edge_id)
            .ok_or(crate::Error::ComponentNotFound)?;
        let edges = edges
            .iter()
            .map(|x| if reverse { x.reverse() } else { x.clone() })
            .collect();
        Ok(GraphQueryInner {
            query: GraphQuery {
                edges,
                phantom_data: PhantomData,
            },
        })
    }

    fn arrays(
        &self,
        py: Python<'_>,
        from_query: QueryInner,
        to_query: QueryInner,
    ) -> Result<PyObject, Error> {
        let dict = PyDict::new(py);
        let exprs = exprs_from_edges_queries(&self.query.edges, from_query.query, to_query.query);
        for (len, (a, b)) in exprs.iter() {
            let a_list = PyList::empty(py);
            let b_list = PyList::empty(py);
            for x in &a.exprs {
                a_list.append(x.to_jax()?)?;
            }
            for x in &b.exprs {
                b_list.append(x.to_jax()?)?;
            }

            dict.set_item(len, (a_list, b_list))?;
        }
        Ok(dict.into())
    }

    fn map(
        &self,
        from_query: QueryInner,
        to_query: QueryInner,
        new_buf: PyObject,
        metadata: PyComponent,
    ) -> QueryInner {
        let mut entity_map = BTreeMap::new();
        let mut len = 0;
        let exprs = exprs_from_edges_queries(&self.query.edges, from_query.query, to_query.query);
        for (_, (from, _to)) in exprs.iter() {
            for (id, index) in from.entity_map.iter() {
                entity_map.insert(*id, index + len);
            }
            len += from.len;
        }
        let expr = Noxpr::jax(new_buf);
        QueryInner {
            query: Query::new(vec![expr], entity_map, len, len == 1),
            metadata: vec![metadata],
        }
    }
}

#[pyclass(name = "Edge")]
pub struct PyEdge {
    inner: Edge,
}

#[pymethods]
impl PyEdge {
    #[new]
    pub fn new(from: crate::entity::EntityId, to: crate::entity::EntityId) -> Self {
        Self {
            inner: Edge {
                from: from.inner,
                to: to.inner,
            },
        }
    }

    fn flatten(&self) -> Result<((PyObject,), Option<()>), Error> {
        let jax = self.inner.clone().into_inner().to_jax()?;
        Ok(((jax,), None))
    }

    #[staticmethod]
    fn unflatten(_aux: PyObject, _jax: PyObject) -> Self {
        todo!()
    }

    #[classattr]
    fn metadata() -> PyComponent {
        PyComponent::from_component::<Edge>()
    }

    #[classattr]
    fn __metadata__() -> (PyComponent,) {
        (Self::metadata(),)
    }

    #[staticmethod]
    fn from_array(_arr: PyObject) -> Self {
        todo!()
    }
}

#[cfg(test)]
mod batch1_graph_tests {
    use super::*;
    use nox::{ElementType, NoxprTy};

    fn two_entity_map() -> BTreeMap<EntityId, usize> {
        BTreeMap::from([(EntityId(1), 0), (EntityId(2), 1)])
    }

    fn batched_query(number: i64, name: &str) -> Query<()> {
        Query {
            exprs: vec![Noxpr::parameter(
                number,
                NoxprTy::ArrayTy(ArrayTy {
                    element_type: ElementType::F64,
                    shape: smallvec::smallvec![2, 3],
                }),
                name.to_owned(),
            )],
            entity_map: two_entity_map(),
            len: 2,
            phantom_data: PhantomData,
            batch1: false,
        }
    }

    #[test]
    fn singleton_edge_groups_recover_local_singletons() {
        let groups = exprs_from_edges_queries(
            &[Edge::new(EntityId(1), EntityId(2))],
            batched_query(0, "from"),
            batched_query(1, "to"),
        );

        let (from, to) = groups.get(&1).expect("singleton edge group");
        assert_eq!(from.len, 1);
        assert_eq!(to.len, 1);
        assert!(from.batch1);
        assert!(to.batch1);
        assert_eq!(from.exprs[0].shape().unwrap().as_slice(), &[3]);
        assert_eq!(to.exprs[0].shape().unwrap().as_slice(), &[3]);
    }

    #[test]
    fn multi_source_single_edge_groups_rebatch_on_append() {
        let groups = exprs_from_edges_queries(
            &[
                Edge::new(EntityId(1), EntityId(2)),
                Edge::new(EntityId(2), EntityId(1)),
            ],
            batched_query(0, "from"),
            batched_query(1, "to"),
        );

        let (from, to) = groups.get(&1).expect("two-source edge group");
        assert_eq!(from.len, 2);
        assert_eq!(to.len, 2);
        assert!(!from.batch1);
        assert!(!to.batch1);
        assert_eq!(from.exprs[0].shape().unwrap().as_slice(), &[2, 3]);
        assert_eq!(to.exprs[0].shape().unwrap().as_slice(), &[2, 1, 3]);
    }
}
