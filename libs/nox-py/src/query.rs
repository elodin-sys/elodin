use crate::component::Component;
use crate::error::Error;
use crate::system::{SystemBuilder, SystemParam};
use crate::utils::SchemaExt;
use elodin_db::ComponentSchema;
use impeller2::types::{ComponentId, EntityId};
use nox::{
    ArrayTy, Builder, CompFn, ElementType, Literal, Noxpr, NoxprFn, NoxprScalarExt, ReprMonad,
};
use smallvec::{SmallVec, smallvec};
use std::iter::once;
use std::{collections::BTreeMap, marker::PhantomData};

// --- ComponentArray from mod.rs ---

pub struct ComponentArray<T> {
    pub buffer: Noxpr,
    pub len: usize,
    pub entity_map: BTreeMap<EntityId, usize>,
    pub phantom_data: PhantomData<T>,
    pub component_id: ComponentId,
}

impl<T> Clone for ComponentArray<T> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            len: self.len,
            entity_map: self.entity_map.clone(),
            phantom_data: PhantomData,
            component_id: self.component_id,
        }
    }
}

impl<T> ComponentArray<T> {
    pub(crate) fn cast<D>(self) -> ComponentArray<D> {
        ComponentArray {
            buffer: self.buffer,
            phantom_data: PhantomData,
            entity_map: self.entity_map,
            len: self.len,
            component_id: self.component_id,
        }
    }

    pub fn buffer(&self) -> &Noxpr {
        &self.buffer
    }
}

impl<T: Component> ComponentArray<T> {
    pub fn get(&self, offset: i64) -> T {
        let ty: ArrayTy = T::schema().to_array_ty();
        let shape = ty.shape;

        let start_indices: SmallVec<_> = once(offset).chain(shape.iter().map(|_| 0)).collect();
        let stop_indices: SmallVec<_> = once(offset + 1).chain(shape.clone()).collect();
        let strides: SmallVec<_> = smallvec![1; shape.len() + 1];

        let op = self
            .buffer
            .clone()
            .slice(start_indices, stop_indices, strides)
            .reshape(shape);
        T::from_inner(op)
    }
}

impl<T: Component + 'static> SystemParam for ComponentArray<T> {
    type Item = Self;

    fn init(builder: &mut SystemBuilder) -> Result<(), Error> {
        let id = T::COMPONENT_ID;
        builder.init_with_column(id)?;
        Ok(())
    }

    fn param(builder: &SystemBuilder) -> Result<Self::Item, Error> {
        let id = T::COMPONENT_ID;
        if let Some(var) = builder.vars.get(&id) {
            Ok(var.clone().cast())
        } else {
            Err(Error::ComponentNotFound)
        }
    }

    fn component_ids() -> impl Iterator<Item = ComponentId> {
        std::iter::once(T::COMPONENT_ID)
    }

    fn output(&self, builder: &mut SystemBuilder) -> Result<Noxpr, Error> {
        if let Some(var) = builder.vars.get_mut(&T::COMPONENT_ID)
            && var.entity_map != self.entity_map
        {
            return Ok(update_var(
                &var.entity_map,
                &self.entity_map,
                &var.buffer,
                &self.buffer,
            ));
        }
        Ok(self.buffer.clone())
    }
}

// --- ComponentArray helpers from mod.rs ---

pub fn update_var(
    old_entity_map: &BTreeMap<EntityId, usize>,
    update_entity_map: &BTreeMap<EntityId, usize>,
    old_buffer: &Noxpr,
    update_buffer: &Noxpr,
) -> Noxpr {
    let (old, new, _) = intersect_ids(old_entity_map, update_entity_map);
    let shape = update_buffer.shape().unwrap();
    old.iter().zip(new.iter()).fold(
        old_buffer.clone(),
        |buffer, (existing_index, update_index)| {
            let mut start = shape.clone();
            start[0] = *update_index as i64;
            for x in start.iter_mut().skip(1) {
                *x = 0;
            }
            let mut stop = shape.clone();
            stop[0] = *update_index as i64 + 1;
            let start = std::iter::once(*update_index as i64)
                .chain(std::iter::repeat_n(0, shape.len() - 1))
                .collect();
            let existing_index = std::iter::once((*existing_index as i64).constant())
                .chain(std::iter::repeat_n(0i64.constant(), shape.len() - 1))
                .collect();
            buffer.dynamic_update_slice(
                existing_index,
                update_buffer
                    .clone()
                    .slice(start, stop, shape.iter().map(|_| 1).collect()),
            )
        },
    )
}

pub fn intersect_ids(
    a: &BTreeMap<EntityId, usize>,
    b: &BTreeMap<EntityId, usize>,
) -> (Vec<u32>, Vec<u32>, BTreeMap<EntityId, usize>) {
    fn intersect_inner(
        small: &BTreeMap<EntityId, usize>,
        large: &BTreeMap<EntityId, usize>,
    ) -> (Vec<u32>, Vec<u32>, BTreeMap<EntityId, usize>) {
        let mut small_indices = Vec::with_capacity(small.len());
        let mut large_indices = Vec::with_capacity(small.len());
        let mut ids = BTreeMap::new();
        let mut k = 0;
        for (id, i) in small.iter() {
            if let Some(j) = large.get(id) {
                small_indices.push(*i as u32);
                large_indices.push(*j as u32);
                ids.insert(*id, k);
                k += 1;
            }
        }
        (small_indices, large_indices, ids)
    }
    match a.len().cmp(&b.len()) {
        std::cmp::Ordering::Equal | std::cmp::Ordering::Less => intersect_inner(a, b),
        std::cmp::Ordering::Greater => {
            let (b, a, ids) = intersect_inner(b, a);
            (a, b, ids)
        }
    }
}

// --- Query ---

pub struct Query<Param> {
    pub exprs: Vec<Noxpr>,
    pub entity_map: BTreeMap<EntityId, usize>,
    pub len: usize,
    pub phantom_data: PhantomData<Param>,
}

impl<Param> std::fmt::Debug for Query<Param> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Query")
            .field("exprs", &self.exprs)
            .field("entity_map", &self.entity_map)
            .field("len", &self.len)
            .field("phantom_data", &self.phantom_data)
            .finish()
    }
}

impl<Param> Clone for Query<Param> {
    fn clone(&self) -> Self {
        Self {
            exprs: self.exprs.clone(),
            entity_map: self.entity_map.clone(),
            len: self.len,
            phantom_data: PhantomData,
        }
    }
}

impl<Param> Query<Param> {
    #[inline(always)]
    pub(crate) fn transmute<B>(self) -> Query<B> {
        Query {
            exprs: self.exprs,
            entity_map: self.entity_map,
            len: self.len,
            phantom_data: PhantomData,
        }
    }
}

// --- ComponentGroup ---

pub trait ComponentGroup {
    type Params;
    type Append<O>;

    fn init(builder: &mut SystemBuilder) -> Result<(), Error>;

    fn component_arrays<'a>(
        builder: &'a SystemBuilder,
    ) -> impl Iterator<Item = ComponentArray<()>> + 'a;

    fn component_types() -> impl Iterator<Item = ComponentSchema>;
    fn component_ids() -> impl Iterator<Item = ComponentId>;
    fn component_count() -> usize;

    fn map_axes() -> &'static [usize];

    fn into_noxpr(self) -> Noxpr;
}

macro_rules! impl_group {
    ($num:tt; $($param:tt),*) => {
      impl<$($param),*> ComponentGroup for ($($param,)*)
            where $($param: ComponentGroup),*
        {
            type Params = Self;
            type Append<O> = ($($param,)* O);

            fn init(builder: &mut SystemBuilder) -> Result<(), Error> {
                $(
                    <$param>::init(builder)?;
                )*
                Ok(())

            }

            fn component_arrays<'a>(
                builder: &'a SystemBuilder,
            ) -> impl Iterator<Item = ComponentArray<()>> + 'a {
                let iter = std::iter::empty();
                $(
                    let iter = iter.chain($param::component_arrays(builder));
                )*
                iter
            }


            fn component_types() -> impl Iterator<Item = ComponentSchema> {
                let iter = std::iter::empty();
                $(
                    let iter = iter.chain($param::component_types());
                )*
                iter
            }


            fn component_count() -> usize {
                0 $(
                    + <$param>::component_count()
                )*
            }

            fn component_ids() -> impl Iterator<Item = ComponentId> {
                    let iter = std::iter::empty();
                    $(
                        let iter = iter.chain($param::component_ids());
                    )*
                    iter
            }

            fn map_axes() -> &'static [usize] {
                &[0; $num]
            }

            #[allow(non_snake_case)]
            fn into_noxpr(self) -> Noxpr {
                let ($($param,)*) = self;
                Noxpr::tuple(vec![
                    $($param.into_noxpr(),)*
                ])
            }
        }
    }
}

impl<T> ComponentGroup for T
where
    T: Component,
    ComponentArray<T>: SystemParam<Item = ComponentArray<T>>,
{
    type Params = T;

    type Append<O> = (T, O);

    fn init(builder: &mut SystemBuilder) -> Result<(), Error> {
        <ComponentArray<T> as SystemParam>::init(builder)
    }

    fn component_arrays<'a>(
        builder: &'a SystemBuilder,
    ) -> impl Iterator<Item = ComponentArray<()>> + 'a {
        std::iter::once(ComponentArray::<T>::param(builder).unwrap().cast())
    }

    fn map_axes() -> &'static [usize] {
        &[0]
    }

    fn component_count() -> usize {
        1
    }

    fn component_types() -> impl Iterator<Item = ComponentSchema> {
        std::iter::once(T::schema().into())
    }

    fn component_ids() -> impl Iterator<Item = ComponentId> {
        std::iter::once(T::COMPONENT_ID)
    }

    fn into_noxpr(self) -> Noxpr {
        self.into_inner()
    }
}

impl_group!(1; T1);
impl_group!(2; T1, T2);
impl_group!(3; T1, T2, T3);
impl_group!(4; T1, T2, T3, T4);
impl_group!(5; T1, T2, T3, T4, T5);
impl_group!(6; T1, T2, T3, T4, T5, T6);
impl_group!(7; T1, T2, T3, T4, T5, T6, T7);
impl_group!(8; T1, T2, T3, T4, T5, T6, T7, T8);
impl_group!(9; T1, T2, T3, T4, T5, T6, T7, T9, T10);
impl_group!(10; T1, T2, T3, T4, T5, T6, T7, T9, T10, T11);
impl_group!(11; T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12);
impl_group!(12; T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13);

impl<G: ComponentGroup> SystemParam for Query<G> {
    type Item = Self;

    fn init(builder: &mut SystemBuilder) -> Result<(), Error> {
        G::init(builder)
    }

    fn param(builder: &SystemBuilder) -> Result<Self::Item, Error> {
        Ok(G::component_arrays(builder)
            .fold(None, |mut query, a| {
                if query.is_some() {
                    query = Some(join_many(query.take().unwrap(), &a));
                } else {
                    let q: Query<_> = a.into();
                    query = Some(q.transmute());
                }
                query
            })
            .expect("query must be non empty")
            .transmute())
    }

    fn component_ids() -> impl Iterator<Item = ComponentId> {
        G::component_ids()
    }

    fn output(&self, builder: &mut SystemBuilder) -> Result<Noxpr, Error> {
        let mut outputs = vec![];
        for (expr, id) in self.exprs.iter().zip(G::component_ids()) {
            let array: ComponentArray<()> = ComponentArray {
                buffer: expr.clone(),
                len: self.len,
                entity_map: self.entity_map.clone(),
                phantom_data: PhantomData,
                component_id: id,
            };

            if let Some(var) = builder.vars.get_mut(&id)
                && var.entity_map != self.entity_map
            {
                outputs.push(update_var(
                    &var.entity_map,
                    &self.entity_map,
                    &var.buffer,
                    expr,
                ));
                continue;
            }
            outputs.push(array.buffer);
        }
        if outputs.len() == 1 {
            Ok(outputs.pop().unwrap())
        } else {
            Ok(Noxpr::tuple(outputs))
        }
    }
}

impl<G: ComponentGroup> Query<G> {
    pub fn get_component(&self, id: ComponentId) -> Option<&Noxpr> {
        let (i, _) = G::component_ids().enumerate().find(|(_, x)| id == *x)?;
        self.exprs.get(i)
    }

    pub fn insert_into_builder(&self, builder: &mut SystemBuilder) {
        let output = self.output(builder).unwrap();
        if G::component_ids().count() == 1 {
            let id = G::component_ids().next().unwrap();
            let array: ComponentArray<()> = ComponentArray {
                buffer: output,
                len: self.len,
                entity_map: self.entity_map.clone(),
                phantom_data: PhantomData,
                component_id: id,
            };
            builder.vars.insert(id, array);
        } else {
            for (i, id) in G::component_ids().enumerate() {
                let expr = output.get_tuple_element(i);
                let array: ComponentArray<()> = ComponentArray {
                    buffer: expr.clone(),
                    len: self.len,
                    entity_map: self.entity_map.clone(),
                    phantom_data: PhantomData,
                    component_id: id,
                };
                builder.vars.insert(id, array);
            }
        }
    }
}

impl<G: ComponentGroup> Query<G> {
    pub fn map<O: ComponentGroup>(
        &self,
        func: impl CompFn<G::Params, O>,
    ) -> Result<Query<O>, Error> {
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
        let func = NoxprFn {
            inner,
            args: builder.params.into_inner(),
        };

        let map_axes: SmallVec<[usize; 4]> = smallvec![0; G::component_count()];
        let buffer = Noxpr::vmap_with_axis(func, &map_axes, &self.exprs)?;
        let exprs = if O::component_count() == 1 {
            vec![buffer]
        } else {
            (0..O::component_count())
                .map(|i| buffer.get_tuple_element(i))
                .collect()
        };
        Ok(Query {
            exprs,
            len: self.len,
            phantom_data: PhantomData,
            entity_map: self.entity_map.clone(),
        })
    }

    pub fn join<B: Component>(self, other: &ComponentArray<B>) -> Query<G::Append<B>> {
        let q = join_many(self, other);
        Query {
            exprs: q.exprs,
            len: q.len,
            entity_map: q.entity_map,
            phantom_data: PhantomData,
        }
    }

    pub fn join_query<B: ComponentGroup>(self, other: Query<B>) -> Query<(G, B)> {
        let q = join_query(self, other);
        Query {
            exprs: q.exprs,
            len: q.len,
            entity_map: q.entity_map,
            phantom_data: PhantomData,
        }
    }
}

impl<G> Query<G> {
    pub fn filter(&self, ids: &[EntityId]) -> Self {
        let indexes: Vec<u32> = ids
            .iter()
            .flat_map(|id| self.entity_map.get(id).copied())
            .map(|id| id as u32)
            .collect();
        let exprs = self
            .exprs
            .iter()
            .map(|expr| filter_index(&indexes, expr))
            .collect();
        let entity_map = self
            .entity_map
            .iter()
            .filter(|(id, _)| ids.contains(id))
            .enumerate()
            .map(|(index, (id, _))| (*id, index))
            .collect();
        Query {
            exprs,
            len: indexes.len(),
            entity_map,
            phantom_data: PhantomData,
        }
    }
}

impl<A: Component> ComponentArray<A> {
    pub fn join<B: Component>(&self, other: &ComponentArray<B>) -> Query<(A, B)> {
        join_many(self.clone().into(), other).transmute()
    }
}

fn filter_index(indexes: &[u32], buffer: &Noxpr) -> Noxpr {
    let n = indexes.len();
    let indexes_lit = Literal::vector(indexes);
    let indexes = Noxpr::constant(
        indexes_lit,
        ArrayTy {
            element_type: ElementType::U32,
            shape: smallvec![indexes.len() as i64],
        },
    )
    .broadcast_in_dim(smallvec![n as i64, 1], smallvec![0]);
    let mut slice_shape = buffer.shape().unwrap();
    slice_shape[0] = 1;
    let offset_dims = (1..slice_shape.len() as i64).collect();
    buffer.clone().gather(
        indexes,
        offset_dims,
        smallvec![0],
        smallvec![0],
        slice_shape,
        1,
    )
}

pub fn join_many<A, B>(mut a: Query<A>, b: &ComponentArray<B>) -> Query<()> {
    if a.entity_map == b.entity_map {
        a.exprs.push(b.buffer.clone());
        Query {
            exprs: a.exprs,
            entity_map: a.entity_map,
            len: a.len,
            phantom_data: PhantomData,
        }
    } else {
        let (a_indexes, b_indexes, ids) = intersect_ids(&a.entity_map, &b.entity_map);
        let mut exprs = a.exprs;
        for expr in &mut exprs {
            *expr = filter_index(&a_indexes, expr);
        }
        exprs.push(filter_index(&b_indexes, &b.buffer));
        Query {
            exprs,
            len: ids.len(),
            entity_map: ids,
            phantom_data: PhantomData,
        }
    }
}

pub fn join_query<A, B>(mut a: Query<A>, mut b: Query<B>) -> Query<()> {
    if a.entity_map == b.entity_map {
        a.exprs.append(&mut b.exprs);
        Query {
            exprs: a.exprs,
            entity_map: a.entity_map,
            len: a.len,
            phantom_data: PhantomData,
        }
    } else {
        let (a_indexes, b_indexes, ids) = intersect_ids(&a.entity_map, &b.entity_map);
        for expr in &mut a.exprs {
            *expr = filter_index(&a_indexes, expr);
        }
        for expr in &mut b.exprs {
            *expr = filter_index(&b_indexes, expr);
        }
        let mut exprs = a.exprs;
        exprs.append(&mut b.exprs);
        Query {
            exprs,
            len: ids.len(),
            entity_map: ids,
            phantom_data: PhantomData,
        }
    }
}

impl<A> From<ComponentArray<A>> for Query<A> {
    fn from(value: ComponentArray<A>) -> Self {
        Query {
            exprs: vec![value.buffer],
            len: value.len,
            entity_map: value.entity_map,
            phantom_data: PhantomData,
        }
    }
}

// --- Python wrappers ---

use crate::PyComponent;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct QueryInner {
    pub query: Query<()>,
    pub metadata: Vec<PyComponent>,
}

#[pymethods]
impl QueryInner {
    #[staticmethod]
    pub fn from_arrays(
        arrays: Vec<PyObject>,
        metadata: QueryMetadata,
    ) -> Result<QueryInner, Error> {
        Ok(QueryInner {
            query: Query {
                exprs: arrays.into_iter().map(Noxpr::jax).collect(),
                entity_map: metadata.entity_map,
                len: metadata.len,
                phantom_data: PhantomData,
            },
            metadata: metadata.metadata,
        })
    }

    #[staticmethod]
    pub fn from_builder(
        builder: crate::system::PySystemBuilder,
        component_ids: Vec<String>,
        args: Vec<PyObject>,
    ) -> Result<QueryInner, Error> {
        let (query, metadata) = component_ids
            .iter()
            .map(|id| {
                let id = ComponentId::new(id);
                let (meta, i) = builder.get_var(id).ok_or(Error::ComponentNotFound)?;
                let buffer = args.get(i).ok_or(Error::ComponentNotFound)?;
                Ok::<_, Error>((
                    ComponentArray {
                        buffer: Noxpr::jax(Python::with_gil(|py| buffer.clone_ref(py))),
                        len: meta.len,
                        entity_map: meta.entity_map,
                        component_id: id,
                        phantom_data: PhantomData,
                    },
                    meta.component,
                ))
            })
            .try_fold((None, vec![]), |(mut query, mut metadata), res| {
                let (a, meta) = res?;
                metadata.push(meta);
                if query.is_some() {
                    query = Some(join_many(query.take().unwrap(), &a));
                } else {
                    let q: Query<()> = a.into();
                    query = Some(q);
                }
                Ok::<_, Error>((query, metadata))
            })?;
        let query = query.ok_or(Error::ComponentNotFound)?;
        Ok(Self { query, metadata })
    }

    pub fn map(&self, new_buf: PyObject, metadata: PyComponent) -> QueryInner {
        let expr = Noxpr::jax(new_buf);
        QueryInner {
            query: Query {
                exprs: vec![expr],
                entity_map: self.query.entity_map.clone(),
                len: self.query.len,
                phantom_data: PhantomData,
            },
            metadata: vec![metadata],
        }
    }

    pub fn output(
        &self,
        builder: crate::system::PySystemBuilder,
        args: Vec<PyObject>,
    ) -> Result<PyObject, Error> {
        let mut outputs = vec![];
        for (expr, id) in self
            .query
            .exprs
            .iter()
            .zip(self.metadata.iter().map(|m| m.component_id()))
        {
            let Some((meta, index)) = builder.get_var(id) else {
                return Err(Error::ComponentNotFound);
            };
            let buffer = args.get(index).ok_or(Error::ComponentNotFound)?;

            if meta.entity_map == self.query.entity_map {
                outputs.push(expr.clone());
            } else {
                let out = update_var(
                    &meta.entity_map,
                    &self.query.entity_map,
                    &Noxpr::jax(Python::with_gil(|py| buffer.clone_ref(py))),
                    expr,
                );
                outputs.push(out);
            }
        }
        if outputs.len() == 1 {
            outputs.pop().unwrap().to_jax().map_err(Error::from)
        } else {
            Noxpr::tuple(outputs).to_jax().map_err(Error::from)
        }
    }

    pub fn arrays(&self) -> Result<Vec<PyObject>, Error> {
        self.query
            .exprs
            .iter()
            .map(|e| e.to_jax().map_err(Error::from))
            .collect()
    }

    pub fn join_query(&self, other: &QueryInner) -> QueryInner {
        let query = join_query(self.query.clone(), other.query.clone());
        let metadata = self
            .metadata
            .iter()
            .cloned()
            .chain(other.metadata.iter().cloned())
            .collect();
        QueryInner { query, metadata }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct QueryMetadata {
    pub entity_map: BTreeMap<EntityId, usize>,
    pub len: usize,
    pub metadata: Vec<PyComponent>,
}

#[pymethods]
impl QueryMetadata {
    pub fn merge(&mut self, other: QueryMetadata) {
        self.metadata.extend(other.metadata);
    }
}

// --- Tests ---

// Query tests disabled during XLA->IREE migration. They require a full
// compile+execute pipeline which now needs Python/IREE at runtime.
#[cfg(any())]
mod tests {
    use super::*;
    use crate::world::{IntoSystemExt, WorldExt};
    use elodin_macros::{Archetype, Component, ComponentGroup, FromBuilder, ReprMonad};
    use nox::{Op, OwnedRepr, Scalar, Vector, tensor};

    #[test]
    fn test_cross_archetype_join() {
        #[derive(Clone, Component, ReprMonad)]
        struct X<R: OwnedRepr = Op>(Scalar<f64, R>);

        #[derive(Clone, Component, ReprMonad)]
        struct E<R: OwnedRepr = Op>(Scalar<f64, R>);

        #[derive(Archetype)]
        struct Body {
            x: X,
        }

        fn add_e(a: Query<(E, X)>) -> Query<X> {
            a.map(|e: E, x: X| X(x.0 + e.0)).unwrap()
        }
        let mut world = add_e.world();
        world.spawn(Body {
            x: X((-91.0).into()),
        });

        world
            .spawn(Body {
                x: X((-55.0).into()),
            })
            .insert(E(1000.0.into()));

        world.spawn(Body { x: X(5.0.into()) });
        world.spawn(Body { x: X(200.0.into()) });

        world
            .spawn(Body { x: X(100.0.into()) })
            .insert(E((-50000.0).into()));
        world.spawn(Body { x: X(400.0.into()) });

        let world = world.run();
        let c = world.column::<X>().unwrap();
        assert_eq!(
            c.typed_buf::<f64>().unwrap(),
            &[-91.0, 945.0, 5.0, 200.0, -49900.0, 400.0]
        );
    }

    #[test]
    fn component_group() {
        #[derive(Component, ReprMonad)]
        struct A<R: OwnedRepr = Op>(Scalar<f64, R>);

        #[derive(Component, ReprMonad)]
        struct B<R: OwnedRepr = Op>(Scalar<f64, R>);

        #[derive(Component, ReprMonad)]
        struct C<R: OwnedRepr = Op>(Scalar<f64, R>);

        #[derive(Archetype)]
        struct Body {
            a: A,
            b: B,
            c: C,
        }

        #[derive(FromBuilder, ComponentGroup)]
        struct AB {
            a: A,
            b: B,
        }

        fn add_system(q: Query<AB>) -> Query<C> {
            q.map(|ab: AB| C(ab.a.0 + ab.b.0)).unwrap()
        }

        let mut world = add_system.world();
        world.spawn(Body {
            a: A(1.0.into()),
            b: B(2.0.into()),
            c: C((-1.0).into()),
        });

        world.spawn(Body {
            a: A(2.0.into()),
            b: B(2.0.into()),
            c: C((-1.0).into()),
        });
        let world = world.run();
        let c = world.column::<C>().unwrap();
        assert_eq!(c.typed_buf::<f64>().unwrap(), &[3.0, 4.0])
    }

    #[test]
    fn test_simple() {
        #[derive(Component, ReprMonad)]
        struct A<R: OwnedRepr = Op>(Scalar<f64, R>);

        #[derive(Component, ReprMonad)]
        struct B<R: OwnedRepr = Op>(Scalar<f64, R>);

        #[derive(Component, ReprMonad)]
        struct C<R: OwnedRepr = Op>(Scalar<f64, R>);

        #[derive(Archetype)]
        struct Body {
            a: A,
            b: B,
            c: C,
        }

        fn add_system(a: Query<(A, B)>) -> Query<C> {
            a.map(|a: A, b: B| C(a.0 + b.0)).unwrap()
        }

        let mut world = add_system.world();
        world.spawn(Body {
            a: A(1.0.into()),
            b: B(2.0.into()),
            c: C((-1.0).into()),
        });

        world.spawn(Body {
            a: A(2.0.into()),
            b: B(2.0.into()),
            c: C((-1.0).into()),
        });
        let world = world.run();
        let c = world.column::<C>().unwrap();
        assert_eq!(c.typed_buf::<f64>().unwrap(), &[3.0, 4.0])
    }

    #[test]
    fn test_get_scalar() {
        #[derive(Component, ReprMonad)]
        struct A<R: OwnedRepr = Op>(Scalar<f64, R>);

        #[derive(Component, ReprMonad)]
        struct B<R: OwnedRepr = Op>(Scalar<f64, R>);

        fn add_system(s: ComponentArray<A>, v: ComponentArray<B>) -> ComponentArray<B> {
            v.map(|v: B| B(v.0 + s.get(0).0)).unwrap()
        }

        let mut world = add_system.world();
        world.spawn(A(5.0.into()));
        world.spawn(B((-1.0).into()));
        world.spawn(B(7.0.into()));
        let world = world.run();
        let v = world.column::<B>().unwrap();
        assert_eq!(v.typed_buf::<f64>().unwrap(), &[4.0, 12.0])
    }

    #[test]
    fn test_get_tensor() {
        #[derive(Component, ReprMonad)]
        struct A<R: OwnedRepr = Op>(Vector<f64, 3, R>);

        #[derive(Component, ReprMonad)]
        struct B<R: OwnedRepr = Op>(Vector<f64, 3, R>);

        fn add_system(s: ComponentArray<A>, v: ComponentArray<B>) -> ComponentArray<B> {
            v.map(|v: B| B(v.0 + s.get(0).0)).unwrap()
        }

        let mut world = add_system.world();
        world.spawn(A(tensor![5.0, 2.0, -3.0].into()));
        world.spawn(B(tensor![-1.0, 3.5, 6.0].into()));
        world.spawn(B(tensor![7.0, -1.0, 1.0].into()));
        let world = world.run();
        let v = world.column::<B>().unwrap();
        assert_eq!(
            v.typed_buf::<f64>().unwrap(),
            &[4.0, 5.5, 3.0, 12.0, 1.0, -2.0]
        )
    }

    #[test]
    fn test_startup() {
        #[derive(Component, ReprMonad)]
        struct A<R: OwnedRepr = Op>(Scalar<f64, R>);

        fn startup(a: ComponentArray<A>) -> ComponentArray<A> {
            a.map(|a: A| A(a.0 * 3.0)).unwrap()
        }

        fn tick(a: ComponentArray<A>) -> ComponentArray<A> {
            a.map(|a: A| A(a.0 + 1.0)).unwrap()
        }

        let mut world = crate::World::default();
        world.spawn(A(1.0.into()));
        let world = world
            .builder()
            .tick_pipeline(tick)
            .startup_pipeline(startup)
            .run();
        let c = world.column::<A>().unwrap();
        assert_eq!(c.typed_buf::<f64>().unwrap(), &[4.0]);
    }
}
