use crate::ecs::{
    Component, component_array::ComponentArray, system::SystemBuilder, system::SystemParam,
};
use elodin_db::ComponentSchema;
use impeller2::types::{ComponentId, EntityId};
use nox::{ArrayTy, Builder, CompFn, Noxpr, NoxprFn, ReprMonad, xla};
use smallvec::{SmallVec, smallvec};
use std::{collections::BTreeMap, marker::PhantomData};

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

pub trait ComponentGroup {
    type Params;
    type Append<O>;

    fn init(builder: &mut SystemBuilder) -> Result<(), crate::Error>;

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

            fn init(builder: &mut SystemBuilder) -> Result<(), crate::Error> {
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

    fn init(builder: &mut SystemBuilder) -> Result<(), crate::Error> {
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

    fn init(builder: &mut SystemBuilder) -> Result<(), crate::Error> {
        G::init(builder)
    }

    fn param(builder: &SystemBuilder) -> Result<Self::Item, crate::Error> {
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

    fn output(&self, builder: &mut SystemBuilder) -> Result<Noxpr, crate::Error> {
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
                outputs.push(crate::ecs::component_array::update_var(
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
    ) -> Result<Query<O>, crate::Error> {
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
    let indexes_lit = xla::Literal::vector(indexes);
    let indexes = Noxpr::constant(
        indexes_lit,
        ArrayTy {
            element_type: xla::ElementType::U32,
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
