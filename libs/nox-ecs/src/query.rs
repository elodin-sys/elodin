use crate::{Component, ComponentArray, Error, SystemParam};
use elodin_conduit::EntityId;
use nox::{xla, ArrayTy, CompFn, Noxpr};
use smallvec::smallvec;
use std::{collections::BTreeMap, marker::PhantomData};

pub struct Query<Param> {
    pub exprs: Vec<Noxpr>,
    pub entity_map: BTreeMap<EntityId, usize>,
    pub len: usize,
    pub phantom_data: PhantomData<Param>,
}

impl<Param> Query<Param> {
    fn transmute<B>(self) -> Query<B> {
        Query {
            exprs: self.exprs,
            entity_map: self.entity_map,
            len: self.len,
            phantom_data: PhantomData,
        }
    }
}

macro_rules! impl_query {
    ($num:tt; $($param:tt),*) => {
        impl<$($param),*> SystemParam for Query<($($param,)*)>
            where $(ComponentArray<$param>: SystemParam<Item = ComponentArray<$param>>),*
        {
            type Item = Self;

            fn init(builder: &mut crate::PipelineBuilder) -> Result<(), crate::Error> {
                $(
                    ComponentArray::<$param>::init(builder)?;
                )*
                Ok(())
            }

            fn from_builder(builder: &crate::PipelineBuilder) -> Self {
                let mut query = None;
                $(
                    let a = ComponentArray::<$param>::from_builder(builder);
                    if query.is_some() {
                        query = Some(join_many(query.take().unwrap(), &a));
                    } else {
                        let q: Query<_> = a.into();
                        query = Some(q.transmute());
                    }
                )*
                let query = query.unwrap();
                Self {
                    exprs: query.exprs,
                    len: query.len,
                    entity_map: query.entity_map,
                    phantom_data: PhantomData,
                }
            }

            fn insert_into_builder(self, _builder: &mut crate::PipelineBuilder) {
            }
        }

        impl<$($param),*> Query<($($param,)*)>
            where $(ComponentArray<$param>: SystemParam<Item = ComponentArray<$param>>),*
        {
            pub fn map<O: Component>(&self, func: impl CompFn<($($param,)*), O>) -> Result<ComponentArray<O>, Error> {
                let func = func.build_expr()?;
                let buffer = Noxpr::vmap_with_axis(func, &[0; $num], &self.exprs)?;
                Ok(ComponentArray {
                    buffer,
                    len: self.len,
                    phantom_data: PhantomData,
                    entity_map: self.entity_map.clone(),
                })
            }

            pub fn join<B: Component>(self, other: &ComponentArray<B>) -> Query<($($param,)* B)> {
                let q = join_many(self, other);
                Query {
                    exprs: q.exprs,
                    len: q.len,
                    entity_map: q.entity_map,
                    phantom_data: PhantomData,
                }
            }
        }
    }
}

impl_query!(1; T1);
impl_query!(2; T1, T2);
impl_query!(3; T1, T2, T3);
impl_query!(4; T1, T2, T3, T4);
impl_query!(5; T1, T2, T3, T4, T5);
impl_query!(6; T1, T2, T3, T4, T5, T6);
impl_query!(7; T1, T2, T3, T4, T5, T6, T7);
impl_query!(8; T1, T2, T3, T4, T5, T6, T7, T8);
impl_query!(9; T1, T2, T3, T4, T5, T6, T7, T9, T10);
impl_query!(10; T1, T2, T3, T4, T5, T6, T7, T9, T10, T11);
impl_query!(11; T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12);
impl_query!(12; T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Archetype, System};
    use nox::Scalar;

    #[test]
    fn test_cross_archetype_join() {
        #[derive(Clone, Component)]
        struct X(Scalar<f64>);

        #[derive(Clone, Component)]
        struct E(Scalar<f64>);

        #[derive(Archetype)]
        struct Body {
            x: X,
        }

        fn add_e(a: Query<(E, X)>) -> ComponentArray<X> {
            a.map(|e: E, x: X| X(x.0 + e.0)).unwrap()
        }
        let mut world = add_e.world();
        world.spawn(Body { x: X::host(-91.0) });

        let id = world.spawn(Body { x: X::host(-55.0) });
        world.spawn_with_id(E::host(1000.0), id);

        world.spawn(Body { x: X::host(5.0) });
        world.spawn(Body { x: X::host(200.0) });

        let id = world.spawn(Body { x: X::host(100.0) });
        world.spawn_with_id(E::host(-50000.0), id);
        world.spawn(Body { x: X::host(400.0) });

        let client = nox::Client::cpu().unwrap();
        let mut exec = world.build(&client).unwrap();
        exec.run(&client).unwrap();
        let c = exec.client_world.column::<X>().unwrap();
        let lit = c.column.buffer.to_literal_sync().unwrap();
        assert_eq!(
            lit.typed_buf::<f64>().unwrap(),
            &[-91.0, 945.0, 5.0, 200.0, -49900.0, 400.0]
        );
    }
}
