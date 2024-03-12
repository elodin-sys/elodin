use crate::{Component, ComponentArray, Error, SystemParam};
use conduit::{ComponentId, ComponentType, EntityId};
use nox::{xla, ArrayTy, CompFn, IntoOp, Noxpr};
use smallvec::{smallvec, SmallVec};
use std::{collections::BTreeMap, marker::PhantomData};

pub struct Query<Param> {
    pub exprs: Vec<Noxpr>,
    pub entity_map: BTreeMap<EntityId, usize>,
    pub len: usize,
    pub phantom_data: PhantomData<Param>,
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

    fn init_params(builder: &mut crate::PipelineBuilder) -> Result<(), Error>;
    fn component_arrays(
        builder: &'_ crate::PipelineBuilder,
    ) -> impl Iterator<Item = ComponentArray<()>> + '_;
    fn component_types() -> impl Iterator<Item = ComponentType>;
    fn component_ids() -> impl Iterator<Item = ComponentId>;
    fn component_count() -> usize;

    fn map_axes() -> &'static [usize];
}

macro_rules! impl_group {
    ($num:tt; $($param:tt),*) => {
      impl<$($param),*> ComponentGroup for ($($param,)*)
            where $($param: ComponentGroup),*
        {
            type Params = Self;
            type Append<O> = ($($param,)* O);

            fn init_params(builder: &mut crate::PipelineBuilder) -> Result<(), crate::Error> {
                $(
                    <$param>::init_params(builder)?;
                )*
                Ok(())
            }


            fn component_arrays<'a>(
                builder: &'a crate::PipelineBuilder,
            ) -> impl Iterator<Item = ComponentArray<()>> + 'a {
                let iter = std::iter::empty();
                $(
                    let iter = iter.chain($param::component_arrays(builder));
                )*
                iter
            }

            fn component_types() -> impl Iterator<Item = ComponentType> {
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

    fn init_params(builder: &mut crate::PipelineBuilder) -> Result<(), Error> {
        ComponentArray::<T>::init(builder)
    }

    fn component_arrays(
        builder: &'_ crate::PipelineBuilder,
    ) -> impl Iterator<Item = ComponentArray<()>> + '_ {
        std::iter::once(ComponentArray::<T>::from_builder(builder).cast())
    }

    fn map_axes() -> &'static [usize] {
        &[0]
    }

    fn component_count() -> usize {
        1
    }

    fn component_types() -> impl Iterator<Item = ComponentType> {
        std::iter::once(T::component_type())
    }

    fn component_ids() -> impl Iterator<Item = ComponentId> {
        std::iter::once(T::component_id())
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

    fn init(builder: &mut crate::PipelineBuilder) -> Result<(), Error> {
        G::init_params(builder)
    }

    fn from_builder(builder: &crate::PipelineBuilder) -> Self::Item {
        G::component_arrays(builder)
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
            .transmute()
    }

    fn insert_into_builder(self, builder: &mut crate::PipelineBuilder) {
        self.insert_into_builder_erased(builder, G::component_ids());
    }
}

impl<C> Query<C> {
    #[doc(hidden)]
    pub fn insert_into_builder_erased(
        &self,
        builder: &mut crate::PipelineBuilder,
        component_ids: impl Iterator<Item = ComponentId>,
    ) {
        for (expr, id) in self.exprs.iter().zip(component_ids) {
            let array: ComponentArray<()> = ComponentArray {
                buffer: expr.clone(),
                len: self.len,
                entity_map: self.entity_map.clone(),
                phantom_data: PhantomData,
            };

            if let Some(var) = builder.vars.get_mut(&id) {
                let mut var = var.borrow_mut();
                if var.entity_map != self.entity_map {
                    var.buffer =
                        crate::update_var(&var.entity_map, &self.entity_map, &var.buffer, expr);
                    return;
                }
            }
            builder.vars.insert(id, array.into());
        }
    }
}

impl<G: ComponentGroup> Query<G> {
    pub fn map<O: ComponentGroup + IntoOp>(
        &self,
        func: impl CompFn<G::Params, O>,
    ) -> Result<Query<O>, Error> {
        let func = func.build_expr()?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Archetype, IntoSystem};
    use nox::{Scalar, ScalarExt};
    use nox_ecs_macros::{ComponentGroup, FromBuilder};

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

        fn add_e(a: Query<(E, X)>) -> Query<X> {
            a.map(|e: E, x: X| X(x.0 + e.0)).unwrap()
        }
        let mut world = add_e.world();
        world.spawn(Body {
            x: X((-91.0).constant()),
        });

        world
            .spawn(Body {
                x: X((-55.0).constant()),
            })
            .insert(E(1000.0.constant()));

        world.spawn(Body {
            x: X(5.0.constant()),
        });
        world.spawn(Body {
            x: X(200.0.constant()),
        });

        world
            .spawn(Body {
                x: X(100.0.constant()),
            })
            .insert(E((-50000.0).constant()));
        world.spawn(Body {
            x: X(400.0.constant()),
        });

        let client = nox::Client::cpu().unwrap();
        let mut exec = world.build().unwrap();
        exec.run(&client).unwrap();
        let c = exec.column(X::component_id()).unwrap();
        assert_eq!(
            c.typed_buf::<f64>().unwrap(),
            &[-91.0, 945.0, 5.0, 200.0, -49900.0, 400.0]
        );
    }

    #[test]
    fn component_group() {
        #[derive(Component)]
        struct A(Scalar<f64>);

        #[derive(Component)]
        struct B(Scalar<f64>);

        #[derive(Component)]
        struct C(Scalar<f64>);

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
            a: A(1.0.constant()),
            b: B(2.0.constant()),
            c: C((-1.0).constant()),
        });

        world.spawn(Body {
            a: A(2.0.constant()),
            b: B(2.0.constant()),
            c: C((-1.0).constant()),
        });
        let client = nox::Client::cpu().unwrap();
        let mut exec = world.build().unwrap();
        exec.run(&client).unwrap();
        let c = exec.column(C::component_id()).unwrap();
        assert_eq!(c.typed_buf::<f64>().unwrap(), &[3.0, 4.0])
    }
}
