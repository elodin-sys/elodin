use nox::{Op, OwnedRepr, ReprMonad, Scalar};

pub trait Component:
    impeller2::component::Component + for<'a> nox::FromBuilder<Item<'a> = Self> + ReprMonad<Op>
{
}

// Core component types used in the system
#[derive(Clone)]
pub struct WorldPos<R: OwnedRepr = Op>(pub nox::SpatialTransform<f64, R>);

#[derive(Clone)]
pub struct Seed<R: OwnedRepr = Op>(pub Scalar<u64, R>);

#[derive(Clone)]
pub struct Time<R: OwnedRepr = Op>(pub Scalar<f64, R>);

impl Seed {
    pub fn zero() -> Self {
        Seed(0u64.into())
    }
}

impl Time {
    pub fn zero() -> Self {
        Time(Scalar::from(0f64))
    }
}

// Manual Component implementations (without macros)
impl<R: OwnedRepr> impeller2::component::Component for WorldPos<R> {
    const NAME: &'static str = "world_pos";
    
    fn schema() -> impeller2::schema::Schema<Vec<u64>> {
        nox::SpatialTransform::<f64, R>::schema()
    }
}

impl<R: OwnedRepr> Component for WorldPos<R> 
where
    Self: ReprMonad<Op> + for<'a> nox::FromBuilder<Item<'a> = Self>
{}

impl<R: OwnedRepr> ReprMonad<R> for WorldPos<R> {
    type Elem = <nox::SpatialTransform<f64, R> as ReprMonad<R>>::Elem;
    type Dim = <nox::SpatialTransform<f64, R> as ReprMonad<R>>::Dim;
    type Map<NewRepr: OwnedRepr> = WorldPos<NewRepr>;

    fn map<N: OwnedRepr>(
        self,
        func: impl Fn(R::Inner<Self::Elem, Self::Dim>) -> N::Inner<Self::Elem, Self::Dim>,
    ) -> Self::Map<N> {
        WorldPos(self.0.map(func))
    }

    fn into_inner(self) -> R::Inner<Self::Elem, Self::Dim> {
        self.0.into_inner()
    }

    fn inner(&self) -> &R::Inner<Self::Elem, Self::Dim> {
        self.0.inner()
    }

    fn from_inner(inner: R::Inner<Self::Elem, Self::Dim>) -> Self {
        WorldPos(nox::SpatialTransform::from_inner(inner))
    }
}

// Seed implementations
impl<R: OwnedRepr> impeller2::component::Component for Seed<R> {
    const NAME: &'static str = "seed";
    
    fn schema() -> impeller2::schema::Schema<Vec<u64>> {
        Scalar::<u64, R>::schema()
    }
}

impl<R: OwnedRepr> Component for Seed<R> 
where
    Self: ReprMonad<Op> + for<'a> nox::FromBuilder<Item<'a> = Self>
{}

impl<R: OwnedRepr> ReprMonad<R> for Seed<R> {
    type Elem = <Scalar<u64, R> as ReprMonad<R>>::Elem;
    type Dim = <Scalar<u64, R> as ReprMonad<R>>::Dim;
    type Map<NewRepr: OwnedRepr> = Seed<NewRepr>;

    fn map<N: OwnedRepr>(
        self,
        func: impl Fn(R::Inner<Self::Elem, Self::Dim>) -> N::Inner<Self::Elem, Self::Dim>,
    ) -> Self::Map<N> {
        Seed(self.0.map(func))
    }

    fn into_inner(self) -> R::Inner<Self::Elem, Self::Dim> {
        self.0.into_inner()
    }

    fn inner(&self) -> &R::Inner<Self::Elem, Self::Dim> {
        self.0.inner()
    }

    fn from_inner(inner: R::Inner<Self::Elem, Self::Dim>) -> Self {
        Seed(Scalar::from_inner(inner))
    }
}

// Time implementations  
impl<R: OwnedRepr> impeller2::component::Component for Time<R> {
    const NAME: &'static str = "time";
    
    fn schema() -> impeller2::schema::Schema<Vec<u64>> {
        Scalar::<f64, R>::schema()
    }
}

impl<R: OwnedRepr> Component for Time<R> 
where
    Self: ReprMonad<Op> + for<'a> nox::FromBuilder<Item<'a> = Self>
{}

impl<R: OwnedRepr> ReprMonad<R> for Time<R> {
    type Elem = <Scalar<f64, R> as ReprMonad<R>>::Elem;
    type Dim = <Scalar<f64, R> as ReprMonad<R>>::Dim;
    type Map<NewRepr: OwnedRepr> = Time<NewRepr>;

    fn map<N: OwnedRepr>(
        self,
        func: impl Fn(R::Inner<Self::Elem, Self::Dim>) -> N::Inner<Self::Elem, Self::Dim>,
    ) -> Self::Map<N> {
        Time(self.0.map(func))
    }

    fn into_inner(self) -> R::Inner<Self::Elem, Self::Dim> {
        self.0.into_inner()
    }

    fn inner(&self) -> &R::Inner<Self::Elem, Self::Dim> {
        self.0.inner()
    }

    fn from_inner(inner: R::Inner<Self::Elem, Self::Dim>) -> Self {
        Time(Scalar::from_inner(inner))
    }
}


