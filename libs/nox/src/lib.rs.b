use std::{any::Any, cell::UnsafeCell, marker::PhantomData, sync::Arc};

use nalgebra::{dimension::Const, ArrayStorage, ClosedAdd, Scalar};

pub type Matrix<T, const R: usize, const C: usize> =
    nalgebra::Matrix<T, Const<R>, Const<C>, ArrayStorage<T, R, C>>;

pub trait Computation {
    type Output;

    fn exec<Context: ExecContext>(self, context: &mut Context) -> Context::Handle<Self::Output>;
}

pub trait ExecContext {
    type Handle<Output>;

    fn matrix_literal<const R: usize, const C: usize, T>(
        &mut self,
        literal: MatrixLiteral<T, R, C>,
    ) -> Self::Handle<Matrix<T, R, C>>;

    fn add_matrix<T, A, B, const R: usize, const C: usize>(
        &mut self,
        a: A,
        b: B,
    ) -> Self::Handle<Matrix<T, R, C>>
    where
        A: Computation<Output = Matrix<T, R, C>>,
        B: Computation<Output = Matrix<T, R, C>>,
        T: ClosedAdd + Scalar;

    // fn store_output_handle<C: Computation>(&mut self, comp: C) -> Self::Handle<C::Output>;

    // fn get_output<C: Computation>(&mut self, handle: ExecHandle<C>) -> Self::Handle<C::Output>;
}

pub trait MatrixExt<T, const R: usize, const C: usize> {
    fn literal(self) -> MatrixLiteral<T, R, C>;
}

impl<T, const R: usize, const C: usize> MatrixExt<T, R, C> for Matrix<T, R, C> {
    fn literal(self) -> MatrixLiteral<T, R, C> {
        MatrixLiteral { literal: self }
    }
}

pub struct MatrixLiteral<T, const R: usize, const C: usize> {
    literal: Matrix<T, R, C>,
}

impl<const R: usize, const C: usize, T> Computation for MatrixLiteral<T, R, C> {
    type Output = Matrix<T, R, C>;

    fn exec<Context: ExecContext>(self, context: &mut Context) -> Context::Handle<Self::Output> {
        context.matrix_literal(self)
    }
}

pub struct AddMatrix<T, A, B, const R: usize, const C: usize>
where
    A: Computation<Output = Matrix<T, R, C>>,
    B: Computation<Output = Matrix<T, R, C>>,
    T: ClosedAdd + Scalar,
{
    a: A,
    b: B,
}

struct Nalgebra;

impl ExecContext for Nalgebra {
    type Handle<Output> = Output;

    fn matrix_literal<const R: usize, const C: usize, T>(
        &mut self,
        literal: MatrixLiteral<T, R, C>,
    ) -> Self::Handle<Matrix<T, R, C>> {
        literal.literal
    }

    fn add_matrix<T, A, B, const R: usize, const C: usize>(
        &mut self,
        a: A,
        b: B,
    ) -> Self::Handle<Matrix<T, R, C>>
    where
        A: Computation<Output = Matrix<T, R, C>>,
        B: Computation<Output = Matrix<T, R, C>>,
        T: ClosedAdd + Scalar,
    {
        let a: Matrix<T, R, C> = a.exec(self);
        let b: Matrix<T, R, C> = b.exec(self);
        a + b
    }
}

pub trait ComputationExt {
    fn add<T: ClosedAdd + Scalar, B, const R: usize, const C: usize>(
        self,
        rhs: B,
    ) -> AddMatrix<T, Self, B, R, C>
    where
        Self: Computation<Output = Matrix<T, R, C>> + Sized,
        B: Computation<Output = Matrix<T, R, C>>;

    fn c(self) -> Comp<Self>
    where
        Self: Sized + Computation,
    {
        todo!()
        //Comp(self)
    }
}

impl<A> ComputationExt for A
where
    A: Computation,
{
    fn add<T: ClosedAdd + Scalar, B, const R: usize, const C: usize>(
        self,
        rhs: B,
    ) -> AddMatrix<T, Self, B, R, C>
    where
        Self: Computation<Output = Matrix<T, R, C>> + Sized,
        B: Computation<Output = Matrix<T, R, C>>,
    {
        AddMatrix { a: self, b: rhs }
    }
}

impl<T, A, B, const R: usize, const C: usize> Computation for AddMatrix<T, A, B, R, C>
where
    A: Computation<Output = Matrix<T, R, C>>,
    B: Computation<Output = Matrix<T, R, C>>,
    T: ClosedAdd + Scalar,
{
    type Output = Matrix<T, R, C>;

    fn exec<Context: ExecContext>(self, context: &mut Context) -> Context::Handle<Self::Output> {
        context.add_matrix(self.a, self.b)
    }
}

enum CompInner<C: Computation> {
    Unexec(C),
    Exec(Arc<dyn Any>),
    None,
}

pub struct Comp<C: Computation>(Arc<UnsafeCell<CompInner<C>>>);

impl<C: Computation> Computation for Comp<C> {
    type Output = C::Output;

    fn exec<Context: ExecContext>(self, context: &mut Context) -> Context::Handle<Self::Output> {
        if let CompInner::Exec(e) = unsafe { &*self.0.get() } {
            todo!()
        }

        let slot = unsafe { &mut *self.0.get() };
        let CompInner::Unexec(comp) = std::mem::replace(slot, CompInner::None) else {
            panic!("comp inner must be exec")
        };

        let output = comp.exec(context);
        *slot = CompInner::Exec(Arc::new(output));

        //*slot = CompInner::Exec(context.store_output_handle(comp));
    }
}

impl<T, A, B, const R: usize, const C: usize> std::ops::Add<B> for Comp<A>
where
    A: Computation<Output = Matrix<T, R, C>>,
    B: Computation<Output = Matrix<T, R, C>>,
    T: ClosedAdd + Scalar,
{
    type Output = AddMatrix<T, A, B, R, C>;

    fn add(self, rhs: B) -> Self::Output {
        AddMatrix { a: self.0, b: rhs }
    }
}

pub struct Map<C, I, F, T>
where
    C: Computation<Output = I>,
    I: NoxIter<Item = T>,
{
    iter: C,
    func: F,
}

impl<C, I, F, T> Computation for Map<C, I, F, T>
where
    C: Computation<Output = I>,
    I: NoxIter<Item = T>,
    // P: Computation<Output = T>,
    //F: Fn(P),
{
    type Output = ();

    fn exec<Context: ExecContext>(self, context: &Context) -> Context::Handle<Self::Output> {
        todo!()
    }
}

pub trait MapFunc {
    type Param;
    fn apply(self, param: impl Computation<Output = Self::Param>);
}

// impl<P, C: Computation<Output = P>, F: Fn(C)> MapFunc for F {
//     type Param = P;

//     fn apply(self, param: impl Computation<Output = Self::Param>) {}
// }

pub trait NoxIter {
    type Item;
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use crate::{Computation, ComputationExt, MatrixExt, Nalgebra};

    #[test]
    fn test_nalgebra_add() {
        let a = matrix![1, 2, 3].literal();
        let b = matrix![2, 3, 4].literal();
        let c = matrix![2, 3, 4].literal();
        let out = (a.c() + b.c()).c() + c.c();
        let out = out.exec(&Nalgebra);
        assert_eq!(out, matrix![5, 8, 11]);
    }
}
