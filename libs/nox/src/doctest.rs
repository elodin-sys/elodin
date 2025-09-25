#[cfg(feature = "noxpr")]
pub mod noxpr {
    use crate::{ArrayTy, Noxpr, NoxprFn, NoxprScalarExt, NoxprTy};
    use smallvec::smallvec;
    use xla::ElementType;

    pub fn example_function() -> NoxprFn {
        // Parameters
        let a = Noxpr::parameter(
            0,
            NoxprTy::ArrayTy(ArrayTy::new(ElementType::F32, smallvec![3])),
            "a".into(),
        );
        let b = Noxpr::parameter(
            1,
            NoxprTy::ArrayTy(ArrayTy::new(ElementType::F32, smallvec![3])),
            "b".into(),
        );

        // ((a + 1) * b).dot(a)
        let expr = ((a.clone() + 1.0f32.constant()) * b.clone()).dot(&a);

        // Function wrapper (for compile / pretty-print / etc.)
        NoxprFn::new(vec![a, b], expr)
    }
}
