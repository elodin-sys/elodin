use crate::Decomponentize;

pub trait Componentize {
    fn sink_columns(&self, output: &mut impl Decomponentize);

    const MAX_SIZE: usize = usize::MAX;
}

impl Componentize for () {
    fn sink_columns(&self, _output: &mut impl Decomponentize) {}

    const MAX_SIZE: usize = 0;
}

impl<T1, T2> Componentize for (T1, T2)
where
    T1: Componentize,
    T2: Componentize,
{
    fn sink_columns(&self, output: &mut impl Decomponentize) {
        self.0.sink_columns(output);
        self.1.sink_columns(output);
    }

    const MAX_SIZE: usize = T1::MAX_SIZE + T2::MAX_SIZE;
}

impl<T1, T2, T3> Componentize for (T1, T2, T3)
where
    T1: Componentize,
    T2: Componentize,
    T3: Componentize,
{
    fn sink_columns(&self, output: &mut impl Decomponentize) {
        self.0.sink_columns(output);
        self.1.sink_columns(output);
        self.2.sink_columns(output);
    }

    const MAX_SIZE: usize = T1::MAX_SIZE + T2::MAX_SIZE + T3::MAX_SIZE;
}
