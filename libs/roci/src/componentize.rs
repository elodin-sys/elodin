use crate::Decomponentize;
use conduit::{ComponentId, Metadata};

pub trait Componentize {
    fn sink_columns(&self, output: &mut impl Decomponentize);
    fn get_metadata(&self, component_id: ComponentId) -> Option<&Metadata>;

    fn metadata() -> impl Iterator<Item = Metadata> {
        std::iter::empty()
    }

    const MAX_SIZE: usize = usize::MAX;
}

impl Componentize for () {
    fn sink_columns(&self, _output: &mut impl Decomponentize) {}

    fn get_metadata(&self, _component_id: ComponentId) -> Option<&Metadata> {
        None
    }

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

    fn get_metadata(&self, component_id: ComponentId) -> Option<&Metadata> {
        self.1
            .get_metadata(component_id)
            .or_else(|| self.0.get_metadata(component_id))
    }

    const MAX_SIZE: usize = T1::MAX_SIZE + T2::MAX_SIZE;
}
