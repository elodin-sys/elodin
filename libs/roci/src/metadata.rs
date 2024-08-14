use conduit::{ComponentId, Metadata};

pub trait Metadatatize {
    fn get_metadata(&self, component_id: ComponentId) -> Option<&Metadata>;

    fn metadata() -> impl Iterator<Item = Metadata> {
        std::iter::empty()
    }
}
