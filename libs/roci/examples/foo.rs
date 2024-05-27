use std::time::Duration;

use conduit::{ser_de::Frozen, ComponentId, EntityId, Metadata, Query, ValueRepr};
use roci::*;
use roci_macros::{Componentize, Decomponentize};

#[derive(Default, Debug, Componentize, Decomponentize)]
struct FooWorld {
    #[roci(entity_id = 0, component_id = "foo")]
    foo: [f32; 3],
    #[roci(entity_id = 0, component_id = "bar")]
    bar: [f32; 3],
}

struct FooHandler;

impl Handler for FooHandler {
    type World = FooWorld;
    fn tick(&mut self, world: &mut Self::World) {
        println!("tick {:?}", world);
    }
}

fn main() {
    tracing_subscriber::fmt::init();
    roci::tcp::builder(
        FooHandler,
        Duration::from_millis(200),
        "127.0.0.1:2241".parse().unwrap(),
    )
    .subscribe(
        Query::with_id(ComponentId::new("bar")),
        "127.0.0.1:2242".parse().unwrap(),
    )
    .run()
}
