use std::time::Duration;

use conduit::{ser_de::Frozen, ComponentId, EntityId, Metadata, Query, ValueRepr};
use roci::*;
use roci_macros::{Componentize, Decomponentize};

#[derive(Default, Debug, Componentize, Decomponentize)]
struct BarWorld {
    #[roci(entity_id = 0, component_id = "bar")]
    bar: [f32; 3],
}

struct BarHandler;

impl Handler for BarHandler {
    type World = BarWorld;
    fn tick(&mut self, world: &mut Self::World) {
        world.bar[0] += 1.0;
    }
}

fn main() {
    tracing_subscriber::fmt::init();
    roci::tcp::builder(
        BarHandler,
        Duration::from_millis(200),
        "127.0.0.1:2242".parse().unwrap(),
    )
    .run()
}
