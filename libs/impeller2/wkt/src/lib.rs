#[cfg(feature = "nox")]
use impeller2::component::Component;
use impeller2::types::Timestamp;
use serde::{Deserialize, Serialize};

mod assets;
mod metadata;
mod msgs;
mod path;
#[cfg(feature = "nox")]
mod value;

pub use assets::*;
pub use metadata::*;
pub use msgs::*;
pub use path::*;
#[cfg(feature = "nox")]
pub use value::*;

#[cfg(feature = "gui")]
mod gui;
#[cfg(feature = "gui")]
pub use gui::*;

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Color {
    pub const BLACK: Self = Self::rgb(0., 0., 0.);
    pub const WHITE: Self = Self::rgb(1., 1., 1.);

    pub const TURQUOISE: Self = Self::rgb(0.41, 0.7, 0.75);
    pub const SLATE: Self = Self::rgb(0.5, 0.44, 1.);
    pub const PUMPKIN: Self = Self::rgb(1.0, 0.44, 0.12);
    pub const YOLK: Self = Self::rgb(1., 0.77, 0.02);
    pub const PEACH: Self = Self::rgb(1., 0.84, 0.7);
    pub const REDDISH: Self = Self::rgb(0.913, 0.125, 0.0335);
    pub const HYPERBLUE: Self = Self::rgb(0.08, 0.38, 0.82);
    pub const MINT: Self = Self::rgb(0.53, 0.87, 0.62);

    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }
}

#[derive(Clone, Serialize, Deserialize, Default, Copy)]
#[cfg_attr(
    feature = "bevy",
    derive(bevy::prelude::Resource, bevy::prelude::Component)
)]
pub struct Tick(pub u64);

#[cfg(feature = "nox")]
impl impeller2::com_de::Decomponentize for Tick {
    type Error = core::convert::Infallible;

    fn apply_value(
        &mut self,
        component_id: impeller2::types::ComponentId,
        value: impeller2::types::ComponentView<'_>,
        _timestamp: Option<Timestamp>,
    ) -> Result<(), Self::Error> {
        if component_id != Tick::COMPONENT_ID {
            return Ok(());
        }
        let impeller2::types::ComponentView::U64(view) = value else {
            return Ok(());
        };
        let buf = view.buf();
        self.0 = buf[0];
        Ok(())
    }
}

impl impeller2::component::Component for Tick {
    const NAME: &'static str = "tick";

    fn schema() -> impeller2::schema::Schema<Vec<u64>> {
        impeller2::schema::Schema::new(impeller2::types::PrimType::U64, [0u64; 0])
            .expect("failed to create schema")
    }
}

#[derive(Clone, Serialize, Deserialize, Default)]
#[cfg_attr(
    feature = "bevy",
    derive(bevy::prelude::Resource, bevy::prelude::Component)
)]
pub struct SimulationTimeStep(pub f64);

impl SimulationTimeStep {
    pub fn as_duration(&self) -> std::time::Duration {
        std::time::Duration::from_secs_f64(self.0)
    }
}

impl impeller2::component::Component for SimulationTimeStep {
    const NAME: &'static str = "simulation_time_step";
    const ASSET: bool = false;

    fn schema() -> impeller2::schema::Schema<Vec<u64>> {
        impeller2::schema::Schema::new(impeller2::types::PrimType::F64, [0usize; 0])
            .expect("failed to create schema")
    }
}

#[cfg(feature = "nox")]
impl impeller2::com_de::Decomponentize for SimulationTimeStep {
    type Error = core::convert::Infallible;
    fn apply_value(
        &mut self,
        component_id: impeller2::types::ComponentId,
        value: impeller2::types::ComponentView<'_>,
        _timestamp: Option<Timestamp>,
    ) -> Result<(), Self::Error> {
        if component_id != SimulationTimeStep::COMPONENT_ID {
            return Ok(());
        }
        let impeller2::types::ComponentView::F64(view) = value else {
            return Ok(());
        };
        let buf = view.buf();
        self.0 = buf[0];
        Ok(())
    }
}

#[derive(Clone, Serialize, Deserialize, Copy)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Resource))]
pub struct LastUpdated(pub Timestamp);

#[derive(Clone, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Resource))]
pub struct IsRecording(pub bool);

#[cfg(feature = "nox")]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct WorldPos {
    pub att: nox::Quaternion<f64, nox::ArrayRepr>,
    pub pos: nox::Vector3<f64, nox::ArrayRepr>,
}

#[cfg(feature = "nox")]
impl Component for WorldPos {
    const NAME: &'static str = "world_pos";
    const ASSET: bool = false;

    #[cfg(feature = "std")]
    fn schema() -> impeller2::schema::Schema<Vec<u64>> {
        impeller2::schema::Schema::new(impeller2::types::PrimType::F64, [7usize])
            .expect("failed to create schema")
    }
}

#[cfg(feature = "nox")]
impl impeller2::com_de::Decomponentize for WorldPos {
    type Error = core::convert::Infallible;

    fn apply_value(
        &mut self,
        component_id: impeller2::types::ComponentId,
        value: impeller2::types::ComponentView<'_>,
        _timestamp: Option<Timestamp>,
    ) -> Result<(), Self::Error> {
        if component_id != WorldPos::COMPONENT_ID {
            return Ok(());
        }
        let impeller2::types::ComponentView::F64(view) = value else {
            return Ok(());
        };
        let buf = view.buf();
        let att: [f64; 4] = buf[..4].try_into().expect("slice size wrong");
        self.att = nox::Quaternion(nox::Tensor::from_buf(att));
        let pos: [f64; 3] = buf[4..].try_into().expect("slice size wrong");
        self.pos = nox::Tensor::from_buf(pos);
        Ok(())
    }
}

#[derive(Clone, Serialize, Deserialize, Copy)]
#[cfg_attr(
    feature = "bevy",
    derive(bevy::prelude::Resource, bevy::prelude::Component)
)]
pub struct CurrentTimestamp(pub Timestamp);

impl Default for CurrentTimestamp {
    fn default() -> Self {
        Self(Timestamp::EPOCH)
    }
}

impl impeller2::component::Component for CurrentTimestamp {
    const NAME: &'static str = "current_timestamp";

    fn schema() -> impeller2::schema::Schema<Vec<u64>> {
        impeller2::schema::Schema::new(impeller2::types::PrimType::I64, [1usize])
            .expect("failed to create schema")
    }
}

#[cfg(feature = "nox")]
impl impeller2::com_de::Decomponentize for CurrentTimestamp {
    type Error = core::convert::Infallible;

    fn apply_value(
        &mut self,
        component_id: impeller2::types::ComponentId,
        value: impeller2::types::ComponentView<'_>,
        _timestamp: Option<Timestamp>,
    ) -> Result<(), Self::Error> {
        if component_id != CurrentTimestamp::COMPONENT_ID {
            return Ok(());
        }
        let impeller2::types::ComponentView::I64(view) = value else {
            return Ok(());
        };
        let buf = view.buf();
        self.0 = Timestamp(buf[0]);
        Ok(())
    }
}
