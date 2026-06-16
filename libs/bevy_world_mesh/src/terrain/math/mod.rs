mod coordinate;
mod ellipsoid;
mod terrain_model;

pub use crate::terrain::math::{
    coordinate::{Coordinate, TileCoordinate},
    terrain_model::{
        generate_terrain_model_approximation, TerrainModel, TerrainModelApproximation,
    },
};

/// The square of the parameter c of the algebraic sigmoid function, used to convert between uv and st coordinates.
const C_SQR: f64 = 0.87 * 0.87;
