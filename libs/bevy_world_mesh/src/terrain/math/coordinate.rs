use crate::terrain::math::{TerrainModel, C_SQR};
use bevy::{
    math::{DVec2, DVec3, IVec2},
    render::render_resource::ShaderType,
};
use bincode::{Decode, Encode};
use std::fmt;

const NEIGHBOURING_SIDES: [[u32; 5]; 6] = [
    [0, 4, 2, 1, 5],
    [1, 0, 2, 3, 5],
    [2, 0, 4, 3, 1],
    [3, 2, 4, 5, 1],
    [4, 2, 0, 5, 3],
    [5, 4, 0, 1, 3],
];

#[derive(Clone, Copy)]
enum SideInfo {
    Fixed0,
    Fixed1,
    PositiveS,
    PositiveT,
}

impl SideInfo {
    const EVEN_LIST: [[SideInfo; 2]; 6] = [
        [SideInfo::PositiveS, SideInfo::PositiveT],
        [SideInfo::Fixed0, SideInfo::PositiveT],
        [SideInfo::Fixed0, SideInfo::PositiveS],
        [SideInfo::PositiveT, SideInfo::PositiveS],
        [SideInfo::PositiveT, SideInfo::Fixed0],
        [SideInfo::PositiveS, SideInfo::Fixed0],
    ];
    const ODD_LIST: [[SideInfo; 2]; 6] = [
        [SideInfo::PositiveS, SideInfo::PositiveT],
        [SideInfo::PositiveS, SideInfo::Fixed1],
        [SideInfo::PositiveT, SideInfo::Fixed1],
        [SideInfo::PositiveT, SideInfo::PositiveS],
        [SideInfo::Fixed1, SideInfo::PositiveS],
        [SideInfo::Fixed1, SideInfo::PositiveT],
    ];

    fn project_to_side(side: u32, other_side: u32) -> [SideInfo; 2] {
        let index = ((6 + other_side - side) % 6) as usize;

        if side.is_multiple_of(2) {
            SideInfo::EVEN_LIST[index]
        } else {
            SideInfo::ODD_LIST[index]
        }
    }
}

/// Describes a location on the unit cube sphere.
/// The side index refers to one of the six cube faces and the uv coordinate describes the location within this side.
#[derive(Copy, Clone, Debug, Default)]
pub struct Coordinate {
    pub side: u32,
    pub uv: DVec2,
}

impl Coordinate {
    pub fn new(side: u32, uv: DVec2) -> Self {
        Self { side, uv }
    }

    /// Calculates the coordinate for for the local position on the unit cube sphere.
    pub(crate) fn from_world_position(world_position: DVec3, model: &TerrainModel) -> Self {
        let local_position = model.position_world_to_local(world_position);

        let (side, uv) = if model.is_spherical() {
            let normal = local_position;
            let abs_normal = normal.abs();

            let (side, uv) = if abs_normal.x > abs_normal.y && abs_normal.x > abs_normal.z {
                if normal.x < 0.0 {
                    (0, DVec2::new(-normal.z / normal.x, normal.y / normal.x))
                } else {
                    (3, DVec2::new(-normal.y / normal.x, normal.z / normal.x))
                }
            } else if abs_normal.z > abs_normal.y {
                if normal.z > 0.0 {
                    (1, DVec2::new(normal.x / normal.z, -normal.y / normal.z))
                } else {
                    (4, DVec2::new(normal.y / normal.z, -normal.x / normal.z))
                }
            } else if normal.y > 0.0 {
                (2, DVec2::new(normal.x / normal.y, normal.z / normal.y))
            } else {
                (5, DVec2::new(-normal.z / normal.y, -normal.x / normal.y))
            };

            let w = uv * ((1.0 + C_SQR) / (1.0 + C_SQR * uv * uv)).powf(0.5);
            let uv = 0.5 * w + 0.5;

            (side, uv)
        } else {
            let uv = DVec2::new(local_position.x + 0.5, local_position.z + 0.5)
                .clamp(DVec2::ZERO, DVec2::ONE);

            (0, uv)
        };

        Self { side, uv }
    }

    pub(crate) fn world_position(self, model: &TerrainModel, height: f32) -> DVec3 {
        let local_position = if model.is_spherical() {
            let w = (self.uv - 0.5) / 0.5;
            let uv = w / (1.0 + C_SQR - C_SQR * w * w).powf(0.5);

            match self.side {
                0 => DVec3::new(-1.0, -uv.y, uv.x),
                1 => DVec3::new(uv.x, -uv.y, 1.0),
                2 => DVec3::new(uv.x, 1.0, uv.y),
                3 => DVec3::new(1.0, -uv.x, uv.y),
                4 => DVec3::new(uv.y, -uv.x, -1.0),
                5 => DVec3::new(uv.y, -1.0, uv.x),
                _ => unreachable!(),
            }
            .normalize()
        } else {
            DVec3::new(self.uv.x - 0.5, 0.0, self.uv.y - 0.5)
        };

        model.position_local_to_world(local_position, height as f64)
    }

    /// Projects the coordinate onto one of the six cube faces.
    /// Thereby it chooses the closest location on this face to the original coordinate.
    pub(crate) fn project_to_side(self, side: u32, model: &TerrainModel) -> Self {
        if model.is_spherical() {
            let info = SideInfo::project_to_side(self.side, side);

            let uv = info
                .map(|info| match info {
                    SideInfo::Fixed0 => 0.0,
                    SideInfo::Fixed1 => 1.0,
                    SideInfo::PositiveS => self.uv.x,
                    SideInfo::PositiveT => self.uv.y,
                })
                .into();

            Self { side, uv }
        } else {
            self
        }
    }
}

/// The global coordinate and identifier of a tile.
#[derive(Copy, Clone, Default, Debug, Hash, Eq, PartialEq, ShaderType, Encode, Decode)]
pub struct TileCoordinate {
    /// The side of the cube sphere the tile is located on.
    pub side: u32,
    /// The lod of the tile, where 0 is the highest level of detail with the smallest size
    /// and highest resolution
    pub lod: u32,
    /// The x position of the tile in tile sizes.
    pub x: u32,
    /// The y position of the tile in tile sizes.
    pub y: u32,
}

impl TileCoordinate {
    pub const INVALID: TileCoordinate = TileCoordinate {
        side: u32::MAX,
        lod: u32::MAX,
        x: u32::MAX,
        y: u32::MAX,
    };

    pub fn new(side: u32, lod: u32, x: u32, y: u32) -> Self {
        Self { side, lod, x, y }
    }

    pub fn count(lod: u32) -> u32 {
        1 << lod
    }

    pub fn path(self, path: &str, extension: &str) -> String {
        format!("{path}/{self}.{extension}")
    }

    pub fn parent(self) -> Self {
        Self {
            side: self.side,
            lod: self.lod.wrapping_sub(1),
            x: self.x >> 1,
            y: self.y >> 1,
        }
    }

    pub fn children(self) -> impl Iterator<Item = Self> {
        (0..4).map(move |index| {
            TileCoordinate::new(
                self.side,
                self.lod + 1,
                (self.x << 1) + index % 2,
                (self.y << 1) + index / 2,
            )
        })
    }

    pub fn neighbours(self, spherical: bool) -> impl Iterator<Item = Self> {
        const OFFSETS: [IVec2; 8] = [
            IVec2::new(0, -1),
            IVec2::new(1, 0),
            IVec2::new(0, 1),
            IVec2::new(-1, 0),
            IVec2::new(-1, -1),
            IVec2::new(1, -1),
            IVec2::new(1, 1),
            IVec2::new(-1, 1),
        ];

        OFFSETS.iter().map(move |&offset| {
            let neighbour_position = IVec2::new(self.x as i32, self.y as i32) + offset;

            self.neighbour_coordinate(neighbour_position, spherical)
        })
    }

    fn neighbour_coordinate(self, neighbour_position: IVec2, spherical: bool) -> Self {
        let tile_count = Self::count(self.lod) as i32;

        if spherical {
            let edge_index = match neighbour_position {
                IVec2 { x, y } if (x < 0 || x >= tile_count) && (y < 0 || y >= tile_count) => {
                    return Self::INVALID;
                }
                IVec2 { x, .. } if x < 0 => 1,
                IVec2 { y, .. } if y < 0 => 2,
                IVec2 { x, .. } if x >= tile_count => 3,
                IVec2 { y, .. } if y >= tile_count => 4,
                _ => 0,
            };

            let neighbour_position = neighbour_position
                .clamp(IVec2::ZERO, IVec2::splat(tile_count - 1))
                .as_uvec2();

            let neighbour_side = NEIGHBOURING_SIDES[self.side as usize][edge_index];

            let info = SideInfo::project_to_side(self.side, neighbour_side);

            let [x, y] = info.map(|info| match info {
                SideInfo::Fixed0 => 0,
                SideInfo::Fixed1 => tile_count as u32 - 1,
                SideInfo::PositiveS => neighbour_position.x,
                SideInfo::PositiveT => neighbour_position.y,
            });

            Self::new(neighbour_side, self.lod, x, y)
        } else if neighbour_position.x < 0
            || neighbour_position.y < 0
            || neighbour_position.x >= tile_count
            || neighbour_position.y >= tile_count
        {
            Self::INVALID
        } else {
            Self::new(
                self.side,
                self.lod,
                neighbour_position.x as u32,
                neighbour_position.y as u32,
            )
        }
    }
}

impl fmt::Display for TileCoordinate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{}_{}_{}_{}", self.side, self.lod, self.x, self.y)
    }
}
