pub mod tiff;

use crate::terrain::math::TileCoordinate;
use anyhow::Result;
use bincode::{config, Decode, Encode};
use std::{fs, path::Path};

#[derive(Encode, Decode, Debug)]
pub struct TC {
    pub tiles: Vec<TileCoordinate>,
}

impl TC {
    pub fn decode_alloc(encoded: &[u8]) -> Result<Self> {
        let config = config::standard();
        let decoded = bincode::decode_from_slice(encoded, config)?;
        Ok(decoded.0)
    }

    pub fn encode_alloc(&self) -> Result<Vec<u8>> {
        let config = config::standard();
        let encoded = bincode::encode_to_vec(self, config)?;
        Ok(encoded)
    }

    pub fn load_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let encoded = fs::read(path)?;
        Self::decode_alloc(&encoded)
    }

    pub fn save_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let encoded = self.encode_alloc()?;
        fs::write(path, encoded)?;
        Ok(())
    }
}
