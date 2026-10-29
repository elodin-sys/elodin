use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Duration,
};

use bevy_cesium::{CartographicPosition, TerrainSampler};
use miette::IntoDiagnostic;
use serde::{Deserialize, Serialize};

const MAX_GRID_POINTS: usize = 25_000_000;
const BATCH_POINTS: usize = 1_000_000;

/// Sample a fixed WGS84 height grid from Cesium ion before running a simulation.
#[derive(clap::Args, Clone)]
pub struct Args {
    /// JSON grid request (asset_id, bounds_degrees, columns, rows).
    request: PathBuf,
    /// Output JSON file; must not already exist.
    output: PathBuf,
    /// Timeout per native batch (up to one million positions).
    #[arg(long, default_value_t = 600, value_parser = clap::value_parser!(u64).range(1..=3600))]
    timeout_seconds: u64,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Request {
    asset_id: i64,
    // west, south, east, north, in degrees.
    bounds_degrees: [f64; 4],
    columns: usize,
    rows: usize,
}

impl Request {
    fn point_count(&self) -> miette::Result<usize> {
        let [west, south, east, north] = self.bounds_degrees;
        let count = self.columns.checked_mul(self.rows);
        if self.asset_id <= 0
            || !self.bounds_degrees.iter().all(|x| x.is_finite())
            || west < -180.0
            || east > 180.0
            || south <= -90.0
            || north >= 90.0
            || west >= east
            || south >= north
            || self.columns < 2
            || self.rows < 2
            || count.is_none_or(|n| n > MAX_GRID_POINTS)
        {
            return Err(miette::miette!("Invalid terrain grid request"));
        }
        Ok(count.unwrap())
    }

    fn sample(
        &self,
        mut query: impl FnMut(&[CartographicPosition]) -> miette::Result<Vec<f64>>,
    ) -> miette::Result<Vec<f64>> {
        let count = self.point_count()?;
        let [west, south, east, north] = self.bounds_degrees;
        let mut heights = Vec::with_capacity(count);
        for start in (0..count).step_by(BATCH_POINTS) {
            let end = (start + BATCH_POINTS).min(count);
            // Index the full grid even when a batch ends in the middle of a row.
            let positions: Vec<_> = (start..end)
                .map(|index| CartographicPosition {
                    longitude_radians: (west
                        + (east - west) * (index % self.columns) as f64
                            / (self.columns - 1) as f64)
                        .to_radians(),
                    latitude_radians: (south
                        + (north - south) * (index / self.columns) as f64 / (self.rows - 1) as f64)
                        .to_radians(),
                })
                .collect();
            heights.extend(query(&positions)?);
        }
        Ok(heights)
    }
}

#[derive(Serialize)]
struct Grid<'a> {
    version: u8,
    ellipsoid: &'static str,
    #[serde(flatten)]
    request: &'a Request,
    heights_m: &'a [f64],
}

pub fn run(args: Args) -> miette::Result<()> {
    let request: Request =
        serde_json::from_reader(File::open(args.request).into_diagnostic()?).into_diagnostic()?;
    let count = request.point_count()?;
    let token = std::env::var("CESIUM_ION_TOKEN")
        .map_err(|_| miette::miette!("Set CESIUM_ION_TOKEN to prepare terrain heights"))?;
    eprintln!("Preparing {count} terrain heights from Cesium ion…");
    // Reuse the tileset and downloaded tiles across all batches. A failed query
    // aborts the request; Python can retry with a fresh native session.
    let mut sampler = TerrainSampler::from_ion(request.asset_id, &token).into_diagnostic()?;
    let heights = request.sample(|positions| {
        sampler
            .sample_heights(positions, Duration::from_secs(args.timeout_seconds))
            .into_diagnostic()
    })?;
    let grid = Grid {
        version: 1,
        ellipsoid: "WGS84",
        request: &request,
        heights_m: &heights,
    };
    let file = File::create_new(args.output).into_diagnostic()?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &grid).into_diagnostic()?;
    writer.flush().into_diagnostic()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(columns: usize, rows: usize) -> Request {
        Request {
            asset_id: 1,
            bounds_degrees: [10.0, 20.0, 12.0, 22.0],
            columns,
            rows,
        }
    }

    #[test]
    fn grid_order_and_bounds_are_explicit() {
        request(3, 2)
            .sample(|positions| {
                let expected = [
                    [10.0_f64, 20.0_f64],
                    [11.0, 20.0],
                    [12.0, 20.0],
                    [10.0, 22.0],
                    [11.0, 22.0],
                    [12.0, 22.0],
                ];
                assert_eq!(positions.len(), expected.len());
                for (position, [lon, lat]) in positions.iter().zip(expected) {
                    assert_eq!(position.longitude_radians, lon.to_radians());
                    assert_eq!(position.latitude_radians, lat.to_radians());
                }
                Ok(vec![100.0; positions.len()])
            })
            .unwrap();
    }

    #[test]
    fn batches_preserve_every_point_including_partial_rows() {
        let mut visited = 0;
        let mut batches = 0;
        let heights = request(1001, 1001)
            .sample(|positions| {
                batches += 1;
                assert!(positions.len() <= BATCH_POINTS);
                let mut result = Vec::with_capacity(positions.len());
                for p in positions {
                    let column = ((p.longitude_radians.to_degrees() - 10.0) * 500.0).round();
                    let row = ((p.latitude_radians.to_degrees() - 20.0) * 500.0).round();
                    let index = row as usize * 1001 + column as usize;
                    assert_eq!(index, visited);
                    result.push(index as f64);
                    visited += 1;
                }
                Ok(result)
            })
            .unwrap();
        assert_eq!(batches, 2);
        assert_eq!(heights.len(), 1001 * 1001);
        assert!(heights.iter().enumerate().all(|(i, h)| *h == i as f64));
    }

    #[test]
    fn failed_batch_aborts_the_grid() {
        let mut batches = 0;
        assert!(
            request(1001, 1001)
                .sample(|positions| {
                    batches += 1;
                    if batches == 2 {
                        Err(miette::miette!("Missing terrain height"))
                    } else {
                        Ok(vec![100.0; positions.len()])
                    }
                })
                .is_err()
        );
        assert_eq!(batches, 2);
    }

    #[test]
    fn invalid_geometry_fails_before_querying() {
        for request in [request(usize::MAX, 2), request(1, 2), request(5001, 5000)] {
            assert!(request.sample(|_| panic!("must not query")).is_err());
        }
        for bounds in [[10.0, -90.0, 12.0, 22.0], [10.0, 20.0, 12.0, 90.0]] {
            let mut request = request(2, 2);
            request.bounds_degrees = bounds;
            assert!(request.sample(|_| panic!("must not query")).is_err());
        }
    }
}
