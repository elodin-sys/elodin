use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Duration,
};

use miette::IntoDiagnostic;
use serde::Deserialize;

/// Sample a fixed WGS84 height grid from Cesium ion before running a simulation.
#[derive(clap::Args, Clone)]
pub struct Args {
    /// JSON grid request (asset_id, bounds_degrees, columns, rows).
    request: PathBuf,
    /// Output JSON file; must not already exist.
    output: PathBuf,
    #[arg(long, default_value_t = 600, value_parser = clap::value_parser!(u64).range(1..=3600))]
    timeout_seconds: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    asset_id: i64,
    // west, south, east, north, in degrees.
    bounds_degrees: [f64; 4],
    columns: usize,
    rows: usize,
}

impl Request {
    fn positions(&self) -> miette::Result<Vec<[f64; 2]>> {
        let [west, south, east, north] = self.bounds_degrees;
        if self.asset_id <= 0
            || !self.bounds_degrees.iter().all(|x| x.is_finite())
            || west < -180.0
            || east > 180.0
            || south < -90.0
            || north > 90.0
            || west >= east
            || south >= north
            || self.columns < 2
            || self.rows < 2
            || self
                .columns
                .checked_mul(self.rows)
                .is_none_or(|n| n > 1_000_000)
        {
            return Err(miette::miette!("Invalid terrain grid request"));
        }
        Ok((0..self.rows)
            .flat_map(|row| {
                (0..self.columns).map(move |column| {
                    [
                        west + (east - west) * column as f64 / (self.columns - 1) as f64,
                        south + (north - south) * row as f64 / (self.rows - 1) as f64,
                    ]
                })
            })
            .collect())
    }
}

pub fn run(args: Args) -> miette::Result<()> {
    let request: Request =
        serde_json::from_reader(File::open(args.request).into_diagnostic()?).into_diagnostic()?;
    let positions = request.positions()?;
    let token = std::env::var("CESIUM_ION_TOKEN")
        .map_err(|_| miette::miette!("Set CESIUM_ION_TOKEN to prepare terrain heights"))?;
    eprintln!(
        "Preparing {} terrain heights from Cesium ion…",
        positions.len()
    );
    let heights = elodin_editor::plugins::cesium::sample_terrain_heights(
        request.asset_id,
        &token,
        &positions,
        Duration::from_secs(args.timeout_seconds),
    )
    .into_diagnostic()?;
    let grid = serde_json::json!({
        "version": 1,
        "asset_id": request.asset_id,
        "ellipsoid": "WGS84",
        "bounds_degrees": request.bounds_degrees,
        "columns": request.columns,
        "rows": request.rows,
        "heights_m": heights,
    });
    let file = File::create_new(args.output).into_diagnostic()?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &grid).into_diagnostic()?;
    writer.flush().into_diagnostic()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grid_order_and_bounds_are_explicit() {
        let mut request = Request {
            asset_id: 1,
            bounds_degrees: [10.0, 20.0, 12.0, 22.0],
            columns: 3,
            rows: 2,
        };
        assert_eq!(
            request.positions().unwrap(),
            vec![
                [10.0, 20.0],
                [11.0, 20.0],
                [12.0, 20.0],
                [10.0, 22.0],
                [11.0, 22.0],
                [12.0, 22.0]
            ]
        );
        request.columns = usize::MAX;
        assert!(request.positions().is_err());
        request.columns = 1;
        assert!(request.positions().is_err());
    }
}
