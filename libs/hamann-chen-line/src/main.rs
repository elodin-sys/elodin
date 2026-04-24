//! CLI to exercise Hamann–Chen simplification on CSV data.

use std::io::{self, BufRead, Write};

use clap::{Parser, ValueEnum};
use glam::{Vec2, Vec3};
use hamann_chen_line::{
    select_polyline2_indices, select_polyline3_indices, select_time_value_indices,
    select_trajectory_time_norm_indices,
};

#[derive(Copy, Clone, Debug, ValueEnum)]
#[clap(rename_all = "kebab-case")]
enum Kind {
    /// Columns: `x,y` — planar 2D polyline
    Polyline2,
    /// Columns: `x,y,z` — 3D spatial polyline
    Polyline3,
    /// Columns: `t,y` — time series (polyline in the `(t,y)` plane)
    TimeValue,
    /// Columns: `t,x,y,z` — one index set from `(t, ‖p‖)` (aligned axes, not full 3D curvature)
    TrajectoryTimeNorm,
}

#[derive(Parser, Debug)]
#[command(
    name = "hamann-chen-line",
    about = "Hamann–Chen polyline simplification (CSV in/out)"
)]
struct Args {
    /// How many vertices to keep (≥ 2)
    #[arg(short = 'n', long, default_value_t = 100)]
    target: usize,

    /// Row format (comma-separated floats, no header)
    #[arg(long, value_enum, default_value = "polyline2")]
    kind: Kind,

    /// Input file (`-` = stdin)
    #[arg(short, long, default_value = "-")]
    input: String,

    /// Output file (`-` = stdout)
    #[arg(short, long, default_value = "-")]
    output: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.target < 2 {
        anyhow::bail!("target must be at least 2");
    }

    let rows = read_csv_rows(&args.input)?;
    if rows.is_empty() {
        anyhow::bail!("no data rows");
    }

    let idx = match args.kind {
        Kind::Polyline2 => {
            let pts: Vec<Vec2> = rows
                .iter()
                .filter_map(|r| {
                    if r.len() >= 2 {
                        Some(Vec2::new(r[0], r[1]))
                    } else {
                        None
                    }
                })
                .collect();
            if pts.len() < 2 {
                anyhow::bail!("need at least two valid x,y rows");
            }
            select_polyline2_indices(&pts, args.target)
        }
        Kind::Polyline3 => {
            let pts: Vec<Vec3> = rows
                .iter()
                .filter_map(|r| {
                    if r.len() >= 3 {
                        Some(Vec3::new(r[0], r[1], r[2]))
                    } else {
                        None
                    }
                })
                .collect();
            if pts.len() < 2 {
                anyhow::bail!("need at least two valid x,y,z rows");
            }
            select_polyline3_indices(&pts, args.target)
        }
        Kind::TimeValue => {
            if rows.iter().any(|r| r.len() < 2) {
                anyhow::bail!("time-value rows need at least two columns: t,y");
            }
            let t: Vec<f32> = rows.iter().map(|r| r[0]).collect();
            let y: Vec<f32> = rows.iter().map(|r| r[1]).collect();
            select_time_value_indices(&t, &y, args.target)
        }
        Kind::TrajectoryTimeNorm => {
            if rows.iter().any(|r| r.len() < 4) {
                anyhow::bail!("trajectory rows need four columns: t,x,y,z");
            }
            let t: Vec<f32> = rows.iter().map(|r| r[0]).collect();
            let p: Vec<Vec3> = rows.iter().map(|r| Vec3::new(r[1], r[2], r[3])).collect();
            select_trajectory_time_norm_indices(&t, &p, args.target)
        }
    };

    let out_rows: Vec<Vec<f32>> = idx.iter().map(|&i| rows[i].clone()).collect();
    write_csv_rows(&args.output, &out_rows)?;
    Ok(())
}

fn read_csv_rows(path: &str) -> anyhow::Result<Vec<Vec<f32>>> {
    let reader: Box<dyn BufRead> = if path == "-" {
        Box::new(io::BufReader::new(io::stdin()))
    } else {
        Box::new(io::BufReader::new(
            std::fs::File::open(path).map_err(|e| anyhow::anyhow!("open {path}: {e}"))?,
        ))
    };

    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let row: Vec<f32> = line
            .split(',')
            .map(|s| s.trim().parse::<f32>())
            .collect::<Result<_, _>>()
            .map_err(|e| anyhow::anyhow!("parse float in {line:?}: {e}"))?;
        out.push(row);
    }
    Ok(out)
}

fn write_csv_rows(path: &str, rows: &[Vec<f32>]) -> anyhow::Result<()> {
    let mut w: Box<dyn Write> = if path == "-" {
        Box::new(io::stdout())
    } else {
        Box::new(std::fs::File::create(path).map_err(|e| anyhow::anyhow!("create {path}: {e}"))?)
    };
    for row in rows {
        for (i, v) in row.iter().enumerate() {
            if i > 0 {
                write!(w, ",")?;
            }
            write!(w, "{v}")?;
        }
        writeln!(w)?;
    }
    Ok(())
}
