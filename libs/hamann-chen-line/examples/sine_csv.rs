//! Generate `sin(x)` on `[0, π]` (100 samples), write CSVs, and Hamann–Chen downsample to 49 / 24 / 9 vertices.
//!
//! Output directory: `examples/sine_plot_out/` (under this crate). Then run gnuplot from the crate root:
//! `gnuplot examples/sine_hamann_plot.gnu`

use std::fs;
use std::path::PathBuf;

use hamann_chen_line::select_time_value_indices;

const N: usize = 100;

fn main() -> anyhow::Result<()> {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let out_dir = manifest_dir.join("examples/sine_plot_out");
    fs::create_dir_all(&out_dir)?;

    let mut t = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    for i in 0..N {
        let x = (i as f32) * 2.0 * std::f32::consts::PI / ((N - 1) as f32);
        t.push(x);
        y.push(x.sin());
    }

    write_csv(&out_dir.join("sine_full.csv"), &t, &y)?;

    for (m, name) in [
        (49, "sine_m49.csv"),
        (24, "sine_m24.csv"),
        (9, "sine_m9.csv"),
    ] {
        let idx = select_time_value_indices(&t, &y, m);
        let (tt, yy): (Vec<f32>, Vec<f32>) = idx.iter().map(|&i| (t[i], y[i])).unzip();
        write_csv(&out_dir.join(name), &tt, &yy)?;
        eprintln!("m={m}: {} vertices written to {name}", tt.len());
    }

    eprintln!("Wrote CSVs under {}", out_dir.display());
    Ok(())
}

fn write_csv(path: &std::path::Path, t: &[f32], y: &[f32]) -> anyhow::Result<()> {
    debug_assert_eq!(t.len(), y.len());
    let mut s = String::new();
    for i in 0..t.len() {
        s.push_str(&format!("{},{}\n", t[i], y[i]));
    }
    fs::write(path, s)?;
    Ok(())
}
