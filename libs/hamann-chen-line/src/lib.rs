//! # Hamann–Chen (1994) curvature-based polyline sampling
//!
//! This crate reduces a long polyline to about **`m` vertices** by **sampling where curvature
//! concentrates**, instead of uniform decimation in index space. It is **not** Douglas–Peucker.
//!
//! ## Reference implementation (C#)
//!
//! The control flow matches the algorithm laid out in Shane Celis’s C#
//! [`PiecewiseLinearCurveApproximation.cs`](https://gist.github.com/shanecelis/2e0ffd790e31507fba04dd56f806667a):
//! curvature along the polyline, filter low-curvature vertices (`xbars` / `ki`), build arc length
//! `ss`, integrate curvature, walk intervals, pick indices, sort/dedup/endpoints.
//!
//! ## Differences from that gist
//!
//! - **Inverting cumulative curvature** — the gist uses Math.NET `LinearSpline` and
//!   `RobustNewtonRaphson`. This port uses **trapezoidal integration** on `s` with respect to
//!   `ki`, stores cumulative curvature, then **linear interpolation in `s`** at each target
//!   fraction of total integrated curvature (same idea, no extra numerics crates).
//! - **3D** — [`select_polyline3_indices`] flattens each vertex neighborhood to a **planar
//!   triangle** `(p_{i-1}, p_i, p_{i+1})`, runs the **same 2D** curvature pipeline on that
//!   triangle, and maps picks back to indices in the original 3D vertex list.
//!
//! ## Public API (pick one)
//!
//! | Use case | Entry point |
//! |----------|-------------|
//! | Planar `(x, y)` path | [`select_polyline2_indices`] |
//! | Telemetry / graph `(t, y)` | [`select_time_value_indices`] |
//! | Spatial `(x, y, z)` path | [`select_polyline3_indices`] |
//! | One **shared** index set for aligned `(t, x, y, z)` (cheap, not full 3D shape) | [`select_trajectory_time_norm_indices`] |
//!
//! All return **sorted indices** into your slices (you copy data out yourself). Long-form docs,
//! CLI examples, and Elodin Editor integration live in **`README.md`** next to this crate’s
//! `Cargo.toml`.

use glam::{Vec2, Vec3};

fn approx_zero(x: f32) -> bool {
    x.abs() <= f32::EPSILON * 8.0
}

fn linear_solve(a: Vec2, b: Vec2, c: Vec2) -> Vec2 {
    let det = a.x * b.y - a.y * b.x;
    if !det.is_finite() || det.abs() < 1e-20 {
        return Vec2::ZERO;
    }
    Vec2::new((c.x * b.y - c.y * b.x) / det, (a.x * c.y - a.y * c.x) / det)
}

/// Hamann–Chen curvature samples for a 2D polyline (same length as `points`).
fn curvature_samples_polyline2(points: &[Vec2]) -> Vec<f32> {
    let n = points.len();
    if n < 3 {
        return vec![0.0; n];
    }
    let mut seq = Vec::with_capacity(n);
    for i in 1..n - 1 {
        let d1 = (points[i - 1] - points[i]).normalize_or_zero();
        let d2 = (points[i + 1] - points[i]).normalize_or_zero();
        let mut b2 = (d1 + d2).normalize_or_zero();
        let b1 = if !approx_zero(d1.length())
            && !approx_zero(d2.length())
            && !approx_zero(b2.length())
        {
            Vec2::new(b2.y, -b2.x)
        } else {
            let t = (points[i + 1] - points[i]).normalize_or_zero();
            b2 = Vec2::new(-t.y, t.x);
            t
        };
        let alpha = d1.dot(b1);
        let beta = d1.dot(b2);
        let gamma = d2.dot(b1);
        let delta = d2.dot(b2);
        let soln = linear_solve(
            Vec2::new(alpha, gamma),
            Vec2::new(alpha * alpha, gamma * gamma),
            Vec2::new(beta, delta),
        );
        let a1 = soln.x;
        let a2 = soln.y;
        if i == 1 {
            let e = a1 + 2.0 * a2 * alpha;
            let k0 = 2.0 * a2 / (1.0 + e * e).powf(1.5);
            seq.push(k0);
        }
        let ki = 2.0 * a2 / (1.0 + a1 * a1).powf(1.5);
        seq.push(ki);
        if i == n - 2 {
            let e = a1 + 2.0 * a2 * beta;
            let kn = 2.0 * a2 / (1.0 + e * e).powf(1.5);
            seq.push(kn);
        }
    }
    debug_assert_eq!(seq.len(), n);
    seq
}

/// Map `(p_{i-1}, p_i, p_{i+1})` to 2D coordinates with `p_i` at the origin and `p_{i-1}` on −x.
fn triangle_to_local_xy(p_im1: Vec3, p_i: Vec3, p_ip1: Vec3) -> [Vec2; 3] {
    let a3 = p_im1 - p_i;
    let w = p_ip1 - p_i;
    let u = a3.normalize_or_zero();
    let a = a3.length();
    let x = w.dot(u);
    let y_sq = (w.length_squared() - x * x).max(0.0);
    let y = y_sq.sqrt();
    [Vec2::new(-a, 0.0), Vec2::ZERO, Vec2::new(x, y)]
}

/// Per-vertex curvature for a 3D polyline (length `n`), via the planar triangle at each interior vertex.
fn curvature_samples_polyline3(points: &[Vec3]) -> Vec<f32> {
    let n = points.len();
    if n < 3 {
        return vec![0.0; n];
    }
    let mut ks = vec![0.0_f32; n];
    for i in 1..n - 1 {
        let tri = triangle_to_local_xy(points[i - 1], points[i], points[i + 1]);
        let k3 = curvature_samples_polyline2(&tri);
        ks[i] = if k3.len() >= 2 { k3[1] } else { 0.0 };
        if !ks[i].is_finite() {
            ks[i] = 0.0;
        }
    }
    ks[0] = ks[1];
    ks[n - 1] = ks[n - 2];
    ks
}

fn cumulative_integral_trapezoid(ss: &[f32], ki: &[f32]) -> Vec<f32> {
    debug_assert_eq!(ss.len(), ki.len());
    let mut cum = vec![0.0_f32; ss.len()];
    for i in 0..ss.len().saturating_sub(1) {
        let ds = ss[i + 1] - ss[i];
        let seg = ds * (ki[i] + ki[i + 1]) * 0.5;
        cum[i + 1] = cum[i] + seg;
    }
    cum
}

fn invert_cum_integral(ss: &[f32], cum: &[f32], target: f32) -> f32 {
    if ss.is_empty() {
        return 0.0;
    }
    if target <= cum[0] {
        return ss[0];
    }
    let last = cum.len() - 1;
    if target >= cum[last] {
        return ss[last];
    }
    let j = cum.partition_point(|&v| v < target);
    let j = j.min(last);
    let j0 = j.saturating_sub(1);
    let (c0, c1) = (cum[j0], cum[j]);
    let (s0, s1) = (ss[j0], ss[j]);
    if (c1 - c0).abs() < 1e-30 {
        return s0;
    }
    let t = ((target - c0) / (c1 - c0)).clamp(0.0, 1.0);
    s0 + t * (s1 - s0)
}

fn nearest_point_index2(points: &[Vec2], q: Vec2) -> usize {
    points
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let da = (*a - q).length_squared();
            let db = (*b - q).length_squared();
            da.total_cmp(&db)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn nearest_point_index3(points: &[Vec3], q: Vec3) -> usize {
    points
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let da = (*a - q).length_squared();
            let db = (*b - q).length_squared();
            da.total_cmp(&db)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn uniform_indices(n: usize, m: usize) -> Vec<usize> {
    if m < 2 {
        return vec![0];
    }
    let mut out = Vec::with_capacity(m);
    for j in 0..m {
        let idx = ((j as f64) * (n.saturating_sub(1) as f64) / ((m - 1) as f64)).round() as usize;
        out.push(idx.min(n - 1));
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn select_indices_curvature2(points: &[Vec2], ks: &[f32], m: usize) -> Vec<usize> {
    let n = points.len();
    debug_assert_eq!(ks.len(), n);
    if m < 2 || n <= 2 {
        return (0..n.min(m.max(1))).collect();
    }
    if n <= m {
        return (0..n).collect();
    }

    let xbars_ks: Vec<(Vec2, f32)> = points
        .iter()
        .copied()
        .zip(ks.iter().copied())
        .filter(|&(_, k)| !k.is_nan() && !approx_zero(k))
        .collect();

    if xbars_ks.len() < 2 {
        return uniform_indices(n, m);
    }

    let xbars: Vec<Vec2> = xbars_ks.iter().map(|&(p, _)| p).collect();
    let ki: Vec<f32> = xbars_ks.iter().map(|&(_, k)| k).collect();

    let si: Vec<f32> = xbars.windows(2).map(|w| (w[1] - w[0]).length()).collect();
    let mut ss = Vec::with_capacity(xbars.len());
    ss.push(0.0);
    for seg in &si {
        ss.push(*ss.last().unwrap_or(&0.0) + seg);
    }

    if ss.len() != ki.len() {
        return uniform_indices(n, m);
    }

    let cum_k = cumulative_integral_trapezoid(&ss, &ki);
    let k_total = *cum_k.last().unwrap_or(&0.0);
    if !k_total.is_finite() || k_total.abs() < 1e-30 {
        return uniform_indices(n, m);
    }

    let mut picked = Vec::with_capacity(m);
    picked.push(0);

    for j in 1..m - 1 {
        let target = (j as f32) * k_total / (m as f32);
        let s_t = invert_cum_integral(&ss, &cum_k, target);
        let mut l = 0usize;
        let mut found = false;
        while l + 1 < ss.len() {
            let sl = ss[l];
            let sl_1 = ss[l + 1];
            if s_t > sl && s_t < sl_1 {
                // `ss` and `xbars` have the same length; the last valid interval has `l == xbars.len() - 2`,
                // so `l + 2 == xbars.len()` would be out of bounds — clamp to `l + 1` on that edge.
                let pick = if (s_t - sl).abs() <= (s_t - sl_1).abs() {
                    nearest_point_index2(points, xbars[l + 1])
                } else if l + 2 < xbars.len() {
                    nearest_point_index2(points, xbars[l + 2])
                } else {
                    nearest_point_index2(points, xbars[l + 1])
                };
                picked.push(pick);
                found = true;
                break;
            }
            l += 1;
        }
        if !found {
            picked.push(nearest_point_index2(
                points,
                xbars[xbars.len().saturating_sub(1)],
            ));
        }
    }

    picked.push(n - 1);
    picked.sort_unstable();
    picked.dedup();
    picked
}

fn select_indices_curvature3(points: &[Vec3], ks: &[f32], m: usize) -> Vec<usize> {
    let n = points.len();
    debug_assert_eq!(ks.len(), n);
    if m < 2 || n <= 2 {
        return (0..n.min(m.max(1))).collect();
    }
    if n <= m {
        return (0..n).collect();
    }

    let xbars_ks: Vec<(Vec3, f32)> = points
        .iter()
        .copied()
        .zip(ks.iter().copied())
        .filter(|&(_, k)| !k.is_nan() && !approx_zero(k))
        .collect();

    if xbars_ks.len() < 2 {
        return uniform_indices(n, m);
    }

    let xbars: Vec<Vec3> = xbars_ks.iter().map(|&(p, _)| p).collect();
    let ki: Vec<f32> = xbars_ks.iter().map(|&(_, k)| k).collect();

    let si: Vec<f32> = xbars.windows(2).map(|w| (w[1] - w[0]).length()).collect();
    let mut ss = Vec::with_capacity(xbars.len());
    ss.push(0.0);
    for seg in &si {
        ss.push(*ss.last().unwrap_or(&0.0) + seg);
    }

    if ss.len() != ki.len() {
        return uniform_indices(n, m);
    }

    let cum_k = cumulative_integral_trapezoid(&ss, &ki);
    let k_total = *cum_k.last().unwrap_or(&0.0);
    if !k_total.is_finite() || k_total.abs() < 1e-30 {
        return uniform_indices(n, m);
    }

    let mut picked = Vec::with_capacity(m);
    picked.push(0);

    for j in 1..m - 1 {
        let target = (j as f32) * k_total / (m as f32);
        let s_t = invert_cum_integral(&ss, &cum_k, target);
        let mut l = 0usize;
        let mut found = false;
        while l + 1 < ss.len() {
            let sl = ss[l];
            let sl_1 = ss[l + 1];
            if s_t > sl && s_t < sl_1 {
                let pick = if (s_t - sl).abs() <= (s_t - sl_1).abs() {
                    nearest_point_index3(points, xbars[l + 1])
                } else if l + 2 < xbars.len() {
                    nearest_point_index3(points, xbars[l + 2])
                } else {
                    nearest_point_index3(points, xbars[l + 1])
                };
                picked.push(pick);
                found = true;
                break;
            }
            l += 1;
        }
        if !found {
            picked.push(nearest_point_index3(
                points,
                xbars[xbars.len().saturating_sub(1)],
            ));
        }
    }

    picked.push(n - 1);
    picked.sort_unstable();
    picked.dedup();
    picked
}

/// Simplifies a **planar** polyline (2D positions).
///
/// - **`m`** — desired vertex budget (after internal dedup, length is typically ≤ `m`).
/// - **Endpoints** — when `m ≥ 2` and `points.len() ≥ 2`, the first and last original indices are
///   kept when the generic picker succeeds.
/// - **Fallback** — if curvature is everywhere negligible or degenerate, falls back to **uniform**
///   index spacing (see tests).
#[inline]
pub fn select_polyline2_indices(points: &[Vec2], m: usize) -> Vec<usize> {
    let ks = curvature_samples_polyline2(points);
    select_indices_curvature2(points, &ks, m)
}

/// Same as [`select_polyline2_indices`] on points `(t_i, y_i)` built by zipping `times` and
/// `values`.
///
/// Only `min(times.len(), values.len())` samples are considered. Empty input returns an empty
/// vector. Use this for **telemetry graphs** where the visual is the curve in **time–value**
/// space, not distance along the index axis.
#[inline]
pub fn select_time_value_indices(times: &[f32], values: &[f32], m: usize) -> Vec<usize> {
    let n = times.len().min(values.len());
    if n == 0 {
        return Vec::new();
    }
    let pts: Vec<Vec2> = (0..n).map(|i| Vec2::new(times[i], values[i])).collect();
    select_polyline2_indices(&pts, m)
}

/// Simplifies a **spatial** polyline in 3D using per-vertex **local 2D** curvature
/// (triangle at each interior vertex), then the same integrated-curvature sampler as 2D.
///
/// Prefer this when the **geometry in 3D** matters (e.g. a flight path). For **three synchronized
/// scalar streams** (X/Y/Z vs time) where you need **one index set** but a lighter model is OK, see
/// [`select_trajectory_time_norm_indices`].
#[inline]
pub fn select_polyline3_indices(points: &[Vec3], m: usize) -> Vec<usize> {
    let ks = curvature_samples_polyline3(points);
    select_indices_curvature3(points, &ks, m)
}

/// **Joint** simplification: builds a 2D polyline `(t_i, ‖p_i‖)` and returns indices from
/// [`select_polyline2_indices`].
///
/// Guarantees **one shared index list** for time-aligned `x/y/z` (copy each component by the same
/// indices). Does **not** use full 3D turning curvature; the reduced series may miss detail that
/// [`select_polyline3_indices`] would keep. Useful when speed and **axis alignment** matter more
/// than exact 3D shape fidelity.
pub fn select_trajectory_time_norm_indices(times: &[f32], pos: &[Vec3], m: usize) -> Vec<usize> {
    let n = times.len().min(pos.len());
    if n == 0 {
        return Vec::new();
    }
    let pts: Vec<Vec2> = (0..n)
        .map(|i| Vec2::new(times[i], pos[i].length()))
        .collect();
    select_polyline2_indices(&pts, m)
}

/// Alias for [`select_polyline2_indices`] (legacy name; “points” meant `Vec2` vertices).
#[inline]
pub fn select_point_indices(points: &[Vec2], m: usize) -> Vec<usize> {
    select_polyline2_indices(points, m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_preserved_polyline2() {
        let pts: Vec<Vec2> = (0..100)
            .map(|i| Vec2::new(i as f32, (i as f32) * 0.01))
            .collect();
        let idx = select_polyline2_indices(&pts, 8);
        assert_eq!(idx.first().copied(), Some(0));
        assert_eq!(idx.last().copied(), Some(99));
        assert!(idx.len() <= 8);
    }

    #[test]
    fn endpoints_preserved_polyline3() {
        let pts: Vec<Vec3> = (0..50)
            .map(|i| {
                let t = i as f32;
                Vec3::new(t.cos(), t.sin(), t * 0.05)
            })
            .collect();
        let idx = select_polyline3_indices(&pts, 12);
        assert_eq!(idx.first().copied(), Some(0));
        assert_eq!(idx.last().copied(), Some(49));
        assert!(idx.len() <= 12);
    }

    #[test]
    fn downsample_indices_stay_in_bounds() {
        let pts2: Vec<Vec2> = (0..40)
            .map(|i| {
                let t = i as f32;
                Vec2::new(t, t.sin() * 10.0)
            })
            .collect();
        for m in 3..=18 {
            let idx = select_polyline2_indices(&pts2, m);
            assert!(idx.len() >= 2);
            assert!(idx.iter().all(|&i| i < pts2.len()));
        }

        let pts3: Vec<Vec3> = (0..35)
            .map(|i| {
                let t = i as f32;
                Vec3::new(t.cos() * 2.0, t.sin() * 2.0, t * 0.1)
            })
            .collect();
        for m in 3..=16 {
            let idx = select_polyline3_indices(&pts3, m);
            assert!(idx.len() >= 2);
            assert!(idx.iter().all(|&i| i < pts3.len()));
        }
    }
}
