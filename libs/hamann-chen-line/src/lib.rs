//! Hamann–Chen (1994) curvature-based polyline sampling.
//!
//! Implementation is a direct port of the algorithm in Shane Celis’s C#
//! [`PiecewiseLinearCurveApproximation.cs`](https://gist.github.com/shanecelis/2e0ffd790e31507fba04dd56f806667a)
//! (curvature samples, `xbars` / `ss` / `ki` filtering, interval walk, endpoints). Inverting
//! cumulative curvature along arc length uses trapezoidal integration on `s` and linear
//! interpolation instead of Math.NET `LinearSpline` + `RobustNewtonRaphson` from that gist.
//!
//! - **2D** — [`select_polyline2_indices`] for a planar polyline, and [`select_time_value_indices`]
//!   for `(t, y)` graph data (same algorithm in the `(t,y)` plane).
//! - **3D** — [`select_polyline3_indices`] for a spatial polyline (local osculating 2D frame per
//!   vertex, then the same 2D curvature construction as above on each triangle).

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
                let pick = if (s_t - sl).abs() <= (s_t - sl_1).abs() {
                    nearest_point_index2(points, xbars[l + 1])
                } else {
                    nearest_point_index2(points, xbars[l + 2])
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
                } else {
                    nearest_point_index3(points, xbars[l + 2])
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

/// **2D planar polyline** — indices into `points` (≤ `m`), always including endpoints.
#[inline]
pub fn select_polyline2_indices(points: &[Vec2], m: usize) -> Vec<usize> {
    let ks = curvature_samples_polyline2(points);
    select_indices_curvature2(points, &ks, m)
}

/// **2D graph / time series** — `(t, y)` rows as a polyline in the `(t,y)` plane.
#[inline]
pub fn select_time_value_indices(times: &[f32], values: &[f32], m: usize) -> Vec<usize> {
    let n = times.len().min(values.len());
    if n == 0 {
        return Vec::new();
    }
    let pts: Vec<Vec2> = (0..n).map(|i| Vec2::new(times[i], values[i])).collect();
    select_polyline2_indices(&pts, m)
}

/// **3D spatial polyline** — indices into `points` (≤ `m`), always including endpoints.
#[inline]
pub fn select_polyline3_indices(points: &[Vec3], m: usize) -> Vec<usize> {
    let ks = curvature_samples_polyline3(points);
    select_indices_curvature3(points, &ks, m)
}

/// Joint downsampling of `(t, x, y, z)` using one index set from `(t, ‖p‖)` (fast, not full 3D shape).
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

/// Alias for [`select_polyline2_indices`] (legacy name).
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
}
