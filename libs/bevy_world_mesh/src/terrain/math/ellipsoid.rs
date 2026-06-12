use bevy::math::{DVec2, DVec3, Vec3Swizzles};
use std::cmp::Ordering;

// Adapted from https://www.geometrictools.com/Documentation/DistancePointEllipseEllipsoid.pdf
// Original licensed under Creative Commons Attribution 4.0 International License
// http://creativecommons.org/licenses/by/4.0/

// After 1074 iterations, s0 == s1 == s
// This should probably be relaxed to only limit the error s1-s0 to a constant e.
const MAX_ITERATIONS: usize = 1074;

pub fn project_point_ellipsoid(e: DVec3, y: DVec3) -> DVec3 {
    let sign = y.signum();
    let y = y.xzy().abs();

    let x = if y.z > 0.0 {
        if y.y > 0.0 {
            if y.x > 0.0 {
                let z = y / e;
                let g = z.length_squared() - 1.0;

                if g != 0.0 {
                    let r = DVec3::new((e.x * e.x) / (e.z * e.z), (e.y * e.y) / (e.z * e.z), 1.0);

                    r * y / (get_root_3d(r, z, g) + r)
                } else {
                    y
                }
            } else {
                project_point_ellipse(e.yz(), y.yz()).extend(0.0).zxy()
            }
        } else if y.x > 0.0 {
            project_point_ellipse(e.xz(), y.xz()).extend(0.0).xzy()
        } else {
            DVec3::new(0.0, 0.0, e.z)
        }
    } else {
        let denom0 = e.x * e.x - e.z * e.z;
        let denom1 = e.y * e.y - e.z * e.z;
        let numer0 = e.x * y.x;
        let numer1 = e.y * y.y;

        let mut x = None;

        if numer0 < denom0 && numer1 < denom1 {
            let xde0 = numer0 / denom0;
            let xde1 = numer1 / denom1;
            let xde0sqr = xde0 * xde0;
            let xde1sqr = xde1 * xde1;
            let discr = 1.0 - xde0sqr - xde1sqr;

            if discr > 0.0 {
                x = Some(e * DVec3::new(xde0, xde1, discr.sqrt()));
            }
        }

        x.unwrap_or_else(|| project_point_ellipse(e.xy(), y.xy()).extend(0.0))
    };

    sign * x.xzy()
}

fn project_point_ellipse(e: DVec2, y: DVec2) -> DVec2 {
    if y.y > 0.0 {
        if y.x > 0.0 {
            let z = y / e;
            let g = z.length_squared() - 1.0;

            if g != 0.0 {
                let r = DVec2::new((e.x * e.x) / (e.y * e.y), 1.0);
                r * y / (get_root_2d(r, z, g) + r)
            } else {
                y
            }
        } else {
            DVec2::new(0.0, e.y)
        }
    } else {
        let numer0 = e.x * y.x;
        let denom0 = e.x * e.x - e.y * e.y;
        if numer0 < denom0 {
            let xde0 = numer0 / denom0;
            DVec2::new(e.x * xde0, e.y * (1.0 - xde0 * xde0).sqrt())
        } else {
            DVec2::new(e.x, 0.0)
        }
    }
}

fn get_root_3d(r: DVec3, z: DVec3, g: f64) -> f64 {
    let n = r * z;

    let mut s0 = z.z - 1.0;
    let mut s1 = if g < 0.0 { 0.0 } else { n.length() - 1.0 };
    let mut s = 0.0;

    for _ in 0..MAX_ITERATIONS {
        s = (s0 + s1) / 2.0;
        if s == s0 || s == s1 {
            break;
        }

        let ratio = n / (s + r);
        let g = ratio.length_squared() - 1.0;

        match g.total_cmp(&0.0) {
            Ordering::Less => s1 = s,
            Ordering::Equal => break,
            Ordering::Greater => s0 = s,
        }
    }

    s
}

fn get_root_2d(r: DVec2, z: DVec2, g: f64) -> f64 {
    let n = r * z;

    let mut s0 = z.y - 1.0;
    let mut s1 = if g < 0.0 { 0.0 } else { n.length() - 1.0 };
    let mut s = 0.0;

    for _ in 0..MAX_ITERATIONS {
        s = (s0 + s1) / 2.0;
        if s == s0 || s == s1 {
            break;
        }

        let ratio = n / (s + r);
        let g = ratio.length_squared() - 1.0;

        match g.total_cmp(&0.0) {
            Ordering::Less => s1 = s,
            Ordering::Equal => break,
            Ordering::Greater => s0 = s,
        }
    }

    s
}
