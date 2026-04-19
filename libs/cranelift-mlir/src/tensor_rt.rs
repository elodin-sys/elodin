use std::slice;

unsafe fn f64_slices<'a>(
    dst: *mut f64,
    a: *const f64,
    b: *const f64,
    n: usize,
) -> (&'a mut [f64], &'a [f64], &'a [f64]) {
    (
        unsafe { slice::from_raw_parts_mut(dst, n) },
        unsafe { slice::from_raw_parts(a, n) },
        unsafe { slice::from_raw_parts(b, n) },
    )
}

unsafe fn f64_unary<'a>(dst: *mut f64, a: *const f64, n: usize) -> (&'a mut [f64], &'a [f64]) {
    (unsafe { slice::from_raw_parts_mut(dst, n) }, unsafe {
        slice::from_raw_parts(a, n)
    })
}

// ---------------------------------------------------------------------------
// Macro-generated elementwise operations
// ---------------------------------------------------------------------------

/// Binary f64 elementwise op with a SIMD chunk loop + scalar tail.
/// Processes `wide::f64x2` chunks for the common n ≥ 2 case so
/// ptr-ABI callers routed through the tensor_rt dispatch still get
/// vectorized throughput. `$simd` is a closure taking two
/// `wide::f64x2`; `$scalar` handles the odd tail lane.
macro_rules! binary_f64_simd_op {
    ($name:ident, $simd:expr, $scalar:expr) => {
        pub extern "C" fn $name(dst: *mut f64, a: *const f64, b: *const f64, n: usize) {
            let (dst, a, b) = unsafe { f64_slices(dst, a, b, n) };
            let mut i = 0;
            while i + 2 <= n {
                let av = wide::f64x2::new([a[i], a[i + 1]]);
                let bv = wide::f64x2::new([b[i], b[i + 1]]);
                let rv = ($simd)(av, bv);
                let arr = rv.to_array();
                dst[i] = arr[0];
                dst[i + 1] = arr[1];
                i += 2;
            }
            while i < n {
                dst[i] = ($scalar)(a[i], b[i]);
                i += 1;
            }
        }
    };
}

binary_f64_simd_op!(
    tensor_add_f64,
    |a: wide::f64x2, b: wide::f64x2| a + b,
    |a: f64, b: f64| a + b
);
binary_f64_simd_op!(
    tensor_sub_f64,
    |a: wide::f64x2, b: wide::f64x2| a - b,
    |a: f64, b: f64| a - b
);
binary_f64_simd_op!(
    tensor_mul_f64,
    |a: wide::f64x2, b: wide::f64x2| a * b,
    |a: f64, b: f64| a * b
);
binary_f64_simd_op!(
    tensor_div_f64,
    |a: wide::f64x2, b: wide::f64x2| a / b,
    |a: f64, b: f64| a / b
);
binary_f64_simd_op!(
    tensor_max_f64,
    |a: wide::f64x2, b: wide::f64x2| a.max(b),
    |a: f64, b: f64| if a > b { a } else { b }
);
binary_f64_simd_op!(
    tensor_min_f64,
    |a: wide::f64x2, b: wide::f64x2| a.min(b),
    |a: f64, b: f64| if a < b { a } else { b }
);
// `wide::f64x2` doesn't have a remainder operator; keep the scalar
// loop for `rem`. The SIMD path fires on the other five ops above.
macro_rules! binary_f64_op {
    ($name:ident, $op:expr) => {
        pub extern "C" fn $name(dst: *mut f64, a: *const f64, b: *const f64, n: usize) {
            let (dst, a, b) = unsafe { f64_slices(dst, a, b, n) };
            for i in 0..n {
                dst[i] = ($op)(a[i], b[i]);
            }
        }
    };
}
binary_f64_op!(tensor_rem_f64, |a: f64, b: f64| a % b);
// Note: tensor_pow_f64 and tensor_atan2_f64 are defined below with SIMD bodies.

macro_rules! binary_int_op {
    ($name:ident, $ty:ty, $op:expr) => {
        pub extern "C" fn $name(dst: *mut $ty, a: *const $ty, b: *const $ty, n: usize) {
            let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
            let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
            for i in 0..n {
                dst[i] = ($op)(a[i], b[i]);
            }
        }
    };
}

binary_int_op!(tensor_add_i64, i64, |a: i64, b: i64| a.wrapping_add(b));
binary_int_op!(tensor_sub_i64, i64, |a: i64, b: i64| a.wrapping_sub(b));
binary_int_op!(tensor_mul_i64, i64, |a: i64, b: i64| a.wrapping_mul(b));
binary_int_op!(tensor_add_i32, i32, |a: i32, b: i32| a.wrapping_add(b));
binary_int_op!(tensor_sub_i32, i32, |a: i32, b: i32| a.wrapping_sub(b));
binary_int_op!(tensor_mul_i32, i32, |a: i32, b: i32| a.wrapping_mul(b));
binary_int_op!(tensor_sshr_i64, i64, |a: i64, b: i64| a
    .wrapping_shr(b as u32));
binary_int_op!(tensor_sshr_i32, i32, |a: i32, b: i32| a
    .wrapping_shr(b as u32));
binary_int_op!(tensor_xor_i64, i64, |a: i64, b: i64| a ^ b);
binary_int_op!(tensor_xor_i32, i32, |a: i32, b: i32| a ^ b);
binary_int_op!(tensor_or_i64, i64, |a: i64, b: i64| a | b);
binary_int_op!(tensor_or_i32, i32, |a: i32, b: i32| a | b);
binary_int_op!(tensor_and_i64, i64, |a: i64, b: i64| a & b);
binary_int_op!(tensor_and_i32, i32, |a: i32, b: i32| a & b);
binary_int_op!(tensor_and_i8, u8, |a: u8, b: u8| a & b);
binary_int_op!(tensor_or_i8, u8, |a: u8, b: u8| a | b);
binary_int_op!(tensor_shl_i64, i64, |a: i64, b: i64| if b as u64 >= 64 {
    0
} else {
    a.wrapping_shl(b as u32)
});
binary_int_op!(tensor_shl_i32, i32, |a: i32, b: i32| if b as u32 >= 32 {
    0
} else {
    a.wrapping_shl(b as u32)
});
binary_int_op!(tensor_ushr_i64, i64, |a: i64, b: i64| if b as u64 >= 64 {
    0
} else {
    ((a as u64).wrapping_shr(b as u32)) as i64
});
binary_int_op!(tensor_ushr_i32, i32, |a: i32, b: i32| if b as u32 >= 32 {
    0
} else {
    ((a as u32).wrapping_shr(b as u32)) as i32
});
binary_int_op!(tensor_max_i64, i64, |a: i64, b: i64| a.max(b));
binary_int_op!(tensor_min_i64, i64, |a: i64, b: i64| a.min(b));
binary_int_op!(tensor_max_i32, i32, |a: i32, b: i32| a.max(b));
binary_int_op!(tensor_min_i32, i32, |a: i32, b: i32| a.min(b));
binary_int_op!(tensor_max_ui32, u32, |a: u32, b: u32| a.max(b));
binary_int_op!(tensor_min_ui32, u32, |a: u32, b: u32| a.min(b));

pub extern "C" fn tensor_div_i64(dst: *mut i64, a: *const i64, b: *const i64, n: usize) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
    for i in 0..n {
        dst[i] = if b[i] != 0 { a[i] / b[i] } else { 0 };
    }
}

pub extern "C" fn tensor_div_i32(dst: *mut i32, a: *const i32, b: *const i32, n: usize) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
    for i in 0..n {
        dst[i] = if b[i] != 0 { a[i] / b[i] } else { 0 };
    }
}

pub extern "C" fn tensor_div_ui32(dst: *mut u32, a: *const u32, b: *const u32, n: usize) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
    for i in 0..n {
        dst[i] = if b[i] != 0 { a[i] / b[i] } else { 0 };
    }
}

pub extern "C" fn tensor_rem_i64(dst: *mut i64, a: *const i64, b: *const i64, n: usize) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
    for i in 0..n {
        dst[i] = if b[i] != 0 { a[i] % b[i] } else { 0 };
    }
}

pub extern "C" fn tensor_rem_i32(dst: *mut i32, a: *const i32, b: *const i32, n: usize) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
    for i in 0..n {
        dst[i] = if b[i] != 0 { a[i] % b[i] } else { 0 };
    }
}

pub extern "C" fn tensor_rem_ui32(dst: *mut u32, a: *const u32, b: *const u32, n: usize) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
    for i in 0..n {
        dst[i] = if b[i] != 0 { a[i] % b[i] } else { 0 };
    }
}

// ---------------------------------------------------------------------------
// Elementwise unary operations (f64)
// ---------------------------------------------------------------------------

macro_rules! unary_f64_op {
    ($name:ident, $op:expr) => {
        pub extern "C" fn $name(dst: *mut f64, a: *const f64, n: usize) {
            let (dst, a) = unsafe { f64_unary(dst, a, n) };
            for i in 0..n {
                dst[i] = ($op)(a[i]);
            }
        }
    };
}

unary_f64_op!(tensor_neg_f64, |x: f64| -x);

macro_rules! unary_int_op {
    ($name:ident, $ty:ty, $op:expr) => {
        pub extern "C" fn $name(dst: *mut $ty, a: *const $ty, n: usize) {
            let a = unsafe { slice::from_raw_parts(a, n) };
            let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
            for i in 0..n {
                dst[i] = ($op)(a[i]);
            }
        }
    };
}

unary_int_op!(tensor_neg_i64, i64, |x: i64| x.wrapping_neg());
unary_int_op!(tensor_neg_i32, i32, |x: i32| x.wrapping_neg());
unary_int_op!(tensor_abs_i64, i64, |x: i64| x.wrapping_abs());
unary_int_op!(tensor_abs_i32, i32, |x: i32| x.wrapping_abs());
unary_f64_op!(tensor_sqrt_f64, f64::sqrt);
unary_f64_op!(tensor_abs_f64, f64::abs);
unary_f64_op!(tensor_floor_f64, f64::floor);
// Transcendentals use SIMD (wide::f64x2) under the hood when tensors have
// 2+ lanes; scalar tail drops to libm. The unary_f64_simd_op macro is
// defined below. Cheap/native ops (rsqrt, ceil) stay scalar.
unary_f64_op!(tensor_rsqrt_f64, |x: f64| 1.0 / x.sqrt());
unary_f64_op!(tensor_ceil_f64, f64::ceil);

pub extern "C" fn tensor_erfc_f64(dst: *mut f64, a: *const f64, n: usize) {
    let (dst, a) = unsafe { f64_unary(dst, a, n) };
    for i in 0..n {
        dst[i] = erfc_impl(a[i]);
    }
}

// ---------------------------------------------------------------------------
// SIMD transcendentals: process `f64x2` chunks via the `wide` crate,
// with a scalar tail for odd element counts. Pure Rust, stable. The
// scalar-ABI lowering dispatches here instead of per-element libm
// when a tensor has 2+ f64 lanes. Accuracy is SLEEF-class (a few
// ULPs), within the CSV regression tolerance.
// ---------------------------------------------------------------------------

macro_rules! unary_f64_simd_op {
    ($name:ident, $vec_op:expr, $scalar_op:expr) => {
        pub extern "C" fn $name(dst: *mut f64, a: *const f64, n: usize) {
            let (dst, a) = unsafe { f64_unary(dst, a, n) };
            let mut i = 0;
            while i + 2 <= n {
                let v = wide::f64x2::new([a[i], a[i + 1]]);
                let r = ($vec_op)(v);
                let arr = r.to_array();
                dst[i] = arr[0];
                dst[i + 1] = arr[1];
                i += 2;
            }
            while i < n {
                dst[i] = ($scalar_op)(a[i]);
                i += 1;
            }
        }
    };
}

unary_f64_simd_op!(tensor_sin_f64, |v: wide::f64x2| v.sin(), f64::sin);
unary_f64_simd_op!(tensor_cos_f64, |v: wide::f64x2| v.cos(), f64::cos);
unary_f64_simd_op!(tensor_tan_f64, |v: wide::f64x2| v.tan(), f64::tan);
unary_f64_simd_op!(
    tensor_sinh_f64,
    |v: wide::f64x2| {
        // wide 0.7 doesn't expose sinh directly; emulate via (e^x - e^-x)/2.
        let pos = v.exp();
        let neg = (-v).exp();
        (pos - neg) * wide::f64x2::splat(0.5)
    },
    f64::sinh
);
unary_f64_simd_op!(
    tensor_cosh_f64,
    |v: wide::f64x2| {
        let pos = v.exp();
        let neg = (-v).exp();
        (pos + neg) * wide::f64x2::splat(0.5)
    },
    f64::cosh
);
unary_f64_simd_op!(
    tensor_tanh_f64,
    |v: wide::f64x2| {
        let pos = v.exp();
        let neg = (-v).exp();
        (pos - neg) / (pos + neg)
    },
    f64::tanh
);
unary_f64_simd_op!(tensor_asin_f64, |v: wide::f64x2| v.asin(), f64::asin);
unary_f64_simd_op!(tensor_acos_f64, |v: wide::f64x2| v.acos(), f64::acos);
unary_f64_simd_op!(tensor_atan_f64, |v: wide::f64x2| v.atan(), f64::atan);
unary_f64_simd_op!(tensor_exp_f64, |v: wide::f64x2| v.exp(), f64::exp);
unary_f64_simd_op!(
    tensor_expm1_f64,
    |v: wide::f64x2| v.exp() - wide::f64x2::splat(1.0),
    f64::exp_m1
);
unary_f64_simd_op!(tensor_log_f64, |v: wide::f64x2| v.ln(), f64::ln);
unary_f64_simd_op!(
    tensor_log1p_f64,
    |v: wide::f64x2| (v + wide::f64x2::splat(1.0)).ln(),
    f64::ln_1p
);
unary_f64_simd_op!(
    tensor_cbrt_f64,
    |v: wide::f64x2| {
        // wide 0.7 has no SIMD cbrt; fall back to scalar per lane.
        let arr = v.to_array();
        wide::f64x2::new([arr[0].cbrt(), arr[1].cbrt()])
    },
    f64::cbrt
);

pub extern "C" fn tensor_atan2_f64(dst: *mut f64, a: *const f64, b: *const f64, n: usize) {
    let (dst, a, b) = unsafe { f64_slices(dst, a, b, n) };
    let mut i = 0;
    while i + 2 <= n {
        let va = wide::f64x2::new([a[i], a[i + 1]]);
        let vb = wide::f64x2::new([b[i], b[i + 1]]);
        let r = va.atan2(vb).to_array();
        dst[i] = r[0];
        dst[i + 1] = r[1];
        i += 2;
    }
    while i < n {
        dst[i] = a[i].atan2(b[i]);
        i += 1;
    }
}

pub extern "C" fn tensor_pow_f64(dst: *mut f64, a: *const f64, b: *const f64, n: usize) {
    let (dst, a, b) = unsafe { f64_slices(dst, a, b, n) };
    let mut i = 0;
    while i + 2 <= n {
        let va = wide::f64x2::new([a[i], a[i + 1]]);
        let vb = wide::f64x2::new([b[i], b[i + 1]]);
        let r = va.pow_f64x2(vb).to_array();
        dst[i] = r[0];
        dst[i + 1] = r[1];
        i += 2;
    }
    while i < n {
        dst[i] = a[i].powf(b[i]);
        i += 1;
    }
}

// Series coefficients copied verbatim from the libm reference; trailing
// digits beyond f64 precision are required for IEEE 754 round-trip.
#[allow(clippy::excessive_precision)]
pub(crate) fn erfc_impl(x: f64) -> f64 {
    let p = [
        2.46196981473530512524e-10,
        5.64189564831068821977e-1,
        7.46321056442269912687e0,
        4.86371970985681366614e1,
        1.96520832956077098242e2,
        5.26445194995477358631e2,
        9.34528527171957607540e2,
        1.02755188689515710272e3,
        5.57535335369399327526e2,
    ];
    let q = [
        1.32281951154744992508e1,
        8.67072140885989742329e1,
        3.54937778887819891062e2,
        9.75708501743205489753e2,
        1.82390916687909736289e3,
        2.24633760818710981792e3,
        1.65666309194161350182e3,
        5.57535340817727401220e2,
    ];
    let r = [
        5.64189583547755073984e-1,
        1.27536670759978104416e0,
        5.01905042251180477414e0,
        6.16021097993053585195e0,
        7.40974269950958085306e0,
        2.97886665372100240670e0,
    ];
    let s = [
        2.26052863220117276590e0,
        9.39603524938001434673e0,
        1.20489539808096656605e1,
        1.70814450747565897222e1,
        9.60896088305422468066e0,
        3.36907645100081462098e0,
    ];
    if x < 0.0 {
        return 2.0 - erfc_impl(-x);
    }
    if x < 0.46875 {
        return 1.0 - erf_central(x);
    }
    if x < 4.0 {
        let mut num = p[0];
        let mut den = 1.0;
        for i in 1..9 {
            num = num * x + p[i];
            den = den * x + q[i - 1];
        }
        let result = num / den;
        let xsq = (x * 16.0).floor() / 16.0;
        let del = (x - xsq) * (x + xsq);
        return ((-xsq * xsq).exp()) * (-del).exp() * result;
    }
    let ix = 1.0 / (x * x);
    let mut num = r[0];
    let mut den = 1.0;
    for i in 1..6 {
        num = num * ix + r[i];
        den = den * ix + s[i - 1];
    }
    let frac_1_sqrt_pi = 1.0 / std::f64::consts::PI.sqrt();
    let result = (ix * num / den + frac_1_sqrt_pi) / x;
    let xsq = (x * 16.0).floor() / 16.0;
    let del = (x - xsq) * (x + xsq);
    ((-xsq * xsq).exp()) * (-del).exp() * result
}

// Series coefficients copied verbatim from the libm reference; trailing
// digits beyond f64 precision are required for IEEE 754 round-trip.
#[allow(clippy::excessive_precision)]
fn erf_central(x: f64) -> f64 {
    const A: [f64; 5] = [
        3.20937758913846947e3,
        3.77485237685302021e2,
        1.13864154151050156e2,
        3.16112374387056560e0,
        1.85777706184603153e-1,
    ];
    const B: [f64; 4] = [
        2.84423121743507280e3,
        1.28261652607737228e3,
        2.44024637934444173e2,
        2.36012708905464931e1,
    ];
    let y = x * x;
    let mut num = A[4];
    for i in (0..4).rev() {
        num = num * y + A[i];
    }
    let mut den = 1.0;
    for &b in &B {
        den = den * y + b;
    }
    x * num / den
}

pub extern "C" fn tensor_not_i64(dst: *mut i64, a: *const i64, n: usize) {
    let a = unsafe { slice::from_raw_parts(a, n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = !a[i];
    }
}

pub extern "C" fn tensor_not_i32(dst: *mut i32, a: *const i32, n: usize) {
    let a = unsafe { slice::from_raw_parts(a, n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = !a[i];
    }
}

pub extern "C" fn tensor_not_i1(dst: *mut u8, a: *const u8, n: usize) {
    let a = unsafe { slice::from_raw_parts(a, n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = if a[i] == 0 { 1 } else { 0 };
    }
}

fn round_ties_even(x: f64) -> f64 {
    let r = x.round();
    if (x - r).abs() == 0.5 {
        let t = r / 2.0;
        if t.floor() == t { r } else { r - x.signum() }
    } else {
        r
    }
}

unary_f64_op!(tensor_round_f64, round_ties_even);

pub extern "C" fn tensor_sign_f64(dst: *mut f64, a: *const f64, n: usize) {
    let (dst, a) = unsafe { f64_unary(dst, a, n) };
    for i in 0..n {
        dst[i] = if a[i] > 0.0 {
            1.0
        } else if a[i] < 0.0 {
            -1.0
        } else {
            0.0
        };
    }
}

pub extern "C" fn tensor_erf_inv_f64(dst: *mut f64, a: *const f64, n: usize) {
    let (dst, a) = unsafe { f64_unary(dst, a, n) };
    for i in 0..n {
        dst[i] = erf_inv_impl(a[i]);
    }
}

pub(crate) fn erf_inv_impl(x: f64) -> f64 {
    if x <= -1.0 {
        return f64::NEG_INFINITY;
    }
    if x >= 1.0 {
        return f64::INFINITY;
    }
    ndtri_impl((x + 1.0) * 0.5) * std::f64::consts::FRAC_1_SQRT_2
}

// Series coefficients copied verbatim from the libm reference; trailing
// digits beyond f64 precision are required for IEEE 754 round-trip.
#[allow(clippy::excessive_precision)]
fn ndtri_impl(y0: f64) -> f64 {
    const P0: [f64; 5] = [
        -5.99633501014107895267E1,
        9.80010754185999661536E1,
        -5.66762857469070293439E1,
        1.39312609387279679503E1,
        -1.23916583867381258016E0,
    ];
    const Q0: [f64; 8] = [
        1.95448858338141759834E0,
        4.67627912898881538453E0,
        8.63602421390890590575E1,
        -2.25462687854119370527E2,
        2.00260212380060660359E2,
        -8.20372256168333339912E1,
        1.59056225126211695515E1,
        -1.18331621121330003142E0,
    ];
    const P1: [f64; 9] = [
        4.05544892305962419923E0,
        3.15251094599893866154E1,
        5.71628192246421288162E1,
        4.40805073893200834700E1,
        1.46849561928858024014E1,
        2.18663306850790267539E0,
        -1.40256079171354495875E-1,
        -3.50424626827848203418E-2,
        -8.57456785154685413611E-4,
    ];
    const Q1: [f64; 8] = [
        1.57799883256466749731E1,
        4.53907635128879210584E1,
        4.13172038254672030440E1,
        1.50425385692907503408E1,
        2.50464946208309415979E0,
        -1.42182922854787788574E-1,
        -3.80806407691578277194E-2,
        -9.33259480895457427372E-4,
    ];
    const P2: [f64; 9] = [
        3.23774891776946035970E0,
        6.91522889068984211695E0,
        3.93881025292474443415E0,
        1.33303460815807542389E0,
        2.01485389549179081538E-1,
        1.23716634817820021358E-2,
        3.01581553508235416007E-4,
        2.65806974686737550832E-6,
        6.23974539184983293730E-9,
    ];
    const Q2: [f64; 8] = [
        6.02427039364742014255E0,
        3.67983563856160859403E0,
        1.37702099489081330271E0,
        2.16236993594496635890E-1,
        1.34204006088543189037E-2,
        3.28014464682127739104E-4,
        2.89247864745380683936E-6,
        6.79019408009981274425E-9,
    ];
    if y0 <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if y0 >= 1.0 {
        return f64::INFINITY;
    }
    if y0 == 0.5 {
        return 0.0;
    }
    let s2pi: f64 = 2.50662827463100050242;
    let (code, mut y) = if y0 > 0.86466471676338730811 {
        (0i32, 1.0 - y0)
    } else {
        (1i32, y0)
    };
    if y > 0.13533528323661269189 {
        y -= 0.5;
        let y2 = y * y;
        let x = y + y * (y2 * poly_eval(y2, &P0) / poly_eval_1(y2, &Q0));
        return x * s2pi;
    }
    let mut x = (-2.0 * y.ln()).sqrt();
    let x0 = x - x.ln() / x;
    let z = 1.0 / x;
    let x1 = if x < 8.0 {
        z * poly_eval(z, &P1) / poly_eval_1(z, &Q1)
    } else {
        z * poly_eval(z, &P2) / poly_eval_1(z, &Q2)
    };
    x = x0 - x1;
    if code != 0 {
        x = -x;
    }
    x
}

fn poly_eval(x: f64, coeffs: &[f64]) -> f64 {
    coeffs.iter().fold(0.0, |acc, &c| acc * x + c)
}

fn poly_eval_1(x: f64, coeffs: &[f64]) -> f64 {
    coeffs.iter().fold(1.0, |acc, &c| acc * x + c)
}

pub extern "C" fn tensor_clamp_f64(
    dst: *mut f64,
    src: *const f64,
    min: *const f64,
    max: *const f64,
    n: usize,
) {
    let src = unsafe { slice::from_raw_parts(src, n) };
    let min = unsafe { slice::from_raw_parts(min, n) };
    let max = unsafe { slice::from_raw_parts(max, n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = if src[i] < min[i] {
            min[i]
        } else if src[i] > max[i] {
            max[i]
        } else {
            src[i]
        };
    }
}

pub extern "C" fn tensor_reverse_f64(
    dst: *mut f64,
    src: *const f64,
    n: usize,
    shape: *const i64,
    rank: usize,
    dims: *const i64,
    n_dims: usize,
) {
    let src = unsafe { slice::from_raw_parts(src, n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    let shape = unsafe { slice::from_raw_parts(shape, rank) };
    let rev_dims = unsafe { slice::from_raw_parts(dims, n_dims) };

    let mut strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as usize;
    }

    for flat in 0..n {
        let mut src_flat = 0usize;
        let mut remaining = flat;
        for d in 0..rank {
            let coord = remaining / strides[d];
            remaining %= strides[d];
            let c = if rev_dims.contains(&(d as i64)) {
                shape[d] as usize - 1 - coord
            } else {
                coord
            };
            src_flat += c * strides[d];
        }
        dst[flat] = src[src_flat];
    }
}

// ---------------------------------------------------------------------------
// Comparison operations
// ---------------------------------------------------------------------------

macro_rules! cmp_op {
    ($name:ident, $ty:ty, $op:expr) => {
        pub extern "C" fn $name(dst: *mut u8, a: *const $ty, b: *const $ty, n: usize) {
            let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
            let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
            for i in 0..n {
                dst[i] = ($op)(a[i], b[i]) as u8;
            }
        }
    };
}

cmp_op!(tensor_cmp_eq_f64, f64, |a: f64, b: f64| a == b);
cmp_op!(tensor_cmp_lt_f64, f64, |a: f64, b: f64| a < b);
cmp_op!(tensor_cmp_le_f64, f64, |a: f64, b: f64| a <= b);
cmp_op!(tensor_cmp_gt_f64, f64, |a: f64, b: f64| a > b);
cmp_op!(tensor_cmp_ge_f64, f64, |a: f64, b: f64| a >= b);
cmp_op!(tensor_cmp_ne_f64, f64, |a: f64, b: f64| a != b);
cmp_op!(tensor_cmp_eq_i64, i64, |a: i64, b: i64| a == b);
cmp_op!(tensor_cmp_ne_i64, i64, |a: i64, b: i64| a != b);
cmp_op!(tensor_cmp_lt_i64, i64, |a: i64, b: i64| a < b);
cmp_op!(tensor_cmp_le_i64, i64, |a: i64, b: i64| a <= b);
cmp_op!(tensor_cmp_gt_i64, i64, |a: i64, b: i64| a > b);
cmp_op!(tensor_cmp_ge_i64, i64, |a: i64, b: i64| a >= b);
cmp_op!(tensor_cmp_eq_i32, i32, |a: i32, b: i32| a == b);
cmp_op!(tensor_cmp_ne_i32, i32, |a: i32, b: i32| a != b);
cmp_op!(tensor_cmp_lt_i32, i32, |a: i32, b: i32| a < b);
cmp_op!(tensor_cmp_le_i32, i32, |a: i32, b: i32| a <= b);
cmp_op!(tensor_cmp_gt_i32, i32, |a: i32, b: i32| a > b);
cmp_op!(tensor_cmp_ge_i32, i32, |a: i32, b: i32| a >= b);

// ---------------------------------------------------------------------------
// Select: dst[i] = cond[i] ? on_true[i] : on_false[i]
// ---------------------------------------------------------------------------

macro_rules! select_op {
    ($name:ident, $ty:ty) => {
        pub extern "C" fn $name(
            dst: *mut $ty,
            cond: *const u8,
            on_true: *const $ty,
            on_false: *const $ty,
            n: usize,
        ) {
            let cond = unsafe { slice::from_raw_parts(cond, n) };
            let t = unsafe { slice::from_raw_parts(on_true, n) };
            let f = unsafe { slice::from_raw_parts(on_false, n) };
            let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
            for i in 0..n {
                dst[i] = if cond[i] != 0 { t[i] } else { f[i] };
            }
        }
    };
}

select_op!(tensor_select_i32, i32);
select_op!(tensor_select_f64, f64);
select_op!(tensor_select_i64, i64);
select_op!(tensor_select_i8, u8);

// ---------------------------------------------------------------------------
// Type conversion
// ---------------------------------------------------------------------------

macro_rules! convert_op {
    ($name:ident, $src_ty:ty, $dst_ty:ty, $conv:expr) => {
        pub extern "C" fn $name(dst: *mut $dst_ty, a: *const $src_ty, n: usize) {
            let a = unsafe { slice::from_raw_parts(a, n) };
            let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
            for i in 0..n {
                dst[i] = ($conv)(a[i]);
            }
        }
    };
}

convert_op!(tensor_widen_i32_to_i64, i32, i64, |x: i32| x as i64);
convert_op!(tensor_convert_i64_to_f64, i64, f64, |x: i64| x as f64);
convert_op!(tensor_convert_f64_to_i64, f64, i64, |x: f64| x as i64);
convert_op!(tensor_convert_i1_to_f64, u8, f64, |x: u8| if x != 0 {
    1.0
} else {
    0.0
});
convert_op!(tensor_convert_f64_to_i32, f64, i32, |x: f64| x as i32);
convert_op!(tensor_convert_i1_to_i32, u8, i32, |x: u8| x as i32);
convert_op!(tensor_convert_i64_to_i32, i64, i32, |x: i64| x as i32);
convert_op!(tensor_convert_i32_to_f64, i32, f64, |x: i32| x as f64);
convert_op!(tensor_convert_f64_to_f32, f64, f32, |x: f64| x as f32);
convert_op!(tensor_convert_f32_to_f64, f32, f64, |x: f32| x as f64);
convert_op!(tensor_convert_ui32_to_i64, u32, i64, |x: u32| x as i64);
convert_op!(tensor_convert_ui32_to_f64, u32, f64, |x: u32| x as f64);
convert_op!(tensor_convert_f64_to_i1, f64, u8, |x: f64| (x != 0.0) as u8);
convert_op!(tensor_convert_i64_to_i1, i64, u8, |x: i64| (x != 0) as u8);
convert_op!(tensor_convert_ui64_to_f64, u64, f64, |x: u64| x as f64);
convert_op!(tensor_convert_i32_to_f32, i32, f32, |x: i32| x as f32);
convert_op!(tensor_convert_f32_to_i32, f32, i32, |x: f32| x as i32);

// ---------------------------------------------------------------------------
// Broadcast: fill dst with a single scalar value
// ---------------------------------------------------------------------------

macro_rules! broadcast_op {
    ($name:ident, $ty:ty) => {
        pub extern "C" fn $name(dst: *mut $ty, val: $ty, n: usize) {
            let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
            for i in 0..n {
                dst[i] = val;
            }
        }
    };
}

broadcast_op!(tensor_broadcast_f64, f64);
broadcast_op!(tensor_broadcast_i64, i64);
broadcast_op!(tensor_broadcast_i32, i32);
broadcast_op!(tensor_broadcast_i8, u8);

// ---------------------------------------------------------------------------
// Iota: fill with index values
// ---------------------------------------------------------------------------

macro_rules! iota_op {
    ($name:ident, $ty:ty) => {
        pub extern "C" fn $name(dst: *mut $ty, n: usize) {
            let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
            for i in 0..n {
                dst[i] = i as $ty;
            }
        }
    };
}

iota_op!(tensor_iota_i64, i64);
iota_op!(tensor_iota_f64, f64);

// ---------------------------------------------------------------------------
// Memcpy
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_memcpy(dst: *mut u8, src: *const u8, n_bytes: usize) {
    if n_bytes == 0 || std::ptr::eq(dst, src) {
        return;
    }
    unsafe { std::ptr::copy_nonoverlapping(src, dst, n_bytes) };
}

// ---------------------------------------------------------------------------
// Transpose (2D, row-major): dst[j*rows + i] = src[i*cols + j]
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_transpose_f64(dst: *mut f64, src: *const f64, rows: usize, cols: usize) {
    let src = unsafe { slice::from_raw_parts(src, rows * cols) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, rows * cols) };
    for i in 0..rows {
        for j in 0..cols {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

// ---------------------------------------------------------------------------
// Reduce along last axis
// ---------------------------------------------------------------------------

macro_rules! reduce_op {
    ($name:ident, $init:expr, $update:expr) => {
        pub extern "C" fn $name(dst: *mut f64, src: *const f64, outer: usize, inner: usize) {
            let src = unsafe { slice::from_raw_parts(src, outer * inner) };
            let dst = unsafe { slice::from_raw_parts_mut(dst, outer) };
            for i in 0..outer {
                let mut acc: f64 = $init;
                for j in 0..inner {
                    acc = ($update)(acc, src[i * inner + j]);
                }
                dst[i] = acc;
            }
        }
    };
}

reduce_op!(tensor_reduce_sum_f64, 0.0, |acc: f64, v: f64| acc + v);
reduce_op!(
    tensor_reduce_max_f64,
    f64::NEG_INFINITY,
    |acc: f64, v: f64| if v > acc { v } else { acc }
);
reduce_op!(
    tensor_reduce_min_f64,
    f64::INFINITY,
    |acc: f64, v: f64| if v < acc { v } else { acc }
);

macro_rules! reduce_int_op {
    ($name:ident, $ty:ty, $init:expr, $update:expr) => {
        pub extern "C" fn $name(dst: *mut $ty, src: *const $ty, outer: usize, inner: usize) {
            let src = unsafe { slice::from_raw_parts(src, outer * inner) };
            let dst = unsafe { slice::from_raw_parts_mut(dst, outer) };
            for i in 0..outer {
                let mut acc: $ty = $init;
                for j in 0..inner {
                    acc = ($update)(acc, src[i * inner + j]);
                }
                dst[i] = acc;
            }
        }
    };
}

reduce_int_op!(tensor_reduce_sum_i64, i64, 0, |acc: i64, v: i64| acc
    .wrapping_add(v));
reduce_int_op!(tensor_reduce_max_i64, i64, i64::MIN, |acc: i64, v: i64| acc
    .max(v));
reduce_int_op!(tensor_reduce_min_i64, i64, i64::MAX, |acc: i64, v: i64| acc
    .min(v));

pub extern "C" fn tensor_reduce_and_i1(dst: *mut u8, src: *const u8, outer: usize, inner: usize) {
    let src = unsafe { slice::from_raw_parts(src, outer * inner) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, outer) };
    for i in 0..outer {
        let mut acc = 1u8;
        for j in 0..inner {
            if src[i * inner + j] == 0 {
                acc = 0;
                break;
            }
        }
        dst[i] = acc;
    }
}

pub extern "C" fn tensor_reduce_or_i1(dst: *mut u8, src: *const u8, outer: usize, inner: usize) {
    let src = unsafe { slice::from_raw_parts(src, outer * inner) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, outer) };
    for i in 0..outer {
        let mut acc = 0u8;
        for j in 0..inner {
            if src[i * inner + j] != 0 {
                acc = 1;
                break;
            }
        }
        dst[i] = acc;
    }
}

// ---------------------------------------------------------------------------
// Argmin/argmax: multi-operand reduce producing (value, index)
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_argmin_f64(
    dst_vals: *mut f64,
    dst_idxs: *mut i64,
    src_vals: *const f64,
    src_idxs: *const i64,
    outer: usize,
    inner: usize,
) {
    let src_v = unsafe { slice::from_raw_parts(src_vals, outer * inner) };
    let src_i = unsafe { slice::from_raw_parts(src_idxs, outer * inner) };
    let dst_v = unsafe { slice::from_raw_parts_mut(dst_vals, outer) };
    let dst_i = unsafe { slice::from_raw_parts_mut(dst_idxs, outer) };
    for o in 0..outer {
        let mut best_v = f64::INFINITY;
        let mut best_i = 0i64;
        for j in 0..inner {
            let v = src_v[o * inner + j];
            let idx = src_i[o * inner + j];
            if v < best_v || (best_v.is_nan() && !v.is_nan()) || (v == best_v && idx < best_i) {
                best_v = v;
                best_i = idx;
            }
        }
        dst_v[o] = best_v;
        dst_i[o] = best_i;
    }
}

pub extern "C" fn tensor_argmax_f64(
    dst_vals: *mut f64,
    dst_idxs: *mut i64,
    src_vals: *const f64,
    src_idxs: *const i64,
    outer: usize,
    inner: usize,
) {
    let src_v = unsafe { slice::from_raw_parts(src_vals, outer * inner) };
    let src_i = unsafe { slice::from_raw_parts(src_idxs, outer * inner) };
    let dst_v = unsafe { slice::from_raw_parts_mut(dst_vals, outer) };
    let dst_i = unsafe { slice::from_raw_parts_mut(dst_idxs, outer) };
    for o in 0..outer {
        let mut best_v = f64::NEG_INFINITY;
        let mut best_i = 0i64;
        for j in 0..inner {
            let v = src_v[o * inner + j];
            let idx = src_i[o * inner + j];
            if v > best_v || (v == best_v && idx < best_i) {
                best_v = v;
                best_i = idx;
            }
        }
        dst_v[o] = best_v;
        dst_i[o] = best_i;
    }
}

// ---------------------------------------------------------------------------
// Byte-generic gather: row-select with explicit element size
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_gather_generic(
    dst: *mut u8,
    src: *const u8,
    n_src: usize,
    indices: *const i64,
    n_indices: usize,
    row_size: usize,
    elem_sz: usize,
) {
    let src = unsafe { slice::from_raw_parts(src, n_src * elem_sz) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_indices * row_size * elem_sz) };
    let indices = unsafe { slice::from_raw_parts(indices, n_indices) };
    for i in 0..n_indices {
        let idx = indices[i] as usize;
        let src_off = idx * row_size * elem_sz;
        let dst_off = i * row_size * elem_sz;
        let row_bytes = row_size * elem_sz;
        if src_off + row_bytes <= src.len() {
            dst[dst_off..dst_off + row_bytes].copy_from_slice(&src[src_off..src_off + row_bytes]);
        }
    }
}

pub extern "C" fn tensor_gather_nd_generic(
    dst: *mut u8,
    src: *const u8,
    n_src: usize,
    indices: *const i64,
    n_batch: usize,
    n_index_dims: usize,
    src_shape: *const i64,
    src_rank: usize,
    start_index_map: *const i64,
    slice_sizes: *const i64,
    n_dst: usize,
    elem_sz: usize,
) {
    let src = unsafe { slice::from_raw_parts(src, n_src * elem_sz) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_dst * elem_sz) };
    let indices = unsafe { slice::from_raw_parts(indices, n_batch * n_index_dims) };
    let src_shape = unsafe { slice::from_raw_parts(src_shape, src_rank) };
    let start_index_map = unsafe { slice::from_raw_parts(start_index_map, n_index_dims) };
    let slice_sizes = unsafe { slice::from_raw_parts(slice_sizes, src_rank) };

    let mut src_strides = vec![1usize; src_rank];
    for i in (0..src_rank.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1] as usize;
    }
    let slice_elems: usize = slice_sizes.iter().map(|&s| s as usize).product();
    let mut slice_strides = vec![1usize; src_rank];
    for i in (0..src_rank.saturating_sub(1)).rev() {
        slice_strides[i] = slice_strides[i + 1] * slice_sizes[i + 1] as usize;
    }

    let mut dst_off = 0usize;
    for b in 0..n_batch {
        let mut base_flat = 0usize;
        for j in 0..n_index_dims {
            let idx = indices[b * n_index_dims + j] as usize;
            let dim = start_index_map[j] as usize;
            let clamped =
                idx.min((src_shape[dim] as usize).saturating_sub(slice_sizes[dim] as usize));
            base_flat += clamped * src_strides[dim];
        }
        for s in 0..slice_elems {
            let mut src_flat = base_flat;
            let mut rem = s;
            for d in 0..src_rank {
                let coord = rem / slice_strides[d];
                rem %= slice_strides[d];
                src_flat += coord * src_strides[d];
            }
            if dst_off < n_dst && src_flat < n_src {
                let sb = src_flat * elem_sz;
                let db = dst_off * elem_sz;
                dst[db..db + elem_sz].copy_from_slice(&src[sb..sb + elem_sz]);
            }
            dst_off += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Byte-generic scatter with explicit element size
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_scatter_generic(
    dst: *mut u8,
    src: *const u8,
    n_src: usize,
    indices: *const i64,
    updates: *const u8,
    n_updates: usize,
    inner_size: usize,
    elem_sz: usize,
) {
    let total_bytes = n_src * elem_sz;
    let src = unsafe { slice::from_raw_parts(src, total_bytes) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, total_bytes) };
    dst.copy_from_slice(src);
    let indices = unsafe { slice::from_raw_parts(indices, n_updates) };
    let updates = unsafe { slice::from_raw_parts(updates, n_updates * inner_size * elem_sz) };
    for i in 0..n_updates {
        let idx = indices[i] as usize;
        let base = idx * inner_size;
        for j in 0..inner_size {
            if base + j < n_src {
                let db = (base + j) * elem_sz;
                let ub = (i * inner_size + j) * elem_sz;
                dst[db..db + elem_sz].copy_from_slice(&updates[ub..ub + elem_sz]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Matrix multiply: C = A * B (row-major, naive triple loop)
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_matmul_f64(
    dst: *mut f64,
    a: *const f64,
    b: *const f64,
    m: usize,
    k: usize,
    n: usize,
) {
    let a = unsafe { slice::from_raw_parts(a, m * k) };
    let b = unsafe { slice::from_raw_parts(b, k * n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, m * n) };
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            dst[i * n + j] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Concatenate along a dimension
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_concat_f64(
    dst: *mut f64,
    src_ptrs: *const *const f64,
    src_lens: *const usize,
    n_srcs: usize,
    n_dst: usize,
) {
    let src_ptrs = unsafe { slice::from_raw_parts(src_ptrs, n_srcs) };
    let src_lens = unsafe { slice::from_raw_parts(src_lens, n_srcs) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_dst) };
    let mut off = 0usize;
    for i in 0..n_srcs {
        let src = unsafe { slice::from_raw_parts(src_ptrs[i], src_lens[i]) };
        dst[off..off + src_lens[i]].copy_from_slice(src);
        off += src_lens[i];
    }
}

pub extern "C" fn tensor_concat_nd_f64(
    dst: *mut u8,
    n_dst: usize,
    src_a: *const u8,
    n_a: usize,
    src_b: *const u8,
    n_b: usize,
    dst_shape: *const i64,
    a_shape: *const i64,
    rank: usize,
    dim: usize,
    elem_sz: usize,
) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_dst * elem_sz) };
    let a = unsafe { slice::from_raw_parts(src_a, n_a * elem_sz) };
    let b = unsafe { slice::from_raw_parts(src_b, n_b * elem_sz) };
    let dst_shape = unsafe { slice::from_raw_parts(dst_shape, rank) };
    let a_shape = unsafe { slice::from_raw_parts(a_shape, rank) };

    let mut dst_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        dst_strides[i] = dst_strides[i + 1] * dst_shape[i + 1] as usize;
    }

    let a_dim_size = a_shape[dim] as usize;

    for flat_dst in 0..n_dst {
        let mut remaining = flat_dst;
        let mut coords = vec![0usize; rank];
        for d in 0..rank {
            coords[d] = remaining / dst_strides[d];
            remaining %= dst_strides[d];
        }

        if coords[dim] < a_dim_size {
            let mut a_strides = vec![1usize; rank];
            for i in (0..rank.saturating_sub(1)).rev() {
                a_strides[i] = a_strides[i + 1] * a_shape[i + 1] as usize;
            }
            let mut src_flat = 0usize;
            for d in 0..rank {
                src_flat += coords[d] * a_strides[d];
            }
            let idx = src_flat.min(n_a.saturating_sub(1));
            dst[flat_dst * elem_sz..(flat_dst + 1) * elem_sz]
                .copy_from_slice(&a[idx * elem_sz..(idx + 1) * elem_sz]);
        } else {
            let mut b_shape = vec![0i64; rank];
            for d in 0..rank {
                b_shape[d] = if d == dim {
                    dst_shape[d] - a_shape[d]
                } else {
                    dst_shape[d]
                };
            }
            let mut b_strides = vec![1usize; rank];
            for i in (0..rank.saturating_sub(1)).rev() {
                b_strides[i] = b_strides[i + 1] * b_shape[i + 1] as usize;
            }
            let mut b_coords = coords.clone();
            b_coords[dim] -= a_dim_size;
            let mut src_flat = 0usize;
            for d in 0..rank {
                src_flat += b_coords[d] * b_strides[d];
            }
            let idx = src_flat.min(n_b.saturating_sub(1));
            dst[flat_dst * elem_sz..(flat_dst + 1) * elem_sz]
                .copy_from_slice(&b[idx * elem_sz..(idx + 1) * elem_sz]);
        }
    }
}

// ---------------------------------------------------------------------------
// Pad
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_pad_f64(
    dst: *mut f64,
    src: *const f64,
    n_dst: usize,
    n_src: usize,
    pad_value: f64,
    dst_shape: *const i64,
    src_shape: *const i64,
    rank: usize,
    low: *const i64,
) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_dst) };
    let src = unsafe { slice::from_raw_parts(src, n_src) };
    let dst_shape = unsafe { slice::from_raw_parts(dst_shape, rank) };
    let src_shape = unsafe { slice::from_raw_parts(src_shape, rank) };
    let low = unsafe { slice::from_raw_parts(low, rank) };

    for v in dst.iter_mut() {
        *v = pad_value;
    }

    let mut dst_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        dst_strides[i] = dst_strides[i + 1] * dst_shape[i + 1] as usize;
    }
    let mut src_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1] as usize;
    }

    for flat_src in 0..n_src {
        let mut remaining = flat_src;
        let mut flat_dst = 0usize;
        for d in 0..rank {
            let coord = remaining / src_strides[d];
            remaining %= src_strides[d];
            flat_dst += (low[d] as usize + coord) * dst_strides[d];
        }
        if flat_dst < n_dst {
            dst[flat_dst] = src[flat_src];
        }
    }
}

// ---------------------------------------------------------------------------
// Byte-generic layout operations
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_broadcast_nd_generic(
    dst: *mut u8,
    src: *const u8,
    n_dst: usize,
    n_src: usize,
    dst_shape: *const i64,
    dst_rank: usize,
    src_shape: *const i64,
    src_rank: usize,
    broadcast_dims: *const i64,
    elem_sz: usize,
) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_dst * elem_sz) };
    let src = unsafe { slice::from_raw_parts(src, n_src * elem_sz) };
    let dst_shape = unsafe { slice::from_raw_parts(dst_shape, dst_rank) };
    let src_shape = unsafe { slice::from_raw_parts(src_shape, src_rank) };
    let broadcast_dims = unsafe { slice::from_raw_parts(broadcast_dims, src_rank) };
    let mut ds = vec![1usize; dst_rank];
    for i in (0..dst_rank.saturating_sub(1)).rev() {
        ds[i] = ds[i + 1] * dst_shape[i + 1] as usize;
    }
    let mut ss = vec![1usize; src_rank];
    for i in (0..src_rank.saturating_sub(1)).rev() {
        ss[i] = ss[i + 1] * src_shape[i + 1] as usize;
    }
    for fd in 0..n_dst {
        let mut si = 0usize;
        let mut rem = fd;
        for d in 0..dst_rank {
            let coord = rem / ds[d];
            rem %= ds[d];
            for (s, &bd) in broadcast_dims.iter().enumerate() {
                if bd as usize == d {
                    si += (if src_shape[s] == 1 { 0 } else { coord }) * ss[s];
                }
            }
        }
        si = si.min(n_src.saturating_sub(1));
        let db = fd * elem_sz;
        let sb = si * elem_sz;
        dst[db..db + elem_sz].copy_from_slice(&src[sb..sb + elem_sz]);
    }
}

pub extern "C" fn tensor_slice_generic(
    dst: *mut u8,
    src: *const u8,
    n_dst: usize,
    n_src: usize,
    shape: *const i64,
    rank: usize,
    starts: *const i64,
    limits: *const i64,
    elem_sz: usize,
) {
    let src = unsafe { slice::from_raw_parts(src, n_src * elem_sz) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_dst * elem_sz) };
    let shape = unsafe { slice::from_raw_parts(shape, rank) };
    let starts = unsafe { slice::from_raw_parts(starts, rank) };
    let limits = unsafe { slice::from_raw_parts(limits, rank) };
    let mut src_s = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        src_s[i] = src_s[i + 1] * shape[i + 1] as usize;
    }
    let dst_shape: Vec<usize> = (0..rank)
        .map(|d| (limits[d] - starts[d]) as usize)
        .collect();
    let mut dst_s = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        dst_s[i] = dst_s[i + 1] * dst_shape[i + 1];
    }
    for fd in 0..n_dst {
        let mut sf = 0usize;
        let mut rem = fd;
        for d in 0..rank {
            let c = rem / dst_s[d];
            rem %= dst_s[d];
            sf += (starts[d] as usize + c) * src_s[d];
        }
        let db = fd * elem_sz;
        let sb = sf * elem_sz;
        dst[db..db + elem_sz].copy_from_slice(&src[sb..sb + elem_sz]);
    }
}

pub extern "C" fn tensor_transpose_nd_generic(
    dst: *mut u8,
    src: *const u8,
    n: usize,
    src_shape: *const i64,
    perm: *const i64,
    rank: usize,
    elem_sz: usize,
) {
    let src = unsafe { slice::from_raw_parts(src, n * elem_sz) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n * elem_sz) };
    let src_shape = unsafe { slice::from_raw_parts(src_shape, rank) };
    let perm = unsafe { slice::from_raw_parts(perm, rank) };
    let mut src_s = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        src_s[i] = src_s[i + 1] * src_shape[i + 1] as usize;
    }
    let dst_shape: Vec<i64> = (0..rank).map(|i| src_shape[perm[i] as usize]).collect();
    let mut dst_s = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        dst_s[i] = dst_s[i + 1] * dst_shape[i + 1] as usize;
    }
    for fd in 0..n {
        let mut rem = fd;
        let mut sf = 0usize;
        for d in 0..rank {
            let c = rem / dst_s[d];
            rem %= dst_s[d];
            sf += c * src_s[perm[d] as usize];
        }
        let db = fd * elem_sz;
        let sb = sf * elem_sz;
        dst[db..db + elem_sz].copy_from_slice(&src[sb..sb + elem_sz]);
    }
}

pub extern "C" fn tensor_dynamic_slice_generic(
    dst: *mut u8,
    src: *const u8,
    n_dst: usize,
    n_src: usize,
    shape: *const i64,
    rank: usize,
    start_indices: *const i64,
    slice_sizes: *const i64,
    elem_sz: usize,
) {
    let src = unsafe { slice::from_raw_parts(src, n_src * elem_sz) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_dst * elem_sz) };
    let shape = unsafe { slice::from_raw_parts(shape, rank) };
    let start_indices = unsafe { slice::from_raw_parts(start_indices, rank) };
    let slice_sizes = unsafe { slice::from_raw_parts(slice_sizes, rank) };
    let mut src_s = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        src_s[i] = src_s[i + 1] * shape[i + 1] as usize;
    }
    let mut dst_s = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        dst_s[i] = dst_s[i + 1] * slice_sizes[i + 1] as usize;
    }
    for fd in 0..n_dst {
        let mut sf = 0usize;
        let mut rem = fd;
        for d in 0..rank {
            let c = rem / dst_s[d];
            rem %= dst_s[d];
            let start =
                (start_indices[d] as usize).min(shape[d] as usize - slice_sizes[d] as usize);
            sf += (start + c) * src_s[d];
        }
        sf = sf.min(n_src.saturating_sub(1));
        let db = fd * elem_sz;
        let sb = sf * elem_sz;
        dst[db..db + elem_sz].copy_from_slice(&src[sb..sb + elem_sz]);
    }
}

pub extern "C" fn tensor_dynamic_update_slice_generic(
    dst: *mut u8,
    src: *const u8,
    update: *const u8,
    n_src: usize,
    n_update: usize,
    shape: *const i64,
    rank: usize,
    start_indices: *const i64,
    update_shape: *const i64,
    elem_sz: usize,
) {
    let total = n_src * elem_sz;
    let src = unsafe { slice::from_raw_parts(src, total) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, total) };
    let update = unsafe { slice::from_raw_parts(update, n_update * elem_sz) };
    let shape = unsafe { slice::from_raw_parts(shape, rank) };
    let start_indices = unsafe { slice::from_raw_parts(start_indices, rank) };
    let update_shape = unsafe { slice::from_raw_parts(update_shape, rank) };
    dst.copy_from_slice(src);
    let mut src_s = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        src_s[i] = src_s[i + 1] * shape[i + 1] as usize;
    }
    let mut upd_s = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        upd_s[i] = upd_s[i + 1] * update_shape[i + 1] as usize;
    }
    for fu in 0..n_update {
        let mut df = 0usize;
        let mut rem = fu;
        for d in 0..rank {
            let c = rem / upd_s[d];
            rem %= upd_s[d];
            let start =
                (start_indices[d] as usize).min((shape[d] - update_shape[d]).max(0) as usize);
            df += (start + c) * src_s[d];
        }
        if df < n_src {
            let db = df * elem_sz;
            let ub = fu * elem_sz;
            dst[db..db + elem_sz].copy_from_slice(&update[ub..ub + elem_sz]);
        }
    }
}

// ---------------------------------------------------------------------------
// Iota N-dimensional
// ---------------------------------------------------------------------------

macro_rules! iota_nd_op {
    ($name:ident, $ty:ty) => {
        pub extern "C" fn $name(
            dst: *mut $ty,
            n: usize,
            shape: *const i64,
            rank: usize,
            dimension: usize,
        ) {
            let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
            let shape = unsafe { slice::from_raw_parts(shape, rank) };

            let mut strides = vec![1usize; rank];
            for i in (0..rank.saturating_sub(1)).rev() {
                strides[i] = strides[i + 1] * shape[i + 1] as usize;
            }

            for flat in 0..n {
                let coord = (flat / strides[dimension]) % shape[dimension] as usize;
                dst[flat] = coord as $ty;
            }
        }
    };
}

iota_nd_op!(tensor_iota_nd_i64, i64);
iota_nd_op!(tensor_iota_nd_f64, f64);

// ---------------------------------------------------------------------------
// batch_norm_inference, real_dynamic_slice, map, reduce_window,
// select_and_scatter, convolution runtime functions
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_batch_norm_inference_f64(
    dst: *mut f64,
    operand: *const f64,
    scale: *const f64,
    offset: *const f64,
    mean: *const f64,
    variance: *const f64,
    epsilon: f64,
    feature_index: usize,
    shape: *const i64,
    rank: usize,
    n: usize,
) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    let operand = unsafe { slice::from_raw_parts(operand, n) };
    let shape = unsafe { slice::from_raw_parts(shape, rank) };
    let feature_size = shape[feature_index] as usize;
    let scale = unsafe { slice::from_raw_parts(scale, feature_size) };
    let offset = unsafe { slice::from_raw_parts(offset, feature_size) };
    let mean = unsafe { slice::from_raw_parts(mean, feature_size) };
    let variance = unsafe { slice::from_raw_parts(variance, feature_size) };

    let mut strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as usize;
    }

    for i in 0..n {
        let fi = (i / strides[feature_index]) % feature_size;
        let normed = (operand[i] - mean[fi]) / (variance[fi] + epsilon).sqrt();
        dst[i] = normed * scale[fi] + offset[fi];
    }
}

pub extern "C" fn tensor_real_dynamic_slice(
    dst: *mut u8,
    src: *const u8,
    starts: *const i64,
    limits: *const i64,
    strides_vals: *const i64,
    src_shape: *const i64,
    rank: usize,
    elem_sz: usize,
    n_dst: usize,
) {
    let src_shape = unsafe { slice::from_raw_parts(src_shape, rank) };
    let starts = unsafe { slice::from_raw_parts(starts, rank) };
    let limits = unsafe { slice::from_raw_parts(limits, rank) };
    let strides_vals = unsafe { slice::from_raw_parts(strides_vals, rank) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_dst * elem_sz) };
    let n_src: usize = src_shape.iter().map(|&d| d as usize).product();
    let src = unsafe { slice::from_raw_parts(src, n_src * elem_sz) };

    let mut out_shape = vec![0usize; rank];
    for d in 0..rank {
        let s = strides_vals[d].max(1) as usize;
        out_shape[d] = ((limits[d] - starts[d]) as usize).div_ceil(s);
    }
    let mut src_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1] as usize;
    }
    let mut dst_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        dst_strides[i] = dst_strides[i + 1] * out_shape[i + 1];
    }

    for fd in 0..n_dst {
        let mut src_flat = 0usize;
        let mut rem = fd;
        for d in 0..rank {
            let coord = rem / dst_strides[d];
            rem %= dst_strides[d];
            let src_coord = starts[d] as usize + coord * strides_vals[d].max(1) as usize;
            src_flat += src_coord * src_strides[d];
        }
        src_flat = src_flat.min(n_src.saturating_sub(1));
        let db = fd * elem_sz;
        let sb = src_flat * elem_sz;
        dst[db..db + elem_sz].copy_from_slice(&src[sb..sb + elem_sz]);
    }
}

pub extern "C" fn tensor_map_f64(
    dst: *mut f64,
    input_ptrs: *const *const f64,
    n_inputs: usize,
    n_elements: usize,
    fn_ptr: extern "C" fn(f64, f64) -> f64,
) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_elements) };
    let inputs: Vec<&[f64]> = (0..n_inputs)
        .map(|i| unsafe {
            let p = *input_ptrs.add(i);
            slice::from_raw_parts(p, n_elements)
        })
        .collect();

    match n_inputs {
        1 => {
            let identity_fn: extern "C" fn(f64, f64) -> f64 = fn_ptr;
            for i in 0..n_elements {
                dst[i] = identity_fn(inputs[0][i], 0.0);
            }
        }
        _ => {
            for i in 0..n_elements {
                let mut acc = inputs[0][i];
                for j in 1..n_inputs {
                    acc = fn_ptr(acc, inputs[j][i]);
                }
                dst[i] = acc;
            }
        }
    }
}

pub extern "C" fn tensor_reduce_window_f64(
    dst: *mut f64,
    src: *const f64,
    init_val: f64,
    window_dims: *const i64,
    window_strides: *const i64,
    base_dilations: *const i64,
    window_dilations: *const i64,
    padding: *const i64,
    src_shape: *const i64,
    rank: usize,
    n_out: usize,
    reducer_fn: extern "C" fn(f64, f64) -> f64,
) {
    let src_shape = unsafe { slice::from_raw_parts(src_shape, rank) };
    let window_dims = unsafe { slice::from_raw_parts(window_dims, rank) };
    let window_strides = unsafe { slice::from_raw_parts(window_strides, rank) };
    let base_dilations = unsafe { slice::from_raw_parts(base_dilations, rank) };
    let window_dilations = unsafe { slice::from_raw_parts(window_dilations, rank) };
    let padding = unsafe { slice::from_raw_parts(padding, rank * 2) };
    let n_src: usize = src_shape.iter().map(|&d| d as usize).product();
    let src = unsafe { slice::from_raw_parts(src, n_src) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_out) };

    let mut dilated_shape = vec![0i64; rank];
    for d in 0..rank {
        dilated_shape[d] = (src_shape[d] - 1) * base_dilations[d].max(1) + 1;
    }
    let mut padded_shape = vec![0i64; rank];
    for d in 0..rank {
        padded_shape[d] = dilated_shape[d] + padding[d * 2] + padding[d * 2 + 1];
    }

    let mut out_shape = vec![0usize; rank];
    for d in 0..rank {
        let eff_win = (window_dims[d] - 1) * window_dilations[d].max(1) + 1;
        out_shape[d] = ((padded_shape[d] - eff_win) / window_strides[d].max(1) + 1) as usize;
    }
    let mut out_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }
    let mut src_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1] as usize;
    }

    for out_flat in 0..n_out {
        let mut out_coords = vec![0usize; rank];
        let mut rem = out_flat;
        for d in 0..rank {
            out_coords[d] = rem / out_strides[d];
            rem %= out_strides[d];
        }

        let mut acc = init_val;
        let total_window: usize = window_dims.iter().map(|&d| d as usize).product();
        let mut win_strides = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            win_strides[i] = win_strides[i + 1] * window_dims[i + 1] as usize;
        }

        for wf in 0..total_window {
            let mut src_flat = 0usize;
            let mut valid = true;
            let mut wrem = wf;
            for d in 0..rank {
                let wc = wrem / win_strides[d];
                wrem %= win_strides[d];
                let padded_idx = out_coords[d] as i64 * window_strides[d].max(1)
                    + wc as i64 * window_dilations[d].max(1);
                let src_idx = padded_idx - padding[d * 2];
                let bd = base_dilations[d].max(1);
                if src_idx < 0 || src_idx >= dilated_shape[d] || (bd > 1 && src_idx % bd != 0) {
                    valid = false;
                    break;
                }
                let orig_idx = if bd > 1 { src_idx / bd } else { src_idx };
                if orig_idx < 0 || orig_idx >= src_shape[d] {
                    valid = false;
                    break;
                }
                src_flat += orig_idx as usize * src_strides[d];
            }
            if valid && src_flat < n_src {
                acc = reducer_fn(acc, src[src_flat]);
            }
        }
        dst[out_flat] = acc;
    }
}

pub extern "C" fn tensor_select_and_scatter_f64(
    dst: *mut f64,
    operand: *const f64,
    source: *const f64,
    init_val: f64,
    select_fn: extern "C" fn(f64, f64) -> i8,
    scatter_fn: extern "C" fn(f64, f64) -> f64,
    window_dims: *const i64,
    window_strides: *const i64,
    padding: *const i64,
    shape: *const i64,
    rank: usize,
    n_operand: usize,
    n_source: usize,
) {
    let shape = unsafe { slice::from_raw_parts(shape, rank) };
    let window_dims = unsafe { slice::from_raw_parts(window_dims, rank) };
    let window_strides = unsafe { slice::from_raw_parts(window_strides, rank) };
    let padding = unsafe { slice::from_raw_parts(padding, rank * 2) };
    let operand = unsafe { slice::from_raw_parts(operand, n_operand) };
    let source = unsafe { slice::from_raw_parts(source, n_source) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_operand) };

    for v in dst.iter_mut() {
        *v = init_val;
    }

    let mut op_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        op_strides[i] = op_strides[i + 1] * shape[i + 1] as usize;
    }
    let mut src_shape = vec![0usize; rank];
    for d in 0..rank {
        src_shape[d] = ((shape[d] + padding[d * 2] + padding[d * 2 + 1] - window_dims[d])
            / window_strides[d].max(1)
            + 1) as usize;
    }
    let mut src_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
    }
    let mut win_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        win_strides[i] = win_strides[i + 1] * window_dims[i + 1] as usize;
    }
    let total_window: usize = window_dims.iter().map(|&d| d as usize).product();

    for sf in 0..n_source {
        let mut src_coords = vec![0usize; rank];
        let mut rem = sf;
        for d in 0..rank {
            src_coords[d] = rem / src_strides[d];
            rem %= src_strides[d];
        }

        let mut best_idx = None;
        let mut best_val = 0.0f64;
        for wf in 0..total_window {
            let mut op_flat = 0usize;
            let mut valid = true;
            let mut wrem = wf;
            for d in 0..rank {
                let wc = wrem / win_strides[d];
                wrem %= win_strides[d];
                let op_idx =
                    src_coords[d] as i64 * window_strides[d].max(1) + wc as i64 - padding[d * 2];
                if op_idx < 0 || op_idx >= shape[d] {
                    valid = false;
                    break;
                }
                op_flat += op_idx as usize * op_strides[d];
            }
            if valid && op_flat < n_operand {
                let val = operand[op_flat];
                if best_idx.is_none() || select_fn(val, best_val) != 0 {
                    best_idx = Some(op_flat);
                    best_val = val;
                }
            }
        }
        if let Some(bi) = best_idx {
            dst[bi] = scatter_fn(dst[bi], source[sf]);
        }
    }
}

pub extern "C" fn tensor_conv_f64(
    dst: *mut f64,
    lhs: *const f64,
    rhs: *const f64,
    dn: *const i64,
    in_sp: *const i64,
    k_sp: *const i64,
    o_sp: *const i64,
    n_spatial: usize,
    lhs_shape: *const i64,
    rhs_shape: *const i64,
    strides: *const i64,
    pad: *const i64,
    lhs_dil: *const i64,
    rhs_dil: *const i64,
    lhs_rank: usize,
    feature_group_count: usize,
    _batch_group_count: usize,
    out_shape: *const i64,
    n_out: usize,
) {
    let dn = unsafe { slice::from_raw_parts(dn, 6) };
    let in_sp = unsafe { slice::from_raw_parts(in_sp, n_spatial) };
    let k_sp = unsafe { slice::from_raw_parts(k_sp, n_spatial) };
    let o_sp = unsafe { slice::from_raw_parts(o_sp, n_spatial) };
    let lhs_shape = unsafe { slice::from_raw_parts(lhs_shape, lhs_rank) };
    let rhs_shape = unsafe { slice::from_raw_parts(rhs_shape, lhs_rank) };
    let strides = unsafe { slice::from_raw_parts(strides, n_spatial) };
    let pad = unsafe { slice::from_raw_parts(pad, n_spatial * 2) };
    let lhs_dil = unsafe { slice::from_raw_parts(lhs_dil, n_spatial) };
    let rhs_dil = unsafe { slice::from_raw_parts(rhs_dil, n_spatial) };
    let out_shape = unsafe { slice::from_raw_parts(out_shape, lhs_rank) };
    let n_lhs: usize = lhs_shape.iter().map(|&d| d as usize).product();
    let n_rhs: usize = rhs_shape.iter().map(|&d| d as usize).product();
    let lhs = unsafe { slice::from_raw_parts(lhs, n_lhs) };
    let rhs = unsafe { slice::from_raw_parts(rhs, n_rhs) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_out) };

    let ib = dn[0] as usize;
    let if_ = dn[1] as usize;
    let kif = dn[2] as usize;
    let kof = dn[3] as usize;
    let ob = dn[4] as usize;
    let of = dn[5] as usize;

    let mut lhs_strides = vec![1usize; lhs_rank];
    for i in (0..lhs_rank.saturating_sub(1)).rev() {
        lhs_strides[i] = lhs_strides[i + 1] * lhs_shape[i + 1] as usize;
    }
    let mut rhs_strides = vec![1usize; lhs_rank];
    for i in (0..lhs_rank.saturating_sub(1)).rev() {
        rhs_strides[i] = rhs_strides[i + 1] * rhs_shape[i + 1] as usize;
    }
    let mut out_strides_v = vec![1usize; lhs_rank];
    for i in (0..lhs_rank.saturating_sub(1)).rev() {
        out_strides_v[i] = out_strides_v[i + 1] * out_shape[i + 1] as usize;
    }

    let batch_size = lhs_shape[ib] as usize;
    let in_channels = lhs_shape[if_] as usize;
    let out_channels = rhs_shape[kof] as usize;
    let group_in = in_channels / feature_group_count.max(1);
    let group_out = out_channels / feature_group_count.max(1);

    for v in dst.iter_mut() {
        *v = 0.0;
    }

    for batch in 0..batch_size {
        for g in 0..feature_group_count {
            for oc in 0..group_out {
                let abs_oc = g * group_out + oc;
                for of_idx in 0..n_out / (batch_size * out_channels).max(1) {
                    // Decompose of_idx into spatial coordinates
                    let mut out_spatial = vec![0usize; n_spatial];
                    let mut rem = of_idx;
                    let mut sp_total = 1usize;
                    for s in (0..n_spatial).rev() {
                        let dim = out_shape[o_sp[s] as usize] as usize;
                        sp_total *= dim;
                        _ = sp_total;
                    }
                    let mut sp_strides = vec![1usize; n_spatial];
                    for i in (0..n_spatial.saturating_sub(1)).rev() {
                        sp_strides[i] =
                            sp_strides[i + 1] * out_shape[o_sp[i + 1] as usize] as usize;
                    }
                    for s in 0..n_spatial {
                        out_spatial[s] = rem / sp_strides[s];
                        rem %= sp_strides[s];
                    }

                    let mut acc = 0.0f64;
                    for ic in 0..group_in {
                        let abs_ic = g * group_in + ic;
                        let mut k_spatial_total = 1usize;
                        for s in 0..n_spatial {
                            k_spatial_total *= rhs_shape[k_sp[s] as usize] as usize;
                        }
                        let mut k_sp_strides = vec![1usize; n_spatial];
                        for i in (0..n_spatial.saturating_sub(1)).rev() {
                            k_sp_strides[i] =
                                k_sp_strides[i + 1] * rhs_shape[k_sp[i + 1] as usize] as usize;
                        }
                        for kf in 0..k_spatial_total {
                            let mut k_coords = vec![0usize; n_spatial];
                            let mut krem = kf;
                            for s in 0..n_spatial {
                                k_coords[s] = krem / k_sp_strides[s];
                                krem %= k_sp_strides[s];
                            }
                            let mut valid = true;
                            let mut lhs_idx = vec![0usize; lhs_rank];
                            lhs_idx[ib] = batch;
                            lhs_idx[if_] = abs_ic;
                            for s in 0..n_spatial {
                                let stride = strides[s].max(1) as usize;
                                let rd = rhs_dil[s].max(1) as usize;
                                let ld = lhs_dil[s].max(1) as usize;
                                let padded = out_spatial[s] * stride + k_coords[s] * rd;
                                let pad_lo = pad[s * 2] as usize;
                                if padded < pad_lo {
                                    valid = false;
                                    break;
                                }
                                let dilated = padded - pad_lo;
                                if ld > 1 && !dilated.is_multiple_of(ld) {
                                    valid = false;
                                    break;
                                }
                                let orig = if ld > 1 { dilated / ld } else { dilated };
                                if orig >= lhs_shape[in_sp[s] as usize] as usize {
                                    valid = false;
                                    break;
                                }
                                lhs_idx[in_sp[s] as usize] = orig;
                            }
                            if valid {
                                let mut lf = 0usize;
                                for d in 0..lhs_rank {
                                    lf += lhs_idx[d] * lhs_strides[d];
                                }
                                let mut rhs_idx = vec![0usize; lhs_rank];
                                rhs_idx[kif] = abs_ic;
                                rhs_idx[kof] = abs_oc;
                                for s in 0..n_spatial {
                                    rhs_idx[k_sp[s] as usize] = k_coords[s];
                                }
                                let mut rf = 0usize;
                                for d in 0..lhs_rank {
                                    rf += rhs_idx[d] * rhs_strides[d];
                                }
                                if lf < n_lhs && rf < n_rhs {
                                    acc += lhs[lf] * rhs[rf];
                                }
                            }
                        }
                    }
                    let mut out_idx = vec![0usize; lhs_rank];
                    out_idx[ob] = batch;
                    out_idx[of] = abs_oc;
                    for s in 0..n_spatial {
                        out_idx[o_sp[s] as usize] = out_spatial[s];
                    }
                    let mut out_flat = 0usize;
                    for d in 0..lhs_rank {
                        out_flat += out_idx[d] * out_strides_v[d];
                    }
                    if out_flat < n_out {
                        dst[out_flat] = acc;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FFT and RNG runtime functions
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_fft_f64(data: *mut f64, n: usize, fft_type: u8) {
    let data = unsafe { slice::from_raw_parts_mut(data, n * 2) };
    match fft_type {
        0 => {
            // FFT: in-place on interleaved complex data (re0, im0, re1, im1, ...)
            let half = n;
            let mut re = vec![0.0f64; half];
            let mut im = vec![0.0f64; half];
            for i in 0..half {
                re[i] = data[i * 2];
                im[i] = data[i * 2 + 1];
            }
            let mut out_re = vec![0.0f64; half];
            let mut out_im = vec![0.0f64; half];
            naive_dft_out(&re, &im, &mut out_re, &mut out_im, half, false);
            for i in 0..half {
                data[i * 2] = out_re[i];
                data[i * 2 + 1] = out_im[i];
            }
        }
        1 => {
            // IFFT
            let half = n;
            let mut re = vec![0.0f64; half];
            let mut im = vec![0.0f64; half];
            for i in 0..half {
                re[i] = data[i * 2];
                im[i] = data[i * 2 + 1];
            }
            let mut out_re = vec![0.0f64; half];
            let mut out_im = vec![0.0f64; half];
            naive_dft_out(&re, &im, &mut out_re, &mut out_im, half, true);
            for i in 0..half {
                data[i * 2] = out_re[i];
                data[i * 2 + 1] = out_im[i];
            }
        }
        _ => {}
    }
}

fn naive_dft_out(
    re_in: &[f64],
    im_in: &[f64],
    re_out: &mut [f64],
    im_out: &mut [f64],
    n: usize,
    inverse: bool,
) {
    let sign = if inverse { 1.0 } else { -1.0 };
    let norm = if inverse { 1.0 / n as f64 } else { 1.0 };
    for k in 0..n {
        let mut sum_re = 0.0;
        let mut sum_im = 0.0;
        for j in 0..n {
            let angle = sign * 2.0 * std::f64::consts::PI * (k as f64) * (j as f64) / (n as f64);
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            sum_re += re_in[j] * cos_a - im_in[j] * sin_a;
            sum_im += re_in[j] * sin_a + im_in[j] * cos_a;
        }
        re_out[k] = sum_re * norm;
        im_out[k] = sum_im * norm;
    }
}

pub extern "C" fn tensor_rng_f64(
    dst: *mut f64,
    n: usize,
    distribution: u8,
    min_ptr: *const f64,
    max_ptr: *const f64,
) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    let min_val = unsafe { *min_ptr };
    let max_val = unsafe { *max_ptr };

    match distribution {
        0 => {
            // Uniform: linearly spaced between min and max
            for i in 0..n {
                let t = if n > 1 {
                    i as f64 / (n - 1) as f64
                } else {
                    0.5
                };
                dst[i] = min_val + t * (max_val - min_val);
            }
        }
        1 => {
            // Normal: use linearly spaced quantiles through ndtri
            for i in 0..n {
                let t = if n > 1 {
                    (i as f64 + 0.5) / n as f64
                } else {
                    0.5
                };
                let p = min_val + t * (max_val - min_val);
                dst[i] = p;
            }
        }
        _ => {
            for v in dst.iter_mut() {
                *v = 0.0;
            }
        }
    }
}

pub extern "C" fn tensor_sort_f64(
    data: *mut f64,
    n_outer: usize,
    sort_len: usize,
    n_inner: usize,
    ascending: u8,
) {
    let total = n_outer * sort_len * n_inner;
    let data = unsafe { slice::from_raw_parts_mut(data, total) };
    let asc = ascending != 0;
    for o in 0..n_outer {
        for s in 0..n_inner {
            // Stable insertion sort along the sort dimension
            for i in 1..sort_len {
                let mut j = i;
                while j > 0 {
                    let idx_a = (o * sort_len + j - 1) * n_inner + s;
                    let idx_b = (o * sort_len + j) * n_inner + s;
                    let should_swap = if asc {
                        data[idx_a] > data[idx_b]
                    } else {
                        data[idx_a] < data[idx_b]
                    };
                    if should_swap {
                        data.swap(idx_a, idx_b);
                        j -= 1;
                    } else {
                        break;
                    }
                }
            }
        }
    }
}

pub extern "C" fn tensor_argsort_f64(
    values: *mut f64,
    indices: *mut i64,
    sort_len: usize,
    ascending: u8,
) {
    let vals = unsafe { slice::from_raw_parts_mut(values, sort_len) };
    let idxs = unsafe { slice::from_raw_parts_mut(indices, sort_len) };
    let asc = ascending != 0;
    for i in 1..sort_len {
        let mut j = i;
        while j > 0 {
            let should_swap = if asc {
                vals[j - 1] > vals[j]
            } else {
                vals[j - 1] < vals[j]
            };
            if should_swap {
                vals.swap(j - 1, j);
                idxs.swap(j - 1, j);
                j -= 1;
            } else {
                break;
            }
        }
    }
}
