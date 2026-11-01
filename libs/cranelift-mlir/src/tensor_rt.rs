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

/// IEEE 754-2019 maximum, which XLA's `max` computes: a NaN operand wins and +0 is above -0.
pub(crate) fn float_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        a + b
    } else if a == b {
        f64::from_bits(a.to_bits() & b.to_bits())
    } else {
        a.max(b)
    }
}

/// IEEE 754-2019 minimum, which XLA's `min` computes: a NaN operand wins and -0 is below +0.
pub(crate) fn float_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        a + b
    } else if a == b {
        f64::from_bits(a.to_bits() | b.to_bits())
    } else {
        a.min(b)
    }
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
// `wide::f64x2` has no remainder operator, and its max/min drop NaNs; those
// three keep the scalar loop.
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
binary_f64_op!(tensor_max_f64, float_max);
binary_f64_op!(tensor_min_f64, float_min);
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
        dst[i] = a[i].checked_div(b[i]).unwrap_or(0);
    }
}

pub extern "C" fn tensor_div_i32(dst: *mut i32, a: *const i32, b: *const i32, n: usize) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
    for i in 0..n {
        dst[i] = a[i].checked_div(b[i]).unwrap_or(0);
    }
}

pub extern "C" fn tensor_div_ui32(dst: *mut u32, a: *const u32, b: *const u32, n: usize) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
    for i in 0..n {
        dst[i] = a[i].checked_div(b[i]).unwrap_or(0);
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

unary_f64_op!(tensor_erfc_f64, libm::erfc);

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
        let r = va.powf_simd(vb).to_array();
        dst[i] = r[0];
        dst[i + 1] = r[1];
        i += 2;
    }
    while i < n {
        dst[i] = a[i].powf(b[i]);
        i += 1;
    }
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
convert_op!(tensor_convert_f64_to_ui64, f64, u64, |x: f64| x as u64);
convert_op!(tensor_convert_i1_to_f64, u8, f64, |x: u8| if x != 0 {
    1.0
} else {
    0.0
});
convert_op!(tensor_convert_f64_to_i32, f64, i32, |x: f64| x as i32);
convert_op!(tensor_convert_i1_to_i32, u8, i32, |x: u8| x as i32);
convert_op!(tensor_convert_i1_to_i64, u8, i64, |x: u8| x as i64);
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
// Reduce one axis: src viewed as [outer, inner, after], reduced over inner
// ---------------------------------------------------------------------------

macro_rules! reduce_op {
    ($name:ident, $ty:ty, $init:expr, $update:expr) => {
        pub extern "C" fn $name(
            dst: *mut $ty,
            src: *const $ty,
            outer: usize,
            inner: usize,
            after: usize,
        ) {
            let src = unsafe { slice::from_raw_parts(src, outer * inner * after) };
            let dst = unsafe { slice::from_raw_parts_mut(dst, outer * after) };
            if after == 1 {
                for (o, out) in dst.iter_mut().enumerate() {
                    let mut acc: $ty = $init;
                    for &v in &src[o * inner..(o + 1) * inner] {
                        acc = ($update)(acc, v);
                    }
                    *out = acc;
                }
                return;
            }
            for (o, out) in dst.chunks_exact_mut(after).enumerate() {
                out.fill($init);
                for row in src[o * inner * after..(o + 1) * inner * after].chunks_exact(after) {
                    for (acc, &v) in out.iter_mut().zip(row) {
                        *acc = ($update)(*acc, v);
                    }
                }
            }
        }
    };
}

reduce_op!(tensor_reduce_sum_f64, f64, 0.0, |acc: f64, v: f64| acc
    .algebraic_add(v));
reduce_op!(tensor_reduce_prod_f64, f64, 1.0, |acc: f64, v: f64| acc * v);
reduce_op!(tensor_reduce_max_f64, f64, f64::NEG_INFINITY, float_max);
reduce_op!(tensor_reduce_min_f64, f64, f64::INFINITY, float_min);
reduce_op!(tensor_reduce_sum_i64, i64, 0, |acc: i64, v: i64| acc
    .wrapping_add(v));
reduce_op!(tensor_reduce_prod_i64, i64, 1, |acc: i64, v: i64| acc
    .wrapping_mul(v));
reduce_op!(tensor_reduce_max_i64, i64, i64::MIN, |acc: i64, v: i64| acc
    .max(v));
reduce_op!(tensor_reduce_min_i64, i64, i64::MAX, |acc: i64, v: i64| acc
    .min(v));
reduce_op!(tensor_reduce_max_u64, u64, 0, |acc: u64, v: u64| acc.max(v));
reduce_op!(tensor_reduce_min_u64, u64, u64::MAX, |acc: u64, v: u64| acc
    .min(v));
reduce_op!(tensor_reduce_sum_f32, f32, 0.0, |acc: f32, v: f32| acc
    .algebraic_add(v));
reduce_op!(tensor_reduce_prod_f32, f32, 1.0, |acc: f32, v: f32| acc * v);
reduce_op!(
    tensor_reduce_max_f32,
    f32,
    f32::NEG_INFINITY,
    |acc: f32, v: f32| float_max(acc.into(), v.into()) as f32
);
reduce_op!(
    tensor_reduce_min_f32,
    f32,
    f32::INFINITY,
    |acc: f32, v: f32| float_min(acc.into(), v.into()) as f32
);
reduce_op!(tensor_reduce_sum_i32, i32, 0, |acc: i32, v: i32| acc
    .wrapping_add(v));
reduce_op!(tensor_reduce_prod_i32, i32, 1, |acc: i32, v: i32| acc
    .wrapping_mul(v));
reduce_op!(tensor_reduce_max_i32, i32, i32::MIN, |acc: i32, v: i32| acc
    .max(v));
reduce_op!(tensor_reduce_min_i32, i32, i32::MAX, |acc: i32, v: i32| acc
    .min(v));
reduce_op!(tensor_reduce_max_u32, u32, 0, |acc: u32, v: u32| acc.max(v));
reduce_op!(tensor_reduce_min_u32, u32, u32::MAX, |acc: u32, v: u32| acc
    .min(v));
reduce_op!(tensor_reduce_and_i1, u8, 1, |acc: u8, v: u8| acc
    & u8::from(v != 0));
reduce_op!(tensor_reduce_or_i1, u8, 0, |acc: u8, v: u8| acc
    | u8::from(v != 0));

// ---------------------------------------------------------------------------
// Argmin/argmax: multi-operand reduce producing (value, index)
// ---------------------------------------------------------------------------

trait ArgValue: Copy + PartialOrd {
    fn nan(self) -> bool {
        false
    }
}
impl ArgValue for f64 {
    fn nan(self) -> bool {
        self.is_nan()
    }
}
impl ArgValue for f32 {
    fn nan(self) -> bool {
        self.is_nan()
    }
}
impl ArgValue for i64 {}
impl ArgValue for i32 {}
impl ArgValue for u64 {}
impl ArgValue for u32 {}
impl ArgValue for u8 {}

fn argminmax<V: ArgValue, I: Copy + PartialOrd>(
    [dst_v, dst_i, src_v, src_i]: [*mut u8; 4],
    is_min: bool,
    [outer, inner, after]: [usize; 3],
) {
    if inner == 0 {
        return;
    }
    let n = outer * inner * after;
    let src_v = unsafe { slice::from_raw_parts(src_v.cast::<V>(), n) };
    let src_i = unsafe { slice::from_raw_parts(src_i.cast::<I>(), n) };
    let dst_v = unsafe { slice::from_raw_parts_mut(dst_v.cast::<V>(), outer * after) };
    let dst_i = unsafe { slice::from_raw_parts_mut(dst_i.cast::<I>(), outer * after) };
    for o in 0..outer {
        for a in 0..after {
            let at = |k: usize| (o * inner + k) * after + a;
            let mut best = at(0);
            for p in (1..inner).map(at) {
                let (b, v) = (src_v[best], src_v[p]);
                let better = if is_min { v < b } else { v > b };
                if !b.nan() && (v.nan() || better || (v == b && src_i[p] < src_i[best])) {
                    best = p;
                }
            }
            dst_v[o * after + a] = src_v[best];
            dst_i[o * after + a] = src_i[best];
        }
    }
}

/// Argmin or argmax over one axis, with values and indices viewed as [outer, axis, after].
/// `meta` is `[value_code, index_size, is_min, outer, axis, after]`; value codes are 0 f64,
/// 1 f32, 2 i64, 3 i32, 4 u64, 5 u32, 6 i1. Like jax-cpu, the first NaN wins and ties keep
/// the smaller index.
pub extern "C" fn tensor_argminmax(
    dst_v: *mut u8,
    dst_i: *mut u8,
    src_v: *const u8,
    src_i: *const u8,
    meta: *const i64,
) {
    let meta = unsafe { slice::from_raw_parts(meta, 6) };
    let ptrs = [dst_v, dst_i, src_v.cast_mut(), src_i.cast_mut()];
    let (is_min, dims) = (meta[2] != 0, [3, 4, 5].map(|i| meta[i] as usize));
    macro_rules! run {
        ($v:ty) => {
            if meta[1] == 4 {
                argminmax::<$v, i32>(ptrs, is_min, dims)
            } else {
                argminmax::<$v, i64>(ptrs, is_min, dims)
            }
        };
    }
    match meta[0] {
        0 => run!(f64),
        1 => run!(f32),
        2 => run!(i64),
        3 => run!(i32),
        4 => run!(u64),
        5 => run!(u32),
        _ => run!(u8),
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
// Scatter with StableHLO dimension numbers, in place
// ---------------------------------------------------------------------------

/// Scatters `updates` into `dst`, which already holds the operand. `meta` is
/// `[elem_sz, is_float, combiner, n_dst, n_indices, index_vector_stride, n_update_dims]`, then
/// `(size, operand_stride, indices_stride)` per updates dim, then `(operand_stride, max_start)`
/// per index-vector entry. Combiners other than set (0) are f64 or i64 only. Like XLA, an
/// update whose window starts out of bounds is dropped.
pub extern "C" fn tensor_scatter(
    dst: *mut u8,
    indices: *const i64,
    updates: *const u8,
    meta: *const i64,
    n_meta: usize,
) {
    let meta = unsafe { slice::from_raw_parts(meta, n_meta) };
    let [
        elem_sz,
        is_float,
        combiner,
        n_dst,
        n_idx,
        vec_stride,
        n_dims,
    ] = [0, 1, 2, 3, 4, 5, 6].map(|i| meta[i] as usize);
    let (upd_dims, starts) = meta[7..].split_at(3 * n_dims);
    let (upd_dims, starts) = (upd_dims.as_chunks::<3>().0, starts.as_chunks::<2>().0);
    let n_upd: usize = upd_dims.iter().map(|d| d[0] as usize).product();
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_dst * elem_sz) };
    let indices = unsafe { slice::from_raw_parts(indices, n_idx) };
    let updates = unsafe { slice::from_raw_parts(updates, n_upd * elem_sz) };

    'updates: for u in 0..n_upd {
        let (mut flat, mut idx, mut rem) = (0, 0, u);
        for &[size, operand_stride, indices_stride] in upd_dims.iter().rev() {
            let c = rem % size as usize;
            rem /= size as usize;
            flat += c * operand_stride as usize;
            idx += c * indices_stride as usize;
        }
        for (k, &[operand_stride, max_start]) in starts.iter().enumerate() {
            let start = indices[idx + k * vec_stride];
            if start < 0 || start > max_start {
                continue 'updates;
            }
            flat += start as usize * operand_stride as usize;
        }
        let out = &mut dst[flat * elem_sz..(flat + 1) * elem_sz];
        let upd = &updates[u * elem_sz..(u + 1) * elem_sz];
        if combiner == 0 {
            out.copy_from_slice(upd);
        } else if is_float != 0 {
            let (a, b) = (
                f64::from_ne_bytes(out.try_into().unwrap()),
                f64::from_ne_bytes(upd.try_into().unwrap()),
            );
            let r = match combiner {
                1 => a + b,
                2 => a * b,
                3 => float_min(a, b),
                _ => float_max(a, b),
            };
            out.copy_from_slice(&r.to_ne_bytes());
        } else {
            let (a, b) = (
                i64::from_ne_bytes(out.try_into().unwrap()),
                i64::from_ne_bytes(upd.try_into().unwrap()),
            );
            let r = match combiner {
                1 => a.wrapping_add(b),
                2 => a.wrapping_mul(b),
                3 => a.min(b),
                _ => a.max(b),
            };
            out.copy_from_slice(&r.to_ne_bytes());
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
    // ikj keeps B and C rows contiguous so the inner loop can vectorize.
    for i in 0..m {
        let dst_row = &mut dst[i * n..][..n];
        dst_row.fill(0.0);
        for p in 0..k {
            let a_ip = a[i * k + p];
            let b_row = &b[p * n..][..n];
            for (d, &bv) in dst_row.iter_mut().zip(b_row) {
                *d = d.algebraic_add(a_ip.algebraic_mul(bv));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// General dot: any batch, free and contracting axes
// ---------------------------------------------------------------------------

/// Multiply-accumulate in the element type's own arithmetic; integers wrap.
trait DotValue: Copy + Default {
    fn add_product(self, a: Self, b: Self) -> Self;
}
impl DotValue for f64 {
    fn add_product(self, a: Self, b: Self) -> Self {
        self + a * b
    }
}
impl DotValue for f32 {
    fn add_product(self, a: Self, b: Self) -> Self {
        self + a * b
    }
}
impl DotValue for i64 {
    fn add_product(self, a: Self, b: Self) -> Self {
        self.wrapping_add(a.wrapping_mul(b))
    }
}
impl DotValue for i32 {
    fn add_product(self, a: Self, b: Self) -> Self {
        self.wrapping_add(a.wrapping_mul(b))
    }
}

fn dot_general<T: DotValue>(
    [dst, lhs, rhs]: [*mut u8; 3],
    [n_lhs, n_rhs]: [usize; 2],
    out_axes: &[[usize; 3]],
    k_axes: &[[usize; 3]],
) {
    let n: usize = out_axes.iter().map(|a| a[0]).product();
    let n_k: usize = k_axes.iter().map(|a| a[0]).product();
    let lhs = unsafe { slice::from_raw_parts(lhs.cast::<T>(), n_lhs) };
    let rhs = unsafe { slice::from_raw_parts(rhs.cast::<T>(), n_rhs) };
    let dst = unsafe { slice::from_raw_parts_mut(dst.cast::<T>(), n) };
    let offsets = |axes: &[[usize; 3]], mut flat: usize| {
        let (mut l, mut r) = (0, 0);
        for &[size, l_stride, r_stride] in axes.iter().rev() {
            l += flat % size * l_stride;
            r += flat % size * r_stride;
            flat /= size;
        }
        (l, r)
    };
    for (o, d) in dst.iter_mut().enumerate() {
        let (lo, ro) = offsets(out_axes, o);
        *d = (0..n_k).fold(T::default(), |acc, k| {
            let (lk, rk) = offsets(k_axes, k);
            acc.add_product(lhs[lo + lk], rhs[ro + rk])
        });
    }
}

/// `dot_general` with each axis given as `[size, lhs stride, rhs stride]`, output axes
/// (batch, lhs free, then rhs free) before contracting axes. `meta` is `[value_code,
/// lhs_len, rhs_len, n_output_axes, n_contracting_axes, axes...]`, with the value codes of
/// `tensor_argminmax`; unsigned values use the wrapping signed kernels.
pub extern "C" fn tensor_dot_general(
    dst: *mut u8,
    lhs: *const u8,
    rhs: *const u8,
    meta: *const i64,
) {
    let head = unsafe { slice::from_raw_parts(meta, 5) };
    let [code, n_lhs, n_rhs, n_out, n_k] = [0, 1, 2, 3, 4].map(|i| head[i] as usize);
    let axes = unsafe { slice::from_raw_parts(meta.add(5), 3 * (n_out + n_k)) };
    let axes: Vec<[usize; 3]> = axes
        .as_chunks::<3>()
        .0
        .iter()
        .map(|a| a.map(|v| v as usize))
        .collect();
    let (out_axes, k_axes) = axes.split_at(n_out);
    let ptrs = [dst, lhs.cast_mut(), rhs.cast_mut()];
    match code {
        0 => dot_general::<f64>(ptrs, [n_lhs, n_rhs], out_axes, k_axes),
        1 => dot_general::<f32>(ptrs, [n_lhs, n_rhs], out_axes, k_axes),
        2 | 4 => dot_general::<i64>(ptrs, [n_lhs, n_rhs], out_axes, k_axes),
        _ => dot_general::<i32>(ptrs, [n_lhs, n_rhs], out_axes, k_axes),
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

/// Clamp a dynamic-slice start into `[0, max_start]`. Negative starts become 0.
fn clamped_start(start: i64, max_start: i64) -> usize {
    let max_start = max_start.max(0);
    if start <= 0 {
        0
    } else if start >= max_start {
        max_start as usize
    } else {
        start as usize
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
            let start = clamped_start(start_indices[d], shape[d] - slice_sizes[d]);
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
            let start = clamped_start(start_indices[d], shape[d] - update_shape[d]);
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

/// Registered once per value width; unsigned values use the signed instance, since the
/// compiled reducer does the arithmetic.
pub extern "C" fn tensor_reduce_window<T: Copy>(
    dst: *mut T,
    src: *const T,
    init_val: T,
    window_dims: *const i64,
    window_strides: *const i64,
    base_dilations: *const i64,
    window_dilations: *const i64,
    padding: *const i64,
    src_shape: *const i64,
    rank: usize,
    n_out: usize,
    reducer_fn: extern "C" fn(T, T) -> T,
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

/// Registered once per value width, like `tensor_reduce_window`.
pub extern "C" fn tensor_select_and_scatter<T: Copy>(
    dst: *mut T,
    operand: *const T,
    source: *const T,
    init_val: T,
    select_fn: extern "C" fn(T, T) -> i8,
    scatter_fn: extern "C" fn(T, T) -> T,
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

        let mut best: Option<(usize, T)> = None;
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
                // Like XLA: `select(current, candidate)` true keeps the current element.
                if best.is_none_or(|(_, current)| select_fn(current, val) == 0) {
                    best = Some((op_flat, val));
                }
            }
        }
        if let Some((bi, _)) = best {
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
                                    acc = acc.algebraic_add(lhs[lf].algebraic_mul(rhs[rf]));
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
        let mut sum_re = 0.0f64;
        let mut sum_im = 0.0f64;
        for j in 0..n {
            let angle = sign * 2.0 * std::f64::consts::PI * (k as f64) * (j as f64) / (n as f64);
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            sum_re = sum_re
                .algebraic_add(re_in[j].algebraic_mul(cos_a))
                .algebraic_sub(im_in[j].algebraic_mul(sin_a));
            sum_im = sum_im
                .algebraic_add(re_in[j].algebraic_mul(sin_a))
                .algebraic_add(im_in[j].algebraic_mul(cos_a));
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

/// One sort key, read in place.
enum SortKey<'a> {
    F64(&'a [f64]),
    F32(&'a [f32]),
    I64(&'a [i64]),
    I32(&'a [i32]),
    U64(&'a [u64]),
    U32(&'a [u32]),
    U8(&'a [u8]),
}

impl SortKey<'_> {
    fn cmp(&self, i: usize, j: usize) -> std::cmp::Ordering {
        // JAX's comparators treat -0 as 0 and put NaN after everything.
        let float = |a: f64, b: f64| {
            a.partial_cmp(&b)
                .unwrap_or_else(|| a.is_nan().cmp(&b.is_nan()))
        };
        match self {
            SortKey::F64(v) => float(v[i], v[j]),
            SortKey::F32(v) => float(v[i] as f64, v[j] as f64),
            SortKey::I64(v) => v[i].cmp(&v[j]),
            SortKey::I32(v) => v[i].cmp(&v[j]),
            SortKey::U64(v) => v[i].cmp(&v[j]),
            SortKey::U32(v) => v[i].cmp(&v[j]),
            SortKey::U8(v) => v[i].cmp(&v[j]),
        }
    }
}

/// Stable sort of several operands along one axis, each viewed as [outer, axis, after].
/// Orders lexicographically by the first `n_keys` operands, ascending unless
/// `descending`, and applies the same permutation to every operand. `meta` is
/// `[n_ops, n_keys, descending, outer, axis, after]` then a value code per operand
/// (0 f64, 1 f32, 2 i64, 3 i32, 4 u64, 5 u32, 6 i1).
pub extern "C" fn tensor_sort(data: *const *mut u8, meta: *const i64) {
    let head = unsafe { slice::from_raw_parts(meta, 6) };
    let [n_ops, n_keys, descending, outer, len, after] =
        [0, 1, 2, 3, 4, 5].map(|i| head[i] as usize);
    let codes = unsafe { slice::from_raw_parts(meta.add(6), n_ops) };
    let ptrs = unsafe { slice::from_raw_parts(data, n_ops) };
    let n = outer * len * after;
    let sizes: Vec<usize> = codes
        .iter()
        .map(|&c| [8, 4, 8, 4, 8, 4, 1][c as usize])
        .collect();
    let (mut perm, mut scratch) = (Vec::with_capacity(len), Vec::new());
    for o in 0..outer {
        for a in 0..after {
            let at = |k: usize| (o * len + k) * after + a;
            let keys: Vec<SortKey> = (0..n_keys)
                .map(|k| unsafe {
                    let p = ptrs[k];
                    match codes[k] {
                        0 => SortKey::F64(slice::from_raw_parts(p.cast(), n)),
                        1 => SortKey::F32(slice::from_raw_parts(p.cast(), n)),
                        2 => SortKey::I64(slice::from_raw_parts(p.cast(), n)),
                        3 => SortKey::I32(slice::from_raw_parts(p.cast(), n)),
                        4 => SortKey::U64(slice::from_raw_parts(p.cast(), n)),
                        5 => SortKey::U32(slice::from_raw_parts(p.cast(), n)),
                        _ => SortKey::U8(slice::from_raw_parts(p, n)),
                    }
                })
                .collect();
            perm.clear();
            perm.extend(0..len);
            perm.sort_by(|&x, &y| {
                let ord = keys
                    .iter()
                    .map(|key| key.cmp(at(x), at(y)))
                    .find(|ord| ord.is_ne())
                    .unwrap_or(std::cmp::Ordering::Equal);
                if descending != 0 { ord.reverse() } else { ord }
            });
            drop(keys);
            for (&p, &sz) in ptrs.iter().zip(&sizes) {
                let buf = unsafe { slice::from_raw_parts_mut(p, n * sz) };
                scratch.clear();
                for &k in &perm {
                    scratch.extend_from_slice(&buf[at(k) * sz..(at(k) + 1) * sz]);
                }
                for (k, elem) in scratch.chunks_exact(sz).enumerate() {
                    buf[at(k) * sz..(at(k) + 1) * sz].copy_from_slice(elem);
                }
            }
        }
    }
}
