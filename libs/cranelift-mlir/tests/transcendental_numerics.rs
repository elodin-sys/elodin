//! The SIMD transcendentals come from `wide`, whose results may change between its versions and
//! platforms by less than any other test's tolerance, yet enough to move long simulations.

use cranelift_mlir::tensor_rt as rt;

type Unary = extern "C" fn(*mut f64, *const f64, usize);
type Binary = extern "C" fn(*mut f64, *const f64, *const f64, usize);

/// Even, so every element takes the SIMD pairs rather than the scalar tail.
const N: usize = 1024;

/// Most units in the last place a `wide` kernel may be from std, itself within about one of exact.
const MAX_ULPS: u64 = 4;

struct Case {
    name: &'static str,
    out: Vec<f64>,
    /// std's results, for the kernels that are `wide`'s own rather than built from them.
    reference: Option<Vec<f64>>,
}

fn linear(lo: f64, hi: f64) -> Vec<f64> {
    (0..N)
        .map(|i| lo + (hi - lo) * (i as f64 / (N - 1) as f64))
        .collect()
}

/// Magnitudes from 2^lo up to 2^hi, built exactly so every platform hashes the same inputs.
fn logarithmic(lo: i32, hi: i32) -> Vec<f64> {
    let span = (hi - lo) as usize;
    (0..N)
        .map(|i| {
            let mantissa = 1.0 + (i * span % N) as f64 / N as f64;
            let exponent = lo + (i * span / N) as i32;
            mantissa * f64::from_bits(((1023 + exponent) as u64) << 52)
        })
        .collect()
}

fn unary(
    name: &'static str,
    kernel: Unary,
    x: Vec<f64>,
    reference: Option<fn(f64) -> f64>,
) -> Case {
    let mut out = vec![0.0; N];
    kernel(out.as_mut_ptr(), x.as_ptr(), N);
    let reference = reference.map(|f| x.iter().map(|&v| f(v)).collect());
    Case {
        name,
        out,
        reference,
    }
}

fn binary(
    name: &'static str,
    kernel: Binary,
    a: Vec<f64>,
    b: Vec<f64>,
    reference: fn(f64, f64) -> f64,
) -> Case {
    let mut out = vec![0.0; N];
    kernel(out.as_mut_ptr(), a.as_ptr(), b.as_ptr(), N);
    let reference = a.iter().zip(&b).map(|(&x, &y)| reference(x, y)).collect();
    Case {
        name,
        out,
        reference: Some(reference),
    }
}

fn cases() -> Vec<Case> {
    let y = linear(-3.0, 3.0);
    // 389 is coprime with N, so this pairs a permutation of the same values with every y.
    let x = (0..N).map(|i| y[i * 389 % N]).collect();
    // Includes the barometric-formula exponents 5.2559 and 0.190263.
    let exponents = [0.5, 1.5, 2.0, 3.0, -1.0, -0.5, 5.2559, 0.190263];
    let powers = (0..N).map(|i| exponents[i % exponents.len()]).collect();
    vec![
        unary(
            "sin",
            rt::tensor_sin_f64,
            linear(-50.0, 50.0),
            Some(f64::sin),
        ),
        unary(
            "cos",
            rt::tensor_cos_f64,
            linear(-50.0, 50.0),
            Some(f64::cos),
        ),
        unary("tan", rt::tensor_tan_f64, linear(-1.5, 1.5), Some(f64::tan)),
        unary(
            "asin",
            rt::tensor_asin_f64,
            linear(-1.0, 1.0),
            Some(f64::asin),
        ),
        unary(
            "acos",
            rt::tensor_acos_f64,
            linear(-1.0, 1.0),
            Some(f64::acos),
        ),
        unary(
            "atan",
            rt::tensor_atan_f64,
            linear(-20.0, 20.0),
            Some(f64::atan),
        ),
        unary(
            "exp",
            rt::tensor_exp_f64,
            linear(-50.0, 50.0),
            Some(f64::exp),
        ),
        unary(
            "log",
            rt::tensor_log_f64,
            logarithmic(-30, 30),
            Some(f64::ln),
        ),
        binary("atan2", rt::tensor_atan2_f64, y, x, f64::atan2),
        binary(
            "pow",
            rt::tensor_pow_f64,
            logarithmic(-10, 10),
            powers,
            f64::powf,
        ),
        unary("sinh", rt::tensor_sinh_f64, linear(-10.0, 10.0), None),
        unary("cosh", rt::tensor_cosh_f64, linear(-10.0, 10.0), None),
        unary("tanh", rt::tensor_tanh_f64, linear(-10.0, 10.0), None),
        unary("expm1", rt::tensor_expm1_f64, linear(-5.0, 5.0), None),
        unary("log1p", rt::tensor_log1p_f64, linear(-0.9, 10.0), None),
    ]
}

/// Distance in units in the last place; 0 for equal values, including +0 and -0.
fn ulps(a: f64, b: f64) -> u64 {
    if a == b {
        return 0;
    }
    let ordered = |v: f64| {
        let bits = v.to_bits() as i64;
        if bits < 0 { i64::MIN - bits } else { bits }
    };
    ordered(a).abs_diff(ordered(b))
}

#[test]
fn wide_kernels_stay_close_to_std() {
    let mut too_far = Vec::new();
    for case in cases() {
        let Some(reference) = &case.reference else {
            continue;
        };
        let worst = case
            .out
            .iter()
            .zip(reference)
            .map(|(&a, &b)| ulps(a, b))
            .max()
            .unwrap_or(0);
        println!("{:>5}: {worst} ulp from std", case.name);
        // pow's error grows with |y ln x|; it reaches 16 ulp at 34.
        let bound = if case.name == "pow" { 20 } else { MAX_ULPS };
        if worst > bound {
            too_far.push(format!("{} ({worst} ulp, bound {bound})", case.name));
        }
    }
    assert!(too_far.is_empty(), "too far from std: {too_far:?}");
}

#[cfg(all(target_arch = "x86_64", not(target_feature = "fma")))]
mod snapshot {
    use super::cases;

    /// blake3 of each kernel's output bits. `wide` fuses multiply-adds where FMA is available,
    /// which changes the bits, so these hold for x86_64 without it.
    const HASHES: [(&str, &str); 15] = [
        ("sin", "75eec4d56b18d068"),
        ("cos", "5aafc8e077ea05e1"),
        ("tan", "a6ff53a3459ef1eb"),
        ("asin", "d3ac9474eabf5c85"),
        ("acos", "6d818362a5abe7fb"),
        ("atan", "d5ef472fa92b491c"),
        ("exp", "1980442bd8dd1396"),
        ("log", "2cf813d0bb5a638d"),
        ("atan2", "04f92707ab15770e"),
        ("pow", "cd15e6d9f10e9904"),
        ("sinh", "b1913114e16b7f20"),
        ("cosh", "0fb0bf3998d6e795"),
        ("tanh", "50a0ecb330fce735"),
        ("expm1", "84d04d2f1adcdf28"),
        ("log1p", "03bdb17e730b4f2b"),
    ];

    #[test]
    fn transcendental_bits_are_unchanged() {
        let cases = cases();
        assert_eq!(cases.len(), HASHES.len(), "every case needs a hash");
        let changed: Vec<String> = cases
            .iter()
            .zip(HASHES)
            .filter_map(|(case, (name, hash))| {
                assert_eq!(case.name, name);
                let mut hasher = blake3::Hasher::new();
                for v in &case.out {
                    hasher.update(&v.to_le_bytes());
                }
                let got = hasher.finalize().to_hex()[..16].to_string();
                (got != hash).then(|| format!("(\"{name}\", \"{got}\")"))
            })
            .collect();
        assert!(
            changed.is_empty(),
            "SIMD transcendental results changed, which moves long simulations. If that is \
             intended, check the ulp table of wide_kernels_stay_close_to_std (--nocapture), \
             update HASHES to {changed:?} and add a CHANGELOG entry."
        );
    }
}
