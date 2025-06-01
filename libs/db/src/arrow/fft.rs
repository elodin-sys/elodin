use std::any::Any;
use std::sync::Arc;

use arrow::array::{Array, Float64Array, TimestampMicrosecondArray};
use arrow::datatypes::DataType;
use arrow_schema::TimeUnit;
use datafusion::common::Result as DataFusionResult;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use rustfft::{FftPlanner, num_complex::Complex};

#[derive(Debug)]
pub struct FrequencyDomainUDF {
    signature: Signature,
}

impl FrequencyDomainUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![DataType::Timestamp(TimeUnit::Microsecond, None)],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for FrequencyDomainUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "fftfreq"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DataFusionResult<DataType> {
        Ok(DataType::Float64)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DataFusionResult<ColumnarValue> {
        let input = match &args.args[0] {
            ColumnarValue::Array(array) => array.clone(),
            ColumnarValue::Scalar(_scalar) => {
                return Err(datafusion::error::DataFusionError::Internal(
                    "frequency_domain function expects array input".to_string(),
                ));
            }
        };

        let input = input
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Internal(
                    "Expected TimestampMicrosecondArray input".to_string(),
                )
            })?;
        let (count, sum) = input.values()[..]
            .windows(2)
            .map(|window| {
                let a = window[0] as f64;
                let b = window[1] as f64;
                (b - a) * 1e-6
            })
            .fold((0, 0.0), |(count, sum), x| (count + 1, sum + x));
        let period = sum / count as f64;

        let n = input.len();

        if n == 0 {
            return Ok(ColumnarValue::Array(Arc::new(Float64Array::new_null(0))));
        }

        let sample_rate = 1. / period;
        let n = if n % 2 == 0 { n } else { n - 1 };
        let frequencies: Float64Array = (0..n / 2)
            .rev()
            .map(|i| i as f64 * -sample_rate / n as f64)
            .chain((0..n / 2).map(|i| i as f64 * sample_rate / n as f64))
            .chain(std::iter::repeat(0f64))
            .take(input.len())
            .collect();

        Ok(ColumnarValue::Array(Arc::new(frequencies)))
    }
}

#[derive(Debug)]
pub struct FftUDF {
    signature: Signature,
}

impl FftUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(vec![DataType::Float64], Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for FftUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "fft"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DataFusionResult<DataType> {
        Ok(DataType::Float64)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DataFusionResult<ColumnarValue> {
        let input = match &args.args[0] {
            ColumnarValue::Array(array) => array.clone(),
            ColumnarValue::Scalar(_scalar) => {
                return Err(datafusion::error::DataFusionError::Internal(
                    "fft_magnitudes function expects array input".to_string(),
                ));
            }
        };

        let input = input
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Internal(
                    "Expected Float64Array input".to_string(),
                )
            })?;

        let array = compute_fft_magnitudes(input.values());
        Ok(ColumnarValue::Array(Arc::new(array)))
    }
}

fn compute_fft_magnitudes(input: &[f64]) -> Float64Array {
    if input.is_empty() {
        return Float64Array::from(vec![0f64; 0]);
    }

    let mut buffer: Vec<Complex<f64>> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(input.len());
    fft.process(&mut buffer);

    let mut mags: Vec<_> = buffer.iter().map(|c| c.norm()).collect();
    let shift = mags.len() / 2;
    mags.rotate_right(shift);
    Float64Array::from(mags)
}
