//! LTTB (Largest Triangle Three Buckets) downsampling algorithm
//!
//! This algorithm downsamples time series data while preserving visual characteristics.
//! It's the industry standard for perceptually-optimal line chart downsampling.
//!
//! Reference: Sveinn Steinarsson, "Downsampling Time Series for Visual Representation"
//! https://skemman.is/bitstream/1946/15343/3/SS_MSthesis.pdf

/// A single data point with time and value
#[derive(Clone, Copy, Debug)]
pub struct DataPoint {
    pub time: i64,
    pub value: f64,
}

/// Downsample a time series using the LTTB algorithm.
///
/// # Arguments
/// * `data` - The input data points, must be sorted by time
/// * `target_points` - The desired number of output points
///
/// # Returns
/// A vector of downsampled data points preserving visual fidelity
pub fn lttb_downsample(data: &[DataPoint], target_points: usize) -> Vec<DataPoint> {
    let n = data.len();

    // If we have fewer points than target, return all
    if n <= target_points || target_points < 3 {
        return data.to_vec();
    }

    let mut result = Vec::with_capacity(target_points);

    // Always include the first point
    result.push(data[0]);

    // Bucket size (excluding first and last points which are always included)
    let bucket_size = (n - 2) as f64 / (target_points - 2) as f64;

    let mut a_index = 0; // Index of the previously selected point

    for i in 0..(target_points - 2) {
        // Calculate the range of the current bucket
        let bucket_start = ((i as f64 * bucket_size) + 1.0).floor() as usize;
        let bucket_end = (((i + 1) as f64 * bucket_size) + 1.0).floor() as usize;
        let bucket_end = bucket_end.min(n - 1);

        // Calculate the range of the next bucket for averaging
        let next_bucket_start = bucket_end;
        let next_bucket_end = (((i + 2) as f64 * bucket_size) + 1.0).floor() as usize;
        let next_bucket_end = next_bucket_end.min(n);

        // Calculate the average point of the next bucket
        let (avg_time, avg_value) = if next_bucket_end > next_bucket_start {
            let mut sum_time = 0i64;
            let mut sum_value = 0.0f64;
            let count = (next_bucket_end - next_bucket_start) as f64;

            for point in data.iter().take(next_bucket_end).skip(next_bucket_start) {
                sum_time += point.time;
                sum_value += point.value;
            }

            (sum_time as f64 / count, sum_value / count)
        } else {
            // Edge case: use the last point
            (data[n - 1].time as f64, data[n - 1].value)
        };

        // Find the point in the current bucket that forms the largest triangle
        // with the previously selected point and the average of the next bucket
        let a = &data[a_index];
        let mut max_area = -1.0f64;
        let mut max_area_index = bucket_start;

        for (j, point) in data.iter().enumerate().take(bucket_end).skip(bucket_start) {
            // Calculate triangle area using the cross product formula
            // Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
            let area = ((a.time as f64 - avg_time) * (point.value - a.value)
                - (a.time as f64 - point.time as f64) * (avg_value - a.value))
                .abs();

            if area > max_area {
                max_area = area;
                max_area_index = j;
            }
        }

        result.push(data[max_area_index]);
        a_index = max_area_index;
    }

    // Always include the last point
    result.push(data[n - 1]);

    result
}

/// Compute percentile value from a slice of f64 values.
/// Returns None if the slice is empty or contains only non-finite values.
fn percentile(sorted_values: &[f64], p: f64) -> Option<f64> {
    if sorted_values.is_empty() {
        return None;
    }
    let idx = ((p / 100.0) * (sorted_values.len() - 1) as f64) as usize;
    let idx = idx.min(sorted_values.len() - 1);
    Some(sorted_values[idx])
}

/// Downsample time and value arrays using LTTB.
///
/// This function includes outlier filtering: values outside the 0.1-99.9 percentile
/// range are clamped to the nearest bound. This prevents corrupt sensor data from
/// distorting the visualization while preserving the overall signal shape.
///
/// # Arguments
/// * `times` - Timestamp array (microseconds since epoch)
/// * `values` - Value array (f64)
/// * `target_points` - Desired number of output points
///
/// # Returns
/// Tuple of (downsampled_times, downsampled_values)
pub fn lttb_downsample_arrays(
    times: &[i64],
    values: &[f64],
    target_points: usize,
) -> (Vec<i64>, Vec<f64>) {
    assert_eq!(
        times.len(),
        values.len(),
        "times and values must have same length"
    );

    if times.is_empty() {
        return (vec![], vec![]);
    }

    // First, filter outliers using percentile-based clamping
    // This prevents corrupt data (e.g., sensor errors) from distorting the visualization
    let filtered_values = filter_outliers(values);

    // Convert to DataPoints
    let data: Vec<DataPoint> = times
        .iter()
        .zip(filtered_values.iter())
        .map(|(&time, &value)| DataPoint { time, value })
        .collect();

    // Apply LTTB
    let downsampled = lttb_downsample(&data, target_points);

    // Convert back to separate arrays
    let times: Vec<i64> = downsampled.iter().map(|p| p.time).collect();
    let values: Vec<f64> = downsampled.iter().map(|p| p.value).collect();

    (times, values)
}

/// Filter outliers by clamping values outside the 0.1-99.9 percentile range.
/// Non-finite values (NaN, Inf) are replaced with the median.
fn filter_outliers(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return vec![];
    }

    // Collect and sort finite values for percentile calculation
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();

    if sorted.is_empty() {
        // All values are non-finite, return zeros
        return vec![0.0; values.len()];
    }

    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Compute percentile bounds (0.1 and 99.9 percentiles)
    let p_low = percentile(&sorted, 0.1).unwrap_or(sorted[0]);
    let p_high = percentile(&sorted, 99.9).unwrap_or(sorted[sorted.len() - 1]);
    let median = percentile(&sorted, 50.0).unwrap_or(0.0);

    // Clamp outliers and replace non-finite values
    values
        .iter()
        .map(|&v| {
            if !v.is_finite() {
                median
            } else if v < p_low {
                p_low
            } else if v > p_high {
                p_high
            } else {
                v
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lttb_basic() {
        let data: Vec<DataPoint> = (0..100)
            .map(|i| DataPoint {
                time: i * 1000,
                value: (i as f64 * 0.1).sin(),
            })
            .collect();

        let result = lttb_downsample(&data, 20);

        assert_eq!(result.len(), 20);
        assert_eq!(result[0].time, data[0].time);
        assert_eq!(result[19].time, data[99].time);
    }

    #[test]
    fn test_lttb_small_input() {
        let data = vec![
            DataPoint {
                time: 0,
                value: 1.0,
            },
            DataPoint {
                time: 1,
                value: 2.0,
            },
        ];

        let result = lttb_downsample(&data, 10);
        assert_eq!(result.len(), 2); // Can't upsample
    }

    #[test]
    fn test_lttb_arrays() {
        let times: Vec<i64> = (0..50).map(|i| i * 1000).collect();
        let values: Vec<f64> = (0..50).map(|i| (i as f64).sin()).collect();

        let (out_times, out_values) = lttb_downsample_arrays(&times, &values, 10);

        assert_eq!(out_times.len(), 10);
        assert_eq!(out_values.len(), 10);
    }
}
