//! Shared EQL→SQL helpers for query plots and SQL-backed graphs.

use arrow::{
    array::{
        Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, Int32Array, Int64Array,
        TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
        TimestampSecondArray, UInt32Array, UInt64Array,
    },
    datatypes::{DataType, TimeUnit},
    record_batch::RecordBatch,
};
use bevy::asset::{Assets, Handle};
use bevy_egui::egui::Color32;
use impeller2_wkt::PlotMode;

use super::plot::XYLine;

/// Compile an EQL expression to SQL with a leading time column for plotting.
pub fn eql_to_sql_with_time(ctx: &eql::Context, query: &str) -> Result<String, String> {
    let expr = ctx.parse_str(query).map_err(|err| err.to_string())?;
    let time_field = expr.to_sql_time_field().map_err(|err| err.to_string())?;
    let mut sql = expr.to_sql(ctx).map_err(|err| err.to_string())?;
    if let Some(pos) = sql.to_lowercase().find("select ") {
        let after_select = pos + 7;
        sql.insert_str(after_select, &format!("{}, ", time_field));
    } else {
        sql = format!(
            "select {}, {}",
            time_field,
            sql.trim_start_matches("select ")
                .trim_start_matches("SELECT ")
        );
    }
    Ok(sql)
}

/// One plotted Y series produced from a SQL record batch.
#[derive(Clone)]
pub struct SqlPlotSeries {
    pub label: String,
    pub handle: Handle<XYLine>,
    pub color: Color32,
}

/// Result of mapping a SQL batch (col0 = X/time, col1..N = Y) into XY lines.
pub struct SqlBatchPlot {
    pub x_offset: f64,
    pub y_offset: f64,
    pub earliest_timestamp: Option<impeller2::types::Timestamp>,
    pub series: Vec<SqlPlotSeries>,
}

fn extract_x_column(x_col: &ArrayRef) -> (Vec<f64>, Option<i64>) {
    match x_col.data_type() {
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            let values: Vec<i64> = x_col
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap()
                .iter()
                .map(|x| x.unwrap_or_default())
                .collect();
            let earliest = values.iter().min().copied();
            let relative: Vec<f64> = values.iter().map(|&x| x as f64 / 1_000_000.0).collect();
            (relative, earliest)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            let values: Vec<i64> = x_col
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()
                .unwrap()
                .iter()
                .map(|x| x.unwrap_or_default())
                .collect();
            let earliest = values.iter().min().copied();
            let relative: Vec<f64> = values.iter().map(|&x| x as f64 / 1_000_000_000.0).collect();
            (relative, earliest.map(|ns| ns / 1_000))
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            let values: Vec<i64> = x_col
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()
                .unwrap()
                .iter()
                .map(|x| x.unwrap_or_default())
                .collect();
            let earliest = values.iter().min().copied();
            let relative: Vec<f64> = values.iter().map(|&x| x as f64 / 1_000.0).collect();
            (relative, earliest.map(|ms| ms * 1_000))
        }
        DataType::Timestamp(TimeUnit::Second, _) => {
            let values: Vec<i64> = x_col
                .as_any()
                .downcast_ref::<TimestampSecondArray>()
                .unwrap()
                .iter()
                .map(|x| x.unwrap_or_default())
                .collect();
            let earliest = values.iter().min().copied();
            let relative: Vec<f64> = values.iter().map(|&x| x as f64).collect();
            (relative, earliest.map(|s| s * 1_000_000))
        }
        _ => (array_iter(x_col).collect(), None),
    }
}

fn skip_initial_points(points: &[(f64, f64)], is_xy_mode: bool) -> usize {
    if is_xy_mode || points.len() <= 2 {
        return 0;
    }
    let first_time = points[0].0;
    let mut last_same = 0usize;
    for (i, (time, _)) in points.iter().enumerate().skip(1) {
        if (*time - first_time).abs() < 0.001 {
            last_same = i;
        } else {
            break;
        }
    }
    if last_same > 0 && last_same + 1 < points.len() {
        last_same + 1
    } else if points.len() >= 3 {
        let first_y = points[0].1;
        let second_y = points[1].1;
        if (second_y - first_y).abs() > 50.0 {
            1
        } else {
            0
        }
    } else {
        0
    }
}

fn column_label(batch: &RecordBatch, col: usize) -> String {
    batch.schema().field(col).name().to_string()
}

/// Build one XY line per Y column. Column 0 is X (time); columns 1..N are series.
pub fn process_sql_record_batch(
    batch: &RecordBatch,
    plot_mode: PlotMode,
    xy_lines: &mut Assets<XYLine>,
    series_colors: &[Color32],
    default_color: Color32,
) -> Option<SqlBatchPlot> {
    if batch.num_columns() < 2 || batch.num_rows() == 0 {
        return None;
    }

    let x_col = batch.column(0);
    let (x_values, earliest_abs_timestamp_micros) = extract_x_column(x_col);

    let earliest_timestamp = earliest_abs_timestamp_micros.map(impeller2::types::Timestamp);

    let finite_x_values: Vec<f64> = x_values
        .iter()
        .copied()
        .filter(|&x| x.is_finite())
        .collect();

    let is_xy_mode = plot_mode == PlotMode::XY;
    let x_offset = if is_xy_mode {
        0.0
    } else {
        let min = finite_x_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        if min.is_finite() { min } else { 0.0 }
    };
    let y_offset = 0.0;

    let mut series = Vec::with_capacity(batch.num_columns().saturating_sub(1));
    for col_idx in 1..batch.num_columns() {
        let y_col = batch.column(col_idx);
        let mut y_iter = array_iter(y_col);
        let mut points: Vec<(f64, f64)> = Vec::new();
        for &x_value in &x_values {
            if let Some(y_value) = y_iter.next()
                && x_value.is_finite()
                && y_value.is_finite()
            {
                points.push((x_value, y_value));
            }
        }

        let skip = skip_initial_points(&points, is_xy_mode);
        let label = column_label(batch, col_idx);
        let mut xy_line = XYLine {
            label: label.clone(),
            x_shard_alloc: None,
            y_shard_alloc: None,
            x_values: vec![],
            y_values: vec![],
        };
        for (x_value, y_value) in points.into_iter().skip(skip) {
            xy_line.push_x_value((x_value - x_offset) as f32);
            xy_line.push_y_value((y_value - y_offset) as f32);
        }

        let color = series_colors.get(col_idx - 1).copied().unwrap_or_else(|| {
            if col_idx == 1 {
                default_color
            } else {
                crate::ui::colors::get_color_by_index_all(col_idx - 1)
            }
        });

        series.push(SqlPlotSeries {
            label,
            handle: xy_lines.add(xy_line),
            color,
        });
    }

    Some(SqlBatchPlot {
        x_offset,
        y_offset,
        earliest_timestamp,
        series,
    })
}

pub fn array_iter(array_ref: &ArrayRef) -> Box<dyn Iterator<Item = f64> + '_> {
    match array_ref.data_type() {
        DataType::Float32 => Box::new(
            array_ref
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .iter()
                .map(|x| x.unwrap_or_default() as f64),
        ),
        DataType::Float64 => Box::new(
            array_ref
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .iter()
                .map(|x| x.unwrap_or_default()),
        ),
        DataType::Int32 => Box::new(
            array_ref
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .iter()
                .map(|x| x.unwrap_or_default() as f64),
        ),
        DataType::Int64 => Box::new(
            array_ref
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .iter()
                .map(|x| x.unwrap_or_default() as f64),
        ),
        DataType::UInt32 => Box::new(
            array_ref
                .as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap()
                .iter()
                .map(|x| x.unwrap_or_default() as f64),
        ),
        DataType::UInt64 => Box::new(
            array_ref
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .iter()
                .map(|x| x.unwrap_or_default() as f64),
        ),
        DataType::Timestamp(TimeUnit::Second, _) => Box::new(
            array_ref
                .as_any()
                .downcast_ref::<TimestampSecondArray>()
                .unwrap()
                .iter()
                .map(|x| x.unwrap_or_default() as f64),
        ),
        DataType::Timestamp(TimeUnit::Millisecond, _) => Box::new(
            array_ref
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()
                .unwrap()
                .iter()
                .map(|x| x.unwrap_or_default() as f64),
        ),
        DataType::Timestamp(TimeUnit::Microsecond, _) => Box::new(
            array_ref
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap()
                .iter()
                .map(|x| x.unwrap_or_default() as f64),
        ),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => Box::new(
            array_ref
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()
                .unwrap()
                .iter()
                .map(|x| x.unwrap_or_default() as f64),
        ),
        DataType::FixedSizeList(_, list_size) => {
            let list_array = array_ref
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .unwrap();
            let list_size = *list_size as usize;
            if list_size == 0 {
                Box::new(std::iter::empty())
            } else {
                let values = list_array.values();
                let inner_values: Vec<f64> = array_iter(values).collect();
                if inner_values.is_empty() {
                    println!("Unsupported list data type: {:?}", values.data_type());
                    Box::new(std::iter::empty())
                } else {
                    let len = list_array.len();
                    let mut min_vals = vec![f64::INFINITY; list_size];
                    let mut max_vals = vec![f64::NEG_INFINITY; list_size];
                    for row in 0..len {
                        if list_array.is_null(row) {
                            continue;
                        }
                        let base = row * list_size;
                        for i in 0..list_size {
                            if let Some(value) = inner_values.get(base + i)
                                && value.is_finite()
                            {
                                if *value < min_vals[i] {
                                    min_vals[i] = *value;
                                }
                                if *value > max_vals[i] {
                                    max_vals[i] = *value;
                                }
                            }
                        }
                    }
                    let mut selected_index = 0usize;
                    let mut best_range = f64::NEG_INFINITY;
                    for i in 0..list_size {
                        let min = min_vals[i];
                        let max = max_vals[i];
                        if min.is_finite() && max.is_finite() {
                            let range = max - min;
                            if range > best_range {
                                best_range = range;
                                selected_index = i;
                            }
                        }
                    }
                    Box::new((0..len).map(move |row| {
                        if list_array.is_null(row) {
                            0.0
                        } else {
                            inner_values
                                .get(row * list_size + selected_index)
                                .copied()
                                .unwrap_or_default()
                        }
                    }))
                }
            }
        }
        ty => {
            println!("Unsupported data type: {:?}", ty);
            Box::new(std::iter::empty())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, TimestampMicrosecondArray};
    use arrow::datatypes::{Field, Schema, TimeUnit as ArrowTimeUnit};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    fn batch_time_xyz() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "time",
                DataType::Timestamp(ArrowTimeUnit::Microsecond, None),
                false,
            ),
            Field::new("ecef_to_ned.x", DataType::Float64, false),
            Field::new("ecef_to_ned.y", DataType::Float64, false),
            Field::new("ecef_to_ned.z", DataType::Float64, false),
        ]));
        let time = TimestampMicrosecondArray::from(vec![1_000_000i64, 2_000_000, 3_000_000]);
        let x = Float64Array::from(vec![1.0, 2.0, 3.0]);
        let y = Float64Array::from(vec![10.0, 20.0, 30.0]);
        let z = Float64Array::from(vec![100.0, 200.0, 300.0]);
        RecordBatch::try_new(
            schema,
            vec![Arc::new(time), Arc::new(x), Arc::new(y), Arc::new(z)],
        )
        .unwrap()
    }

    #[test]
    fn multi_column_batch_yields_three_series() {
        let mut xy_lines = Assets::<XYLine>::default();
        let batch = batch_time_xyz();
        let colors = [
            Color32::from_rgb(1, 2, 3),
            Color32::from_rgb(4, 5, 6),
            Color32::from_rgb(7, 8, 9),
        ];
        let plot = process_sql_record_batch(
            &batch,
            PlotMode::TimeSeries,
            &mut xy_lines,
            &colors,
            Color32::WHITE,
        )
        .expect("plot");
        assert_eq!(plot.series.len(), 3);
        assert_eq!(plot.series[0].label, "ecef_to_ned.x");
        assert_eq!(plot.series[1].label, "ecef_to_ned.y");
        assert_eq!(plot.series[2].label, "ecef_to_ned.z");
        assert_eq!(plot.series[0].color, colors[0]);
        assert_eq!(plot.series[1].color, colors[1]);
        assert_eq!(plot.series[2].color, colors[2]);
        assert!(plot.x_offset.is_finite());
    }
}
