use std::sync::Arc;

use eql::{Component, Context};
use impeller2::schema::Schema;
use impeller2::types::{ComponentId, PrimType, Timestamp};

fn build_context() -> Context {
    let component = Arc::new(Component::new(
        "a.world_pos".to_string(),
        ComponentId::new("a.world_pos"),
        Schema::new(PrimType::F64, vec![3u64]).unwrap(),
    ));

    Context::from_leaves(
        [component],
        Timestamp(0),
        Timestamp(1_000_000), // 1 second in microseconds
    )
}

#[test]
fn norm_generates_expected_sql() {
    let context = build_context();
    let sql = context.sql("a.world_pos.norm()").expect("norm() SQL");

    assert!(
        sql.contains("sqrt("),
        "norm() should expand to sqrt of summed components, got {sql}"
    );
}

#[test]
fn fft_operates_on_component_elements() {
    let context = build_context();
    let sql = context
        .sql("a.world_pos[0].fft()")
        .expect("fft() SQL generation");

    assert!(
        sql.contains("fft(a_world_pos.a_world_pos[1])"),
        "fft() should reference the first component element, got {sql}"
    );
}

#[test]
fn fftfreq_runs_on_time_expressions() {
    let context = build_context();
    let sql = context
        .sql("a.world_pos.time.fftfreq()")
        .expect("fftfreq() SQL generation");

    assert!(
        sql.contains("fftfreq(a_world_pos.time)"),
        "fftfreq() should target the component time column, got {sql}"
    );
}

#[test]
fn last_filters_using_trailing_window() {
    let context = build_context();
    let sql = context
        .sql("a.world_pos.last(\"PT0.5S\")")
        .expect("last() SQL generation");

    assert!(
        sql.contains(">= to_timestamp(0.5)"),
        "last() should filter using the trailing 0.5s window, got {sql}"
    );
}

#[test]
fn first_filters_using_leading_window() {
    let context = build_context();
    let sql = context
        .sql("a.world_pos.first(\"PT0.5S\")")
        .expect("first() SQL generation");

    assert!(
        sql.contains("<= to_timestamp(0.5)"),
        "first() should filter using the leading 0.5s window, got {sql}"
    );
}
