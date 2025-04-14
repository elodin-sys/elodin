pub fn trace_init() {
    use tracing_subscriber::filter::LevelFilter;
    let collector = tracing_subscriber::fmt()
        .with_max_level(LevelFilter::TRACE)
        .with_test_writer()
        .without_time()
        .finish();
    let _ = tracing::subscriber::set_global_default(collector);
}
