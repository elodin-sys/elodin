mod common;

use common::CHUNKED;
use cranelift_mlir::const_fold::fold_module;
use cranelift_mlir::lower::{compile_module_to_object, runtime_symbol};
use cranelift_mlir::parser::parse_module;
use object::{BinaryFormat, Object, ObjectSymbol};

/// Each symbol of the object `mlir` compiles to, and whether it is undefined.
fn symbols_for_cpu(mlir: &str, prefix: &str, workers: usize, cpu: &str) -> Vec<(String, bool)> {
    let mut module = parse_module(mlir).unwrap();
    fold_module(&mut module);
    let bytes = compile_module_to_object(&module, prefix, workers, cpu).unwrap();
    let file = object::File::parse(&*bytes).unwrap();
    let macho = file.format() == BinaryFormat::MachO;
    file.symbols()
        .filter_map(|s| {
            let name = s.name().ok()?;
            // Mach-O prefixes every C symbol with an underscore.
            let name = if macho {
                name.strip_prefix('_').unwrap_or(name)
            } else {
                name
            };
            Some((name.to_string(), s.is_undefined()))
        })
        .collect()
}

fn symbols(mlir: &str, prefix: &str, workers: usize) -> Vec<(String, bool)> {
    symbols_for_cpu(mlir, prefix, workers, "native")
}

fn assert_bound_through_the_table(symbols: &[(String, bool)], prefix: &str) {
    for suffix in ["step", "imports", "import_names", "import_count"] {
        let name = format!("{prefix}_{suffix}");
        assert!(
            symbols.contains(&(name.clone(), false)),
            "{name} is not defined in the object"
        );
    }
    for (name, undefined) in symbols {
        assert!(
            !undefined || runtime_symbol(name).is_none(),
            "{name} is left for the linker instead of the import table"
        );
    }
}

#[test]
fn ball_object_defines_its_entry_and_binds_imports_through_the_table() {
    let symbols = symbols(include_str!("../testdata/ball.stablehlo.mlir"), "ball", 1);
    assert_bound_through_the_table(&symbols, "ball");
}

#[test]
fn ball_object_compiles_for_the_generic_host_architecture() {
    let symbols = symbols_for_cpu(
        include_str!("../testdata/ball.stablehlo.mlir"),
        "generic_ball",
        1,
        "generic",
    );
    assert_bound_through_the_table(&symbols, "generic_ball");
}

#[test]
fn an_object_for_workers_runs_its_call_groups_through_the_pool() {
    let serial = symbols(CHUNKED, "chunked", 1);
    let parallel = symbols(CHUNKED, "chunked", 4);
    assert_bound_through_the_table(&parallel, "chunked");
    // A call group hands each callee to the pool through a task trampoline.
    let groups = |symbols: &[(String, bool)]| {
        symbols
            .iter()
            .any(|(name, undefined)| name.starts_with("__task_") && !undefined)
    };
    assert!(groups(&parallel), "the object has no call groups");
    assert!(!groups(&serial), "a serial object has call groups");
}
