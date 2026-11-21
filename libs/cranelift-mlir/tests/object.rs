use cranelift_mlir::const_fold::fold_module;
use cranelift_mlir::lower::{compile_module_to_object, runtime_symbol};
use cranelift_mlir::parser::parse_module;
use object::{BinaryFormat, Object, ObjectSymbol};

#[test]
fn ball_object_defines_its_entry_and_binds_imports_through_the_table() {
    let mut module = parse_module(include_str!("../testdata/ball.stablehlo.mlir")).unwrap();
    fold_module(&mut module);
    let bytes = compile_module_to_object(&module, "ball").unwrap();
    let file = object::File::parse(&*bytes).unwrap();

    let macho = file.format() == BinaryFormat::MachO;
    let symbols: Vec<(&str, bool)> = file
        .symbols()
        .filter_map(|s| {
            let name = s.name().ok()?;
            // Mach-O prefixes every C symbol with an underscore.
            let name = if macho {
                name.strip_prefix('_').unwrap_or(name)
            } else {
                name
            };
            Some((name, s.is_undefined()))
        })
        .collect();

    for name in [
        "ball_step",
        "ball_imports",
        "ball_import_names",
        "ball_import_count",
    ] {
        assert!(
            symbols.contains(&(name, false)),
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
