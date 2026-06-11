use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, types};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{ArenaMemoryProvider, JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

#[test]
#[ignore = "validation gate for imported large constants under the production JIT arena"]
fn imported_data_symbol_finalizes_with_production_arena() {
    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "speed").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    let isa_builder = cranelift_native::builder().expect("native ISA");
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .expect("ISA finish");
    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

    const JIT_ARENA_SIZE: usize = 2 * 1024 * 1024 * 1024;
    let arena = ArenaMemoryProvider::new_with_size(JIT_ARENA_SIZE).expect("arena");
    jit_builder.memory_provider(Box::new(arena));

    let table = vec![0u64; 256 * 1024];
    jit_builder.symbol(
        "__elodin_const_validation_blob",
        table.as_ptr().cast::<u8>(),
    );

    let mut module = JITModule::new(jit_builder);
    let data_id = module
        .declare_data(
            "__elodin_const_validation_blob",
            Linkage::Import,
            false,
            false,
        )
        .expect("declare imported data");

    let mut sig = module.make_signature();
    sig.returns.push(AbiParam::new(types::I64));
    let fn_id = module
        .declare_function("read_imported_data", Linkage::Export, &sig)
        .expect("declare function");

    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    let mut fb_ctx = FunctionBuilderContext::new();
    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
        let block = builder.create_block();
        builder.switch_to_block(block);
        builder.seal_block(block);

        let ptr_ty = module.target_config().pointer_type();
        let gv = module.declare_data_in_func(data_id, builder.func);
        let ptr = builder.ins().global_value(ptr_ty, gv);
        let value = builder.ins().load(types::I64, MemFlags::trusted(), ptr, 0);
        builder.ins().return_(&[value]);
        builder.finalize();
    }

    module
        .define_function(fn_id, &mut ctx)
        .expect("define function");
    module.clear_context(&mut ctx);
    module.finalize_definitions().expect("finalize definitions");
}
