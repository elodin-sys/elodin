use std::{mem::ManuallyDrop, pin::Pin};

use crate::{Error, HloModuleProto, Status, XlaOp, XlaOpRaw};
use cpp::{cpp, cpp_class};
use cxx::{CxxString, UniquePtr};
cpp! {{
    #include "xla/client/xla_builder.h"
    #include "xla/client/lib/constants.h"

    #include "mlir/Dialect/Arith/IR/Arith.h"               // from @llvm-project
    #include "mlir/Dialect/Func/IR/FuncOps.h"              // from @llvm-project
    #include "mlir/Dialect/SparseTensor/IR/SparseTensor.h" // from @llvm-project
    #include "mlir/IR/Attributes.h"                        // from @llvm-project
    #include "mlir/IR/Builders.h"                          // from @llvm-project
    #include "mlir/IR/BuiltinOps.h"                        // from @llvm-project
    #include "mlir/IR/BuiltinTypes.h"                      // from @llvm-project
    #include "mlir/IR/MLIRContext.h"                       // from @llvm-project
    #include "mlir/Parser/Parser.h"                        // from @llvm-project
    #include "mlir/Pass/PassManager.h"                     // from @llvm-project
    #include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
    #include "xla/mlir_hlo/mhlo/transforms/passes.h"
    #include "xla/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"

    using namespace xla;
}}

cpp_class!(pub unsafe struct XlaComputation as "XlaComputation");
impl XlaComputation {
    pub fn stmt_while(&self, body: &XlaComputation, init_value: &XlaOp) -> XlaOp {
        let raw = unsafe {
            cpp!([self as "const XlaComputation*", body as "const XlaComputation*", init_value as "const XlaOp*"] -> XlaOpRaw as "XlaOp" {
                return XlaOp(While(*self, *body, *init_value));
            })
        };
        XlaOp {
            raw,
            builder: init_value.builder.clone(),
        }
    }

    pub fn to_hlo_text(&self) -> Result<String, Error> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let cxx_string = unsafe {
            cpp!([self as "const XlaComputation*", out_status as "Status*"] -> UniquePtr<CxxString> as "std::unique_ptr<std::string>" {
                    CompileOptions options;
                    mlir::MLIRContext context;
                    mlir::OwningOpRef<mlir::ModuleOp> module =
                    mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
                    context.loadDialect<mlir::func::FuncDialect>();
                    context.loadDialect<mlir::mhlo::MhloDialect>();
                    auto status = ConvertHloToMlirHlo(*module, &self->proto(), /*import_all_computations=*/true);
                    if(!status.ok()) {
                        *out_status = status;
                        return std::make_unique<std::string>();
                    }
                    mlir::PassManager pm(&context);
                    //pm.addPass(mlir::mhlo::createCollapseElementwiseMapPass());
                    //pm.addPass(mlir::mhlo::createOptimizeMhloPass());
                    if (pm.run(*module).failed()) {
                        *out_status = Status(InvalidArgument("Failed to convert xla computation to mlir"));
                        return std::make_unique<std::string>();
                    }

                    std::string s;
                    llvm::raw_string_ostream os(s);
                    mlir::OpPrintingFlags flags;
                    flags.enableDebugInfo();
                    module->print(os, flags);
                    return std::make_unique<std::string>(s);
            })
        };
        out_status.to_result()?;
        Ok(cxx_string.to_string_lossy().into_owned())
    }

    pub fn to_hlo_module(&self) -> HloModuleProto {
        unsafe {
            cpp!([self as "const XlaComputation*"] -> HloModuleProto as "HloModuleProto" {
                return self->proto();
            })
        }
    }
}

cpp_class!(pub unsafe struct CompileOptionsRaw as "CompileOptions");
#[derive(Default, Clone)]
pub struct CompileOptions(pub ManuallyDrop<CompileOptionsRaw>);
impl CompileOptions {
    pub fn disable_optimizations(&mut self) {
        let raw = &mut self.0;
        unsafe {
            cpp!([raw as "CompileOptions*"] {
                raw->executable_build_options.mutable_debug_options()->set_xla_llvm_disable_expensive_passes(true);
                raw->executable_build_options.mutable_debug_options()->set_xla_backend_optimization_level(0);
            })
        };
    }
}
