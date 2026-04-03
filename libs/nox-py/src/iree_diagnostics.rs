use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FailureStage {
    JaxLower,
    StablehloEmit,
    IreeCompile,
    VmfbLoad,
    RuntimeInvoke,
    Unknown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FailureClass {
    UnsupportedIreeFeature,
    JaxLoweringError,
    ToolchainMisconfigured,
    VersionMismatch,
    PlatformIssue,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticReport {
    pub stage: FailureStage,
    pub classification: FailureClass,
    pub raw_error: String,
    pub matched_patterns: Vec<String>,
    pub suggestion: String,
    pub report_dir: Option<String>,
}

impl DiagnosticReport {
    pub fn should_suggest_jax_fallback(&self) -> bool {
        matches!(self.classification, FailureClass::UnsupportedIreeFeature)
    }
}

pub fn parse_stage_from_error(error_text: &str) -> FailureStage {
    if error_text.contains("stage=jax_lower") {
        FailureStage::JaxLower
    } else if error_text.contains("stage=stablehlo_emit") {
        FailureStage::StablehloEmit
    } else if error_text.contains("stage=iree_compile") {
        FailureStage::IreeCompile
    } else if error_text.contains("stage=vmfb_load") {
        FailureStage::VmfbLoad
    } else if error_text.contains("stage=runtime_invoke") {
        FailureStage::RuntimeInvoke
    } else {
        FailureStage::Unknown
    }
}

fn maybe_report_dir(error_text: &str) -> Option<String> {
    let marker = "Debug artifacts saved to: ";
    let (_, tail) = error_text.split_once(marker)?;
    let first_line = tail.lines().next()?.trim();
    if first_line.is_empty() {
        None
    } else {
        Some(first_line.to_string())
    }
}

pub fn classify_failure(error_text: &str) -> DiagnosticReport {
    let stage = parse_stage_from_error(error_text);
    let mut matched_patterns = Vec::new();
    let lower = error_text.to_ascii_lowercase();

    let (classification, suggestion) = if lower.contains("arith.bitcast")
        && lower.contains("cast incompatible")
    {
        matched_patterns.push("arith.bitcast cast incompatible".to_string());
        (
            FailureClass::UnsupportedIreeFeature,
            "Likely TOTALORDER/f64 demotion incompatibility. Try a newer IREE/JAX pairing; if blocked, use backend=\"jax\" temporarily.".to_string(),
        )
    } else if lower.contains("tensor.expand_shape") && lower.contains("static value of 2") {
        matched_patterns.push("tensor.expand_shape static mismatch".to_string());
        (
            FailureClass::UnsupportedIreeFeature,
            "Likely PRNG/dynamic shape limitation. Try JAX PRNG config changes and consider backend=\"jax\" as a temporary unblocker.".to_string(),
        )
    } else if lower.contains("iree_linalg_ext.scatter") {
        matched_patterns.push("iree_linalg_ext.scatter limitation".to_string());
        (
            FailureClass::UnsupportedIreeFeature,
            "Detected scatter lowering limitation (often from .at[].set()). Rewrite that pattern or use backend=\"jax\" for now.".to_string(),
        )
    } else if lower.contains("failed to legalize operation 'stablehlo.compare'")
        && lower.contains("explicitly marked illegal")
    {
        matched_patterns.push("stablehlo.compare legalization failure".to_string());
        (
            FailureClass::UnsupportedIreeFeature,
            "Detected a stablehlo.compare legalization failure. This is often caused by scalar-vs-singleton broadcast mismatches in singleton-lowered systems; inspect `stablehlo.mlir`, `iree_compile_stderr.txt`, and `compile_context.json`.".to_string(),
        )
    } else if lower.contains("does not dominate this use") {
        matched_patterns.push("control-flow dominance failure".to_string());
        (
            FailureClass::UnsupportedIreeFeature,
            "Detected an MLIR dominance failure during IREE compilation (commonly from large control-flow regions such as `jax.lax.cond` lowered to `stablehlo.case`). Inspect `stablehlo.mlir`, `iree_compile_stderr.txt`, and `compile_context.json`.".to_string(),
        )
    } else if lower.contains("elements hex data size is invalid for provided type")
        && lower.contains("xi1")
    {
        matched_patterns.push("packed i1 dense constant parse failure".to_string());
        (
            FailureClass::VersionMismatch,
            "Detected a packed-boolean StableHLO constant parse failure. This looks like a JAX/StableHLO/IREE compatibility issue; inspect `stablehlo.mlir`, `iree_compile_stderr.txt`, and `compile_context.json` before changing model code.".to_string(),
        )
    } else if lower.contains("no function")
        && lower.contains("exported by module")
        && lower.contains("elodin_lapack")
    {
        matched_patterns.push("elodin_lapack function not found".to_string());
        (
            FailureClass::ToolchainMisconfigured,
            "The IREE runtime LAPACK module does not export a function the compiled module requires. This usually means the elodin package is outdated; rebuild and reinstall.".to_string(),
        )
    } else if lower.contains("custom_call") {
        matched_patterns.push("custom_call unsupported".to_string());
        (
            FailureClass::UnsupportedIreeFeature,
            "Detected custom_call/LAPACK lowering limitation. Replace unsupported linalg ops or use backend=\"jax\" temporarily.".to_string(),
        )
    } else if lower.contains("no such file or directory") && lower.contains("iree-compile") {
        matched_patterns.push("iree-compile missing".to_string());
        (
            FailureClass::ToolchainMisconfigured,
            "IREE compiler binary was not found. Run in `nix develop` or install matching `iree-base-compiler`.".to_string(),
        )
    } else if lower.contains("undefined symbol")
        || lower.contains("symbol not found")
        || (lower.contains("func") && lower.contains("not found"))
    {
        matched_patterns.push("platform linker/symbol issue".to_string());
        (
            FailureClass::ToolchainMisconfigured,
            "Detected unresolved linker symbols during IREE compile (for example cos/sin). Run from `nix develop`, verify your IREE toolchain version, and inspect `iree_compile_cmd.sh` + `iree_compile_stderr.txt` in the dump directory.".to_string(),
        )
    } else if matches!(stage, FailureStage::JaxLower | FailureStage::StablehloEmit) {
        (
            FailureClass::JaxLoweringError,
            "JAX lowering failed before IREE compilation. Inspect the stage-specific traceback file in the dump directory; `stablehlo.mlir` may be absent if lowering failed before StableHLO emission.".to_string(),
        )
    } else {
        (
            FailureClass::Unknown,
            "Unknown failure. Use the dumped artifacts (`compile_context.json`, traceback/StableHLO, stderr, and command) to reproduce and investigate.".to_string(),
        )
    };

    DiagnosticReport {
        stage,
        classification,
        raw_error: error_text.to_string(),
        matched_patterns,
        suggestion,
        report_dir: maybe_report_dir(error_text),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_bitcast_issue() {
        let r = classify_failure(
            "'arith.bitcast' op operand type 'f32' and result type 'i64' are cast incompatible",
        );
        assert_eq!(r.classification, FailureClass::UnsupportedIreeFeature);
    }

    #[test]
    fn classifies_expand_shape_issue() {
        let r = classify_failure(
            "'tensor.expand_shape' op expected dimension 1 of collapsed type to be static value of 2",
        );
        assert_eq!(r.classification, FailureClass::UnsupportedIreeFeature);
    }

    #[test]
    fn classifies_scatter_issue() {
        let r = classify_failure("'iree_linalg_ext.scatter' op dimension map is invalid");
        assert_eq!(r.classification, FailureClass::UnsupportedIreeFeature);
    }

    #[test]
    fn classifies_compare_legalization_issue() {
        let r = classify_failure(
            "stage=iree_compile\niree-compile failed (code 1):\n\
             <stdin>:1057:12: error: failed to legalize operation 'stablehlo.compare' \
             that was explicitly marked illegal",
        );
        assert_eq!(r.classification, FailureClass::UnsupportedIreeFeature);
        assert!(
            r.matched_patterns
                .contains(&"stablehlo.compare legalization failure".to_string())
        );
    }

    #[test]
    fn classifies_dominance_failure() {
        let r = classify_failure(
            "stage=iree_compile\niree-compile failed (code 1):\n\
             error: operand #0 does not dominate this use",
        );
        assert_eq!(r.classification, FailureClass::UnsupportedIreeFeature);
        assert!(
            r.matched_patterns
                .contains(&"control-flow dominance failure".to_string())
        );
    }

    #[test]
    fn classifies_packed_i1_parse_issue() {
        let r = classify_failure(
            "error: elements hex data size is invalid for provided type: 'tensor<81x4xi1>'",
        );
        assert_eq!(r.classification, FailureClass::VersionMismatch);
    }

    #[test]
    fn classifies_lapack_module_missing_function() {
        let r = classify_failure(
            "stage=vmfb_load\nIREE runtime error: NOT_FOUND; \
             no function `dgetrs_2_2_2` exported by module `elodin_lapack`",
        );
        assert_eq!(r.classification, FailureClass::ToolchainMisconfigured);
        assert!(
            r.matched_patterns
                .contains(&"elodin_lapack function not found".to_string())
        );
    }

    #[test]
    fn classifies_missing_toolchain() {
        let r = classify_failure("No such file or directory: iree-compile");
        assert_eq!(r.classification, FailureClass::ToolchainMisconfigured);
    }

    #[test]
    fn classifies_undefined_symbol_linker_error() {
        let r = classify_failure("iree-lld: error: undefined symbol: cos");
        assert_eq!(r.classification, FailureClass::ToolchainMisconfigured);
    }

    #[test]
    fn classifies_unknown() {
        let r = classify_failure("unrecognized garble text");
        assert_eq!(r.classification, FailureClass::Unknown);
    }
}
