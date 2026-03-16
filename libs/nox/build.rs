#[cfg(feature = "jax")]
fn find_python() -> std::path::PathBuf {
    // Prefer the Python that pyo3/maturin selected — this avoids picking up a
    // stale UV-managed interpreter when `uvx maturin develop` injects its tool
    // environment into PATH.
    for var in ["PYO3_PYTHON", "PYTHON_SYS_EXECUTABLE"] {
        if let Ok(p) = std::env::var(var) {
            let path = std::path::PathBuf::from(p);
            if path.exists() {
                return path;
            }
        }
    }
    which::which("python3").expect("python3 not found on PATH")
}

#[cfg(feature = "jax")]
fn run_python_command(python: &std::path::Path, cmd: &str) -> String {
    let output = std::process::Command::new(python)
        .arg("-c")
        .arg(cmd)
        .output()
        .expect("Failed to execute python3 command");

    if !output.status.success() {
        panic!(
            "Python command failed: {:?}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    String::from_utf8(output.stdout)
        .expect("Invalid UTF-8 output")
        .trim()
        .to_string()
}

#[cfg(not(feature = "jax"))]
fn main() {}

#[cfg(feature = "jax")]
fn main() {
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let python_dir = out_dir.join("python");
    std::fs::create_dir_all(&python_dir).unwrap();
    let python = find_python();
    println!("cargo:rerun-if-changed={}", python.display());

    let python_lib = std::path::PathBuf::from(run_python_command(
        &python,
        "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))",
    ));

    let shared_lib_extension = if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };

    let lib_name = run_python_command(
        &python,
        "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')",
    );
    let shared_lib = python_lib.join(format!("lib{}.{}", lib_name, shared_lib_extension));

    // copy the shared library to the output python directory if it exists:
    if shared_lib.exists() {
        let shared_lib_name = shared_lib.file_name().unwrap();
        let shared_lib_dest = python_dir.join(shared_lib_name);
        let _ = std::fs::copy(&shared_lib, &shared_lib_dest);
    } else {
        println!(
            "cargo:warning=Python shared library not found: {}",
            shared_lib.display()
        );
    }

    let stdlib_path = run_python_command(
        &python,
        "import sysconfig; print(sysconfig.get_path('stdlib'))",
    );
    println!("cargo:rustc-env=PYTHON_STDLIB_PATH={}", stdlib_path);

    let purelib_path = run_python_command(
        &python,
        "import sysconfig; print(sysconfig.get_path('purelib'))",
    );
    println!("cargo:rustc-env=PYTHON_PURELIB_PATH={}", purelib_path);

    println!("cargo:rerun-if-env-changed=VIRTUAL_ENV");
    println!("cargo:rerun-if-env-changed=PYTHONPATH");
    println!("cargo:rerun-if-env-changed=PYO3_BUILD_EXTENSION_MODULE");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=PYTHON_SYS_EXECUTABLE");

    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", python_lib.display());
    println!("cargo:rustc-link-search=native={}", python_dir.display());

    // When building a Python extension module (via maturin), do NOT link against
    // libpython — Python provides symbols at runtime. In Nix builds on Darwin,
    // the shared library often isn't available at all.
    // For regular builds (cargo test, cargo build --bin) we DO need the link.
    let is_extension_module = std::env::var("PYO3_BUILD_EXTENSION_MODULE").is_ok()
        || std::env::var("MATURIN_PYTHON").is_ok()
        || std::env::var("MATURIN_TARGET").is_ok()
        || std::env::var("PYO3_CONFIG_FILE")
            .map(|f| f.contains("maturin"))
            .unwrap_or(false);

    if !is_extension_module {
        println!("cargo:rustc-link-lib=dylib={}", lib_name);
    }
}
