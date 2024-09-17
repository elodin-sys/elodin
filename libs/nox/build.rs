fn run_python_command(cmd: &str) -> String {
    let output = std::process::Command::new("python3")
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

    // Get the stdlib path from the command output
    let output = String::from_utf8(output.stdout)
        .expect("Invalid UTF-8 output")
        .trim()
        .to_string();
    output
}

#[cfg(not(feature = "jax"))]
fn main() {}

#[cfg(feature = "jax")]
fn main() {
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let python_dir = out_dir.join("python");
    std::fs::create_dir_all(&python_dir).unwrap();
    let python = std::fs::canonicalize(which::which("python3").unwrap()).unwrap();
    let python_home = python.parent().unwrap().parent().unwrap();
    let python_lib = python_home.join("lib");

    let shared_lib_extension = if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };

    let lib_name = pyo3_build_config::get().lib_name.as_ref().unwrap();
    let shared_lib = python_lib.join(format!("lib{}.{}", lib_name, shared_lib_extension));

    // copy the shared library to the output python directory:
    let shared_lib_name = shared_lib.file_name().unwrap();
    let shared_lib_dest = python_dir.join(shared_lib_name);
    std::fs::copy(&shared_lib, &shared_lib_dest).unwrap();

    let stdlib_path = run_python_command("import sysconfig; print(sysconfig.get_path('stdlib'))");
    println!("cargo:rustc-env=PYTHON_STDLIB_PATH={}", stdlib_path);

    let purelib_path = run_python_command("import sysconfig; print(sysconfig.get_path('purelib'))");
    println!("cargo:rustc-env=PYTHON_PURELIB_PATH={}", purelib_path);

    println!("cargo:rerun-if-env-changed=VIRTUAL_ENV");
    println!("cargo:rerun-if-env-changed=PYTHONPATH");

    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", python_lib.display());
    println!("cargo:rustc-link-search=native={}", python_dir.display());
}
