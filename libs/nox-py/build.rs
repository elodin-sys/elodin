fn main() {
    // The CPU kernels are now compiled directly into noxla
    // We just need to ensure LAPACK/BLAS libraries are available

    #[cfg(target_os = "linux")]
    {
        // Link LAPACK/BLAS libraries - these are provided by the dependencies
        println!("cargo:rustc-link-lib=dylib=lapack");
        println!("cargo:rustc-link-lib=dylib=blas");
        println!("cargo:rustc-link-lib=dylib=gfortran");
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=c++");
        // Link against Accelerate framework for BLAS/LAPACK on macOS
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
