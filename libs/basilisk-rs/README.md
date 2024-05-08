# Basilisk Rust

basilisk-rs is a series of Rust bindings around [Basilisk](https://github.com/AVSLab/basilisk), a astrodynamics and flight software framework. 

Right we only compile and wrap a small number of flight software modules, at some point in the future we may expand this to other simulations

## Updating Basilisk
`basilisk-rs` for the most part builds Basilisk itself, but there are some files it can not build on its own. In particular, Basilisk makes heavy use of SWIG. Instead of calling SWIGF from the `build.rs` we vendor those files into the `vendor` directory. When you want to update Basilisk, first update the submodule reference. Then run `just vendor` which will kick-off building Basilisk. You will need `swig4` installed, ideally from nixpkgs as homebrew's version can cause issues.
