# Bevy Lower-Triangular Mesh Transform (GPU / Vertex Shader)

This project shows how to apply a **lower-triangular matrix transform** to a mesh **in the vertex shader** in Bevy,
while still using Bevy's standard PBR lighting via `StandardMaterial` + `ExtendedMaterial`.

It includes:
- a small **library** (`src/lib.rs`) that defines a `MaterialExtension` carrying the matrices and selecting a custom vertex shader
- an **example** (`examples/sphere_lower_tri.rs`) that builds a **unit sphere** using `SphereMeshBuilder` and renders it with the shader

## What this does

A lower-triangular 3×3 matrix (plus optional translation in 4×4 form) can represent scale + shear-like effects that you
cannot represent with Bevy's built-in TRS `Transform` alone. This implementation applies your matrix to:
- **positions**: `p' = M * p`
- **normals** (correct for shear): `n' = (M3^{-T}) * n`, where `M3` is the top-left 3×3

## Requirements

- Rust toolchain (stable)
- A working graphics backend supported by Bevy (Vulkan/Metal/DX12/OpenGL)

This project targets **Bevy 0.14**.

## Run the example

From the project directory:

```bash
cargo run --example sphere_lower_tri
```

You should see a sheared/deformed sphere lit by a directional light.

## Where the logic lives

- Rust side:
  - `LowerTriTransformExt` (the material extension) lives in `src/lib.rs`
  - The example computes a lower-triangular matrix + normal matrix and passes them to the material

- Shader side:
  - `assets/shaders/lower_tri_transform.wgsl` applies the transform in **local space** before Bevy's usual local→world→clip.
  - It uses Bevy PBR helper functions (`mesh_functions`, `forward_io`) to remain compatible with the standard pipeline.

## Notes / next steps

- If you need shadows and deferred rendering compatibility, you may also want to implement the corresponding
  `deferred_vertex_shader()` (and/or prepass variants) in the extension depending on your renderer method.
- If your matrix is time-varying, update the material asset each frame (see Bevy patterns for mutating `Assets<T>`).

## License

MIT OR Apache-2.0 (same as Bevy's common dual-licensing pattern).
