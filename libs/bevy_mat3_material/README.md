# Bevy Mat3 Mesh Transform (GPU / Vertex Shader)

This crate applies a 3×3 linear transform to a mesh in the vertex shader in
Bevy, while still using Bevy's standard PBR lighting via `StandardMaterial` as
an `ExtendedMaterial`.

It includes:
- a small **library** (`src/lib.rs`) that defines a `MaterialExtension` carrying
  the matrices and selecting a custom vertex shader, and
- an **example** (`examples/sphere_mat3.rs`) that builds a **unit sphere** using
  `SphereMeshBuilder` and renders it with the shader.

## What this does

A 3×3 matrix can represent scale and shear-like effects that you cannot
represent with Bevy's built-in TRS `Transform` alone. This implementation
applies your matrix to:

- positions: `p' = M * p`
- normals (correct for shear): `n' = (M^{-T}) * n`

## Requirements

- Bevy 0.17

## Run the example

From the project directory:

```bash
cargo run --example sphere-mat3
```

You should see a sheared/deformed sphere lit by a directional light.

## Where the logic lives

- Rust side:
  - The material extension `Mat3TransformExt` lives in `src/lib.rs`.
  - The example computes a 3×3 matrix and inverse matrix and passes them to the material.

- Shader side:
  - `assets/shaders/mat3_transform.wgsl` applies the transform in local space
    before Bevy's usual local→world→clip.
  - It uses Bevy PBR helper functions (`mesh_functions`, `forward_io`) to remain
    compatible with the standard pipeline.

## Notes / next steps

- If you need shadows and deferred rendering compatibility, you may also want to implement the corresponding
  `deferred_vertex_shader()` (and/or prepass variants) in the extension depending on your renderer method.

## License

MIT OR Apache-2.0
