# Optimize a glTF/GLB asset for Bevy + Git LFS

Shrink a `.glb`/`.gltf` asset to reduce its Git LFS footprint while keeping it
loadable by the editor's Bevy 0.18 loader. Follow the `gltf-asset-optimization`
skill for the rationale and constraints.

Target file and keep-ratio (fraction of triangles to keep) follow this command;
if omitted, ask which asset to optimize and default the ratio to `0.25`.

Steps:

1. Diagnose first: `npx @gltf-transform/cli inspect <file>` to see whether it's
   geometry- or texture-dominated, and report the current size.
2. Run the bundled script from the repo root (default overwrites the input;
   original is recoverable via git/LFS):

   ```bash
   scripts/optimize-glb.sh <file> <keep-ratio>
   ```

   Ratios: `0.5` conservative, `0.25` default, `0.1` aggressive. The script keeps
   output as plain glTF 2.0 (no Draco/meshopt/quantization/instancing) and aborts
   if the result requires extensions Bevy 0.18 cannot load.
3. Report the before/after sizes and reduction.
4. Remind me to verify visual quality (open in the editor or https://gltf.report)
   and that the asset is LFS-tracked. Do not commit — I will.
