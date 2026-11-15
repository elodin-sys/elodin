# Geographic overlays

Geographic data copied from the [Elodin Terrain Demo](https://github.com/elodin-sys/elodin-terrain-demo/tree/main/data).

- `cape-canaveral.json`: local roads, airport surfaces and place labels. Attribution: © OpenStreetMap contributors; see the source URL in the JSON.
- `orbital-context.json`: regional place labels and water names. Source references and coordinate conventions are preserved in the JSON.
- `falcon9-sites.json`: LC-39A and Landing Zone 1, copied from the demo's `overlay/terrain_demo.kdl` with their original coordinates and altitudes.

Original Git blob IDs:

- `cape-canaveral.json`: `cc7e6a6a68f157298b173df94c8f97910a030823`
- `orbital-context.json`: `a99237fd8097f4f57324f1134ff32874e6d1d1b0`

The F9 schematic declares these assets with `ui.map_overlay(...)`. The editor loads
the JSON from the DB, draws geographic labels and geometry, and shows an
**Overlays on/off** button in the 3D viewport. Attribution links are in its tooltip.
Unreferenced, missing, invalid or empty assets do not create a button.

Overlays are independent of the terrain provider and work with prepared terrain
or streamed terrain, in the native editor and the browser. Changing their runtime
visibility does not modify the recording.
