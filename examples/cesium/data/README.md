# Mojave OSM demonstration data

© OpenStreetMap contributors, ODbL 1.0: https://www.openstreetmap.org/copyright

`mojave-overpass.json` and `mojave.overpassql` are the original response and
query for the active 24 × 24 km flight area. `mojave-overlay.json` contains
prepared ECEF geometry, source IDs, tags, OSM timestamp and heightmap hashes.

`mojave-context-overpass.json` and `mojave-context.overpassql` retain a wider
approximately 100 × 100 km context, with major roads and named summits/places.
This wider context is exported but not yet terrain-prepared or rendered;
the attempted enlarged Cesium height preparation timed out.

OSM `ele` labels retain their mean-sea-level reference. The active geometry
uses the existing cached WGS84 ellipsoid heightmap, with 3 m clearance for
roads and 15 m for labels. See the parent README for regeneration and limits.

The local export also includes 27 building footprints. Their wire outlines use
OSM heights/levels where supplied, or an explicitly illustrative 4 m height.
They are visualization aids; original OSM tags are preserved unchanged.

The local query includes waterway ways and water/reservoir ways and relations.
Prepared blue outlines retain multipolygon outer/inner members and OSM
intermittent/seasonal tags. Shared member ways are not drawn twice. Water uses
the same sampled ellipsoid height grid and 3 m visual clearance as roads.
