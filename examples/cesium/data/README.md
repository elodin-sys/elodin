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
