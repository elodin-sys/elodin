[out:json][timeout:60][bbox:35.2425042578,-117.9410445438,35.4588237422,-117.6770094562];
(
  node["natural"="peak"]["name"];
  node["place"]["name"];
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|track)(_link)?$"];
  way["building"]["building"!="no"];
  way["waterway"~"^(river|stream|canal|drain|ditch|tidal_channel|riverbank)$"];
  way["natural"="water"];
  relation["natural"="water"];
  way["landuse"="reservoir"];
  relation["landuse"="reservoir"];
);
out body geom;
