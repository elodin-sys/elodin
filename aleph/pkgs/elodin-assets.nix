{lib, ...}:
lib.cleanSourceWith {
  name = "elodin-assets";
  src = ../../assets;
  filter = path: _type: baseNameOf path != ".DS_Store";
}
