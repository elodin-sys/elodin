{lib}: {
  src = let
    includeSrc = orig_path: type: let
      path = toString orig_path;
      base = baseNameOf path;
      relPath = lib.removePrefix (toString ../../..) orig_path;
      matchesPrefix = lib.any (prefix: lib.hasPrefix prefix relPath) [
        "/apps"
        "/libs"
        "/fsw"
        "/.config"
      ];
      matchesSuffix = lib.any (suffix: lib.hasSuffix suffix base) [
        "Cargo.toml"
        "Cargo.lock"
        "logo.txt"
        "logo.png"
        ".rs"
        ".c"
        ".h"
        ".cpp"
        ".hpp"
        ".jinja"
      ];
      matchesExclude = lib.any (exclude: lib.hasPrefix exclude relPath) [
        "/libs/basilisk-rs/vendor"
      ];
    in
      (type == "directory" && matchesPrefix && !matchesExclude) || matchesSuffix;
  in
    lib.cleanSourceWith {
      src = ../../..;
      filter = path: type: includeSrc path type;
    };
}
