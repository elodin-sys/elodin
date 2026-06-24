{pkgs, ...}:
pkgs.runCommand "elodin-examples" {} ''
  mkdir -p "$out"
  cp -R ${../../examples/ball} "$out/ball"
  cp -R ${../../examples/sensor-camera} "$out/sensor-camera"
  cp -R ${../../examples/video-stream} "$out/video-stream"
  cp -R ${../../examples/drone} "$out/drone"
  cp -R ${../../examples/cube-sat} "$out/cube-sat"
  cp -R ${../../examples/three-body} "$out/three-body"

  chmod -R u+w "$out"

  for script in "$out/video-stream/stream-video.sh" "$out/video-stream/receive-obs-stream.sh"; do
    substituteInPlace "$script" \
      --replace-fail "Builds the elodinsink GStreamer plugin from source" "Uses the system elodinsink GStreamer plugin" \
      --replace-fail "The plugin is built automatically - no manual prerequisite steps required." "The plugin is provided by the Aleph elodinsink module." \
      --replace-fail 'echo "Building elodinsink GStreamer plugin..."' 'echo "Using system elodinsink GStreamer plugin..."' \
      --replace-fail 'cargo build --release --manifest-path="$REPO_ROOT/fsw/gstreamer/Cargo.toml"' ':' \
      --replace-fail 'export GST_PLUGIN_PATH="$REPO_ROOT/target/release:''${GST_PLUGIN_PATH}"' ':'
  done
''
