{pkgs, ...}:
pkgs.runCommand "elodin-examples" {} ''
  mkdir -p "$out"
  cp -R ${../../examples/ball} "$out/ball"
  cp -R ${../../examples/sensor-camera} "$out/sensor-camera"
  cp -R ${../../examples/video-stream} "$out/video-stream"
  cp -R ${../../examples/drone} "$out/drone"
  cp -R ${../../examples/three-body} "$out/three-body"

  chmod -R u+w "$out"
''
