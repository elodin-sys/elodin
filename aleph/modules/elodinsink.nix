{
  config,
  lib,
  pkgs,
  ...
}:
with lib; let
  cfg = config.services.elodinsink;
in {
  options.services.elodinsink = {
    enable = mkEnableOption "elodinsink GStreamer plugin for streaming H.264 video to Elodin-DB";

    package = mkOption {
      type = types.package;
      default = pkgs.elodinsink;
      description = "The elodinsink GStreamer plugin package to use";
    };

    includeNvidiaPlugins = mkOption {
      type = types.bool;
      default = true;
      description = ''
        Whether to include NVIDIA GStreamer plugins (l4t-gstreamer) for hardware
        encoding support. Recommended on Aleph hardware for better performance.
      '';
    };

    includeBasePlugins = mkOption {
      type = types.bool;
      default = true;
      description = ''
        Whether to include standard GStreamer plugins (base, good, bad, ugly).
        These are required for most video pipelines.
      '';
    };
  };

  config = mkIf cfg.enable {
    environment.systemPackages = with pkgs;
      [
        cfg.package
        gst_all_1.gstreamer
        gst_all_1.gst-plugins-base
      ]
      ++ optionals cfg.includeBasePlugins [
        gst_all_1.gst-plugins-good
        gst_all_1.gst-plugins-bad
        gst_all_1.gst-plugins-ugly
      ]
      ++ optionals cfg.includeNvidiaPlugins [
        nvidia-jetpack.l4t-gstreamer
      ];

    environment.variables.GST_PLUGIN_PATH = with pkgs;
      lib.makeSearchPathOutput "lib" "lib/gstreamer-1.0" (
        [
          cfg.package
          gst_all_1.gstreamer
          gst_all_1.gst-plugins-base
        ]
        ++ optionals cfg.includeBasePlugins [
          gst_all_1.gst-plugins-good
          gst_all_1.gst-plugins-bad
          gst_all_1.gst-plugins-ugly
        ]
        ++ optionals cfg.includeNvidiaPlugins [
          nvidia-jetpack.l4t-gstreamer
        ]
      );

    # Add GStreamer libraries to LD_LIBRARY_PATH for runtime linking
    environment.variables.LD_LIBRARY_PATH = with pkgs;
      lib.makeLibraryPath (
        [
          gst_all_1.gstreamer
          gst_all_1.gst-plugins-base
        ]
        ++ optionals cfg.includeNvidiaPlugins [
          nvidia-jetpack.l4t-gstreamer
          nvidia-jetpack.l4t-multimedia
        ]
      );
  };
}
