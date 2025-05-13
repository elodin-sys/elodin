fn main() {
    // Generate version information for the plugin
    gst_plugin_version_helper::info();

    #[cfg(target_os = "macos")]
    {
        // For macOS, we need to explicitly export the plugin descriptor symbol
        println!("cargo:rustc-cdylib-link-arg=-Wl,-exported_symbol,_gst_plugin_elodin_get_desc");
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        // For Linux/Unix, ensure all symbols are exported
        println!("cargo:rustc-cdylib-link-arg=-Wl,--version-script=gstreamer.map");
    }
}
