fn main() {
    gst_plugin_version_helper::info();

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-exported_symbol,_gst_plugin_elodin_get_desc");
    }
}
