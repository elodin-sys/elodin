pub mod default;
pub mod placement;
pub mod spawn;

pub use default::{
    base_window, default_composite_alpha_mode, default_present_mode, default_window_theme,
    window_theme_for_mode,
};
pub use placement::{
    apply_physical_screen_rect, collect_sorted_screens, detect_window_screen, handle_window_close,
    handle_window_relayout_events, wait_for_winit_window,
};
pub use spawn::{compute_window_title, sync_windows, window_graph_order_base};
