use bevy::{
    ecs::system::SystemParam,
    prelude::*,
    window::{EnabledButtons, PresentMode, PrimaryWindow, WindowResolution, WindowTheme},
};
use bevy_egui::EguiContexts;
use egui::{Color32, CornerRadius, RichText, Stroke, load::SizedTexture};
use hifitime::Epoch;
use impeller2_bevy::{
    ConnectionAddr, ConnectionStatus, CurrentStreamId, PacketRx, PacketTx, ThreadConnectionStatus,
    spawn_tcp_connect,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::{
    collections::BTreeMap,
    net::{Ipv6Addr, SocketAddr, ToSocketAddrs},
    path::PathBuf,
    sync::Arc,
};

use crate::{VERSION, dirs};

use super::{
    button::EButton,
    colors::{self, ColorExt, get_scheme},
    images,
    theme::{self, corner_radius_sm},
    widgets::{RootWidgetSystem, RootWidgetSystemExt},
};

#[derive(Component)]
pub struct StartupWindow;

fn create_startup_window(
    mut commands: Commands,
    status: Res<ThreadConnectionStatus>,
    mut primary: Query<&mut Window, With<PrimaryWindow>>,
) {
    commands.insert_resource(recent_files());
    if status.status() == ConnectionStatus::NoConnection {
        let composite_alpha_mode = if cfg!(target_os = "macos") {
            bevy::window::CompositeAlphaMode::PostMultiplied
        } else {
            bevy::window::CompositeAlphaMode::Opaque
        };

        commands.spawn((
            Window {
                title: "Elodin".to_owned(),
                resolution: WindowResolution::new(730.0, 470.0),
                resize_constraints: WindowResizeConstraints {
                    min_width: 730.0,
                    min_height: 470.0,
                    max_width: 730.0,
                    max_height: 470.0,
                },
                present_mode: PresentMode::AutoVsync,
                window_theme: Some(WindowTheme::Dark),
                enabled_buttons: EnabledButtons {
                    minimize: false,
                    maximize: false,
                    close: true,
                },
                composite_alpha_mode,
                ..Default::default()
            },
            StartupWindow,
        ));
    } else if let Ok(mut primary) = primary.single_mut() {
        primary.visible = true
    }
}

pub fn add_layouts(world: &mut World) {
    world.add_root_widget_with::<StartupLayout, With<StartupWindow>>("startup_layout", ());
}

pub struct StartupPlugin;

impl Plugin for StartupPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, create_startup_window);
        app.add_systems(Update, add_layouts);
    }
}

#[derive(SystemParam)]
pub struct StartupLayout<'w, 's> {
    contexts: EguiContexts<'w, 's>,
    window: Query<'w, 's, Entity, With<StartupWindow>>,
    main_window: Query<'w, 's, &'static mut Window, (With<PrimaryWindow>, Without<StartupWindow>)>,
    images: Local<'s, images::Images>,
    modal_state: Local<'s, ModalState>,
    packet_tx: ResMut<'w, PacketTx>,
    packet_rx: ResMut<'w, PacketRx>,
    current_stream_id: ResMut<'w, CurrentStreamId>,
    status: ResMut<'w, ThreadConnectionStatus>,
    recent_files: ResMut<'w, RecentItems>,
    commands: Commands<'w, 's>,
}

#[derive(Resource, Serialize, Deserialize, Default)]
struct RecentItems {
    recent_files: BTreeMap<Epoch, RecentItem>,
    index: HashMap<RecentItem, Epoch>,
}

impl RecentItems {
    fn push(&mut self, epoch: Epoch, item: RecentItem) {
        if let Some(existing_epoch) = self.index.insert(item.clone(), epoch) {
            self.recent_files.remove(&existing_epoch);
        }
        self.recent_files.insert(epoch, item);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub enum RecentItem {
    File(PathBuf),
    Addr(String),
}

#[derive(Default, Clone)]
pub enum ModalState {
    #[default]
    None,
    ConnectToIp {
        addr: String,
        error: Option<ConnectError>,
        connecting: Option<ThreadConnectionStatus>,
    },
}

#[derive(Clone)]
pub enum ConnectError {
    SocketParse(Arc<std::io::Error>),
    Connection,
    ResolutionFailed,
}

impl std::fmt::Display for ConnectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConnectError::SocketParse(addr_parse_error) => {
                write!(f, "{}", addr_parse_error)
            }
            ConnectError::Connection => {
                write!(f, "Error connecting")
            }
            ConnectError::ResolutionFailed => {
                write!(f, "Resolution failed")
            }
        }
    }
}

impl StartupLayout<'_, '_> {
    fn connect(&mut self, addr: SocketAddr, reconnect: bool) -> ThreadConnectionStatus {
        let (packet_tx, packet_rx, outgoing_packet_rx, incoming_packet_tx) =
            impeller2_bevy::channels();
        let stream_id = fastrand::u64(..);
        let status = spawn_tcp_connect(
            addr,
            outgoing_packet_rx,
            incoming_packet_tx,
            stream_id,
            reconnect,
        );
        self.commands.insert_resource(ConnectionAddr(addr));
        *self.current_stream_id = CurrentStreamId(stream_id);
        *self.packet_tx = packet_tx;
        *self.packet_rx = packet_rx;
        *self.status = status.clone();
        status
    }

    fn switch_to_main(&mut self) {
        let e = self.window.single_mut();
        self.commands
            .entity(e.expect("Window entity should exist"))
            .despawn();
        if let Ok(mut window) = self.main_window.single_mut() {
            window.visible = true;
        }
    }

    fn open_file(&mut self, file: PathBuf) {
        let dirs = dirs();

        let cache_dir = dirs.cache_dir().to_owned();
        self.recent_files.push(
            hifitime::Epoch::now().unwrap(),
            RecentItem::File(file.clone()),
        );
        save_recent_files(&self.recent_files);

        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            #[cfg(not(target_os = "windows"))]
            rt.block_on(crate::run::run_recipe(
                cache_dir,
                file,
                stellarator::util::CancelToken::default(),
            ))
            .unwrap();
        });
        self.connect(SocketAddr::new(Ipv6Addr::UNSPECIFIED.into(), 2240), true);
        self.switch_to_main();
    }

    fn start_connect(&mut self, addr: String) {
        let socket_addr = match addr.to_socket_addrs().map(|mut x| x.next()) {
            Ok(Some(addr)) => addr,
            Ok(None) => {
                *self.modal_state = ModalState::ConnectToIp {
                    addr,
                    error: Some(ConnectError::ResolutionFailed),
                    connecting: None,
                };
                return;
            }

            Err(err) => {
                *self.modal_state = ModalState::ConnectToIp {
                    addr,
                    error: Some(ConnectError::SocketParse(Arc::new(err))),
                    connecting: None,
                };
                return;
            }
        };
        self.recent_files.push(
            hifitime::Epoch::now().unwrap(),
            RecentItem::Addr(addr.clone()),
        );
        save_recent_files(&self.recent_files);

        let status = self.connect(socket_addr, false);
        *self.modal_state = ModalState::ConnectToIp {
            addr,
            error: None,
            connecting: Some(status),
        };
    }
}

impl RootWidgetSystem for StartupLayout<'_, '_> {
    type Args = ();
    type Output = ();

    fn ctx_system(
        world: &mut World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ctx: &mut egui::Context,
        _args: Self::Args,
    ) -> Self::Output {
        let mut state = state.get_mut(world);
        let logo_full = state
            .contexts
            .add_image(state.images.logo_full.clone_weak());
        let folder = state
            .contexts
            .add_image(state.images.icon_folder.clone_weak());

        let arrow = state
            .contexts
            .add_image(state.images.icon_chevron_right.clone_weak());
        let icon_ip_addr = state
            .contexts
            .add_image(state.images.icon_ip_addr.clone_weak());

        theme::set_theme(ctx);
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                ui.allocate_ui_with_layout(
                    egui::vec2(408.0, 470.0),
                    egui::Layout::top_down(egui::Align::Center),
                    |ui| {
                        ui.painter().rect_filled(
                            ui.max_rect(),
                            egui::CornerRadius::ZERO,
                            get_scheme().bg_primary,
                        );
                        ui.add_space(87.);
                        ui.add(
                            egui::Image::new(SizedTexture::new(logo_full, egui::vec2(113., 24.)))
                                .tint(get_scheme().text_primary),
                        );
                        ui.add_space(10.0);
                        ui.label(
                            RichText::new(format!("VERSION {}", VERSION))
                                .color(get_scheme().text_secondary),
                        );
                        ui.add_space(75.0);
                        if !cfg!(target_os = "windows") {
                            if ui
                                .add(startup_button("Open Existing Project", folder))
                                .clicked()
                            {
                                if let Some(file) = rfd::FileDialog::new()
                                    .add_filter("python", &["py"])
                                    .add_filter("s10", &["toml"])
                                    .pick_file()
                                {
                                    state.open_file(file);
                                }
                            }
                            ui.add_space(10.0);
                        }
                        if ui
                            .add(startup_button("Connect to IP Address", icon_ip_addr))
                            .clicked()
                        {
                            *state.modal_state = ModalState::ConnectToIp {
                                addr: if cfg!(target_os = "windows") {
                                    "127.0.0.1:2240"
                                } else {
                                    "[::]:2240"
                                }
                                .to_string(),
                                error: None,
                                connecting: None,
                            }
                        }
                    },
                )
            });

        egui::SidePanel::right("right")
            .exact_width(322.0)
            .frame(egui::Frame::NONE.fill(get_scheme().bg_secondary))
            .resizable(false)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for item in state.recent_files.recent_files.clone().into_values().rev() {
                        if ui.add(recent_item_button(item.clone(), arrow)).clicked() {
                            match item {
                                RecentItem::File(path) => {
                                    state.open_file(path.clone());
                                }
                                RecentItem::Addr(addr) => {
                                    state.start_connect(addr);
                                }
                            }
                        }
                    }
                });
            });
        match state.modal_state.clone() {
            ModalState::None => {}
            ModalState::ConnectToIp {
                mut addr,
                error,
                connecting,
            } => {
                egui::Modal::new(egui::Id::new("connect_to_ip"))
                    .frame(egui::Frame {
                        fill: get_scheme().bg_primary,
                        stroke: egui::Stroke::NONE,
                        inner_margin: egui::Margin::same(24),
                        outer_margin: egui::Margin::symmetric(0, 0),
                        corner_radius: CornerRadius::same(8),
                        shadow: egui::Shadow {
                            color: get_scheme().bg_secondary.opacity(0.2),
                            spread: 3,
                            blur: 32,
                            offset: [0, 5],
                        },
                    })
                    .backdrop_color(get_scheme().bg_secondary.opacity(0.85))
                    .show(ctx, |ui| {
                        ui.add(
                            egui::Label::new(RichText::new("Connect to IP Address").size(18.))
                                .selectable(false),
                        );
                        ui.add_space(5.);
                        ui.add(
                            egui::Label::new(
                                RichText::new("Enter the IP address of a DB or Sim")
                                    .color(get_scheme().text_tertiary)
                                    .size(12.),
                            )
                            .selectable(false),
                        );
                        ui.add_space(32.);

                        let style = ui.style_mut();
                        style.visuals.widgets.active.corner_radius = CornerRadius::ZERO;
                        style.visuals.widgets.hovered.corner_radius = CornerRadius::ZERO;
                        style.visuals.widgets.open.corner_radius = CornerRadius::ZERO;

                        style.visuals.widgets.active.fg_stroke =
                            Stroke::new(0.0, Color32::TRANSPARENT);
                        style.visuals.widgets.active.bg_stroke =
                            Stroke::new(0.0, Color32::TRANSPARENT);
                        style.visuals.widgets.hovered.fg_stroke =
                            Stroke::new(0.0, Color32::TRANSPARENT);
                        style.visuals.widgets.hovered.bg_stroke =
                            Stroke::new(0.0, Color32::TRANSPARENT);
                        style.visuals.widgets.open.fg_stroke =
                            Stroke::new(0.0, Color32::TRANSPARENT);
                        style.visuals.widgets.open.bg_stroke =
                            Stroke::new(0.0, Color32::TRANSPARENT);

                        style.spacing.button_padding = [16.0, 16.0].into();

                        style.visuals.widgets.active.bg_fill = get_scheme().bg_secondary;
                        style.visuals.widgets.open.bg_fill = get_scheme().bg_secondary;
                        style.visuals.widgets.inactive.bg_fill = get_scheme().bg_secondary;
                        style.visuals.widgets.hovered.bg_fill = get_scheme().bg_secondary;
                        ui.add(
                            egui::Label::new(
                                RichText::new("IP:PORT")
                                    .color(get_scheme().text_tertiary)
                                    .size(12.),
                            )
                            .selectable(false),
                        );
                        ui.add_space(5.);
                        let text_edit = ui.add(
                            egui::TextEdit::singleline(&mut addr).margin(egui::Margin::same(16)),
                        );
                        let enter_key = text_edit.lost_focus()
                            && ui.ctx().input(|i| i.key_pressed(egui::Key::Enter));

                        if let Some(error) = &error {
                            ui.add(
                                egui::Label::new(
                                    RichText::new(error.to_string())
                                        .color(colors::REDDISH_DEFAULT)
                                        .size(12.),
                                )
                                .selectable(false),
                            );
                        }

                        *state.modal_state = ModalState::ConnectToIp {
                            addr: addr.clone(),
                            error,
                            connecting: None,
                        };
                        if let Some(ref status) = connecting {
                            match status.status() {
                                ConnectionStatus::NoConnection | ConnectionStatus::Connecting => {
                                    ui.add(
                                        egui::Label::new(
                                            RichText::new("Connecting")
                                                .color(get_scheme().highlight)
                                                .size(12.),
                                        )
                                        .selectable(false),
                                    );
                                    *state.modal_state = ModalState::ConnectToIp {
                                        addr: addr.clone(),
                                        error: None,
                                        connecting,
                                    };
                                }
                                ConnectionStatus::Success => {
                                    state.switch_to_main();
                                }
                                ConnectionStatus::Error => {
                                    *state.modal_state = ModalState::ConnectToIp {
                                        addr: addr.clone(),
                                        error: Some(ConnectError::Connection),
                                        connecting: None,
                                    };
                                }
                            }
                        }
                        ui.add_space(15.);
                        ui.allocate_ui_with_layout(
                            egui::vec2(322.0, 50.0),
                            egui::Layout::left_to_right(egui::Align::Center),
                            move |ui| {
                                if ui.add(EButton::gray("CANCEL").width(156.)).clicked() {
                                    *state.modal_state = ModalState::None;
                                }
                                ui.add_space(10.0);
                                if ui.add(EButton::green("CONNECT").width(156.)).clicked()
                                    || enter_key
                                {
                                    state.start_connect(addr);
                                }
                            },
                        )
                    });
            }
        }
    }
}

fn startup_button(
    label: impl ToString,
    icon: egui::TextureId,
) -> impl FnOnce(&mut egui::Ui) -> egui::Response {
    let label = label.to_string();
    move |ui| {
        ui.allocate_ui_with_layout(
            egui::vec2(298., 42.),
            egui::Layout::left_to_right(egui::Align::Center),
            move |ui| {
                let font_id = egui::TextStyle::Button.resolve(ui.style());
                let response =
                    ui.allocate_rect(ui.max_rect(), egui::Sense::CLICK | egui::Sense::HOVER);
                ui.painter().rect_filled(
                    ui.max_rect(),
                    corner_radius_sm(),
                    if response.is_pointer_button_down_on() {
                        get_scheme().bg_secondary
                    } else if response.hovered() {
                        get_scheme().bg_secondary.opacity(0.75)
                    } else {
                        get_scheme().bg_secondary
                    },
                );

                egui::Image::new(SizedTexture::new(icon, egui::vec2(20., 20.))).paint_at(
                    ui,
                    egui::Rect::from_center_size(
                        ui.max_rect().min + egui::vec2(26.0, ui.max_rect().height() / 2.0),
                        egui::vec2(20., 20.),
                    ),
                );

                ui.painter().text(
                    ui.max_rect().min + egui::vec2(46.0, ui.max_rect().height() / 2.0),
                    egui::Align2::LEFT_CENTER,
                    label,
                    font_id,
                    get_scheme().text_primary,
                );

                response
            },
        )
        .inner
    }
}

fn recent_item_button(
    item: RecentItem,
    arrow: egui::TextureId,
) -> impl FnOnce(&mut egui::Ui) -> egui::Response {
    move |ui| {
        ui.allocate_ui_with_layout(
            egui::vec2(322., 61.),
            egui::Layout::left_to_right(egui::Align::Center),
            move |ui| {
                let mut title_font_id = egui::TextStyle::Button.resolve(ui.style());
                title_font_id.size = 12.0;
                let mut subtitle_font_id = egui::TextStyle::Button.resolve(ui.style());
                subtitle_font_id.size = 10.0;
                let response =
                    ui.allocate_rect(ui.max_rect(), egui::Sense::CLICK | egui::Sense::HOVER);

                let bg_color = if response.is_pointer_button_down_on() {
                    get_scheme().bg_primary.opacity(0.75)
                } else if response.hovered() {
                    get_scheme().bg_primary.opacity(0.5)
                } else {
                    Color32::TRANSPARENT
                };

                ui.painter().rect_filled(
                    egui::Rect::from_min_max(
                        ui.max_rect().min,
                        ui.max_rect().max - egui::vec2(0., 1.),
                    ),
                    CornerRadius::ZERO,
                    bg_color,
                );

                egui::Image::new(SizedTexture::new(arrow, egui::vec2(24., 24.)))
                    .tint(get_scheme().text_primary)
                    .paint_at(
                        ui,
                        egui::Rect::from_center_size(
                            ui.max_rect().max
                                - egui::vec2(7.0 + 12.0, ui.max_rect().height() / 2.0),
                            egui::vec2(20., 20.),
                        ),
                    );

                let (title, subtitle) = match item {
                    RecentItem::File(path) => {
                        let path = path.canonicalize().unwrap_or(path);

                        let dir = path.parent().unwrap_or_else(|| &path);
                        let mut path_display = format!("{}", dir.display());
                        if let Some(dirs) = directories::UserDirs::new() {
                            if let Some(home_dir) = dirs.home_dir().to_str() {
                                path_display = path_display.replace(home_dir, "~");
                            }
                        }
                        let name = if path
                            .file_name()
                            .map(|name| name == "main.py")
                            .unwrap_or_default()
                        {
                            dir.file_stem()
                        } else {
                            path.file_stem()
                        }
                        .and_then(|f| f.to_str())
                        .unwrap_or("N/A");
                        (name.to_string(), path_display)
                    }
                    RecentItem::Addr(addr) => (addr, "IP ADDR".to_string()),
                };

                ui.painter().text(
                    ui.max_rect().min + egui::vec2(16.0, 24.0),
                    egui::Align2::LEFT_CENTER,
                    title,
                    title_font_id,
                    get_scheme().text_primary,
                );

                ui.painter().text(
                    ui.max_rect().min + egui::vec2(16.0, 16.0 + 24.0),
                    egui::Align2::LEFT_CENTER,
                    subtitle,
                    subtitle_font_id,
                    get_scheme().text_secondary,
                );

                ui.painter().rect_filled(
                    egui::Rect::from_min_max(
                        egui::Pos2::new(ui.max_rect().min.x, ui.max_rect().max.y - 1.0),
                        egui::Pos2::new(ui.max_rect().max.x, ui.max_rect().max.y),
                    ),
                    CornerRadius::ZERO,
                    if response.hovered() {
                        get_scheme().text_secondary.opacity(0.05)
                    } else {
                        get_scheme().text_secondary.opacity(0.1)
                    },
                );

                response
            },
        )
        .inner
    }
}

fn recent_files() -> RecentItems {
    let recent_file_path = dirs().data_dir().join("recent_items.postcard");
    let Ok(contents) = std::fs::read(recent_file_path) else {
        return RecentItems::default();
    };
    postcard::from_bytes(&contents).unwrap_or_default()
}

fn save_recent_files(paths: &RecentItems) {
    let dirs = dirs();
    let data_dir = dirs.data_dir();
    let _ = std::fs::create_dir_all(data_dir);
    let recent_file_path = data_dir.join("recent_items.postcard");
    let Ok(contents) = postcard::to_allocvec(&paths) else {
        return;
    };
    let _ = std::fs::write(recent_file_path, contents);
}
