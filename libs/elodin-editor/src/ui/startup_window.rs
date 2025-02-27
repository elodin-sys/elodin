use bevy::{
    ecs::system::SystemParam,
    prelude::*,
    utils::HashMap,
    window::{EnabledButtons, PresentMode, PrimaryWindow, WindowResolution, WindowTheme},
};
use bevy_egui::EguiContexts;
use egui::{load::SizedTexture, Color32, CornerRadius, RichText, Stroke};
use hifitime::Epoch;
use impeller2_bevy::{
    spawn_tcp_connect, ConnectionStatus, CurrentStreamId, PacketRx, PacketTx,
    ThreadConnectionStatus,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    net::{Ipv6Addr, SocketAddr},
    path::PathBuf,
};

use crate::VERSION;

use super::{
    colors::{self, ColorExt},
    images,
    theme::{self, corner_radius_sm},
    widgets::{button::EButton, RootWidgetSystem, RootWidgetSystemExt},
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
    } else {
        let mut primary = primary.single_mut();
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
    recent_files: ResMut<'w, RecentFiles>,
    commands: Commands<'w, 's>,
}

#[derive(Resource, Serialize, Deserialize, Default)]
struct RecentFiles {
    recent_files: BTreeMap<Epoch, PathBuf>,
    index: HashMap<PathBuf, Epoch>,
}

impl RecentFiles {
    fn push(&mut self, epoch: Epoch, path: PathBuf) {
        if let Some(existing_epoch) = self.index.insert(path.clone(), epoch) {
            self.recent_files.remove(&existing_epoch);
        }
        self.recent_files.insert(epoch, path);
    }
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
    SocketParse(std::net::AddrParseError),
    Connection,
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
        *self.current_stream_id = CurrentStreamId(stream_id);
        *self.packet_tx = packet_tx;
        *self.packet_rx = packet_rx;
        *self.status = status.clone();
        status
    }

    fn switch_to_main(&mut self) {
        let e = self.window.single_mut();
        self.commands.entity(e).despawn();
        self.main_window.single_mut().visible = true;
    }

    fn open_file(&mut self, file: PathBuf) {
        let dirs = dirs();

        let cache_dir = dirs.cache_dir().to_owned();
        self.recent_files
            .push(hifitime::Epoch::now().unwrap(), file.clone());
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
                            colors::BLACK_BLACK_600,
                        );
                        ui.add_space(87.);
                        ui.add(egui::Image::new(SizedTexture::new(
                            logo_full,
                            egui::vec2(113., 24.),
                        )));
                        ui.add_space(10.0);
                        ui.label(
                            RichText::new(format!("VERSION {}", VERSION))
                                .color(colors::PRIMARY_CREAME_6),
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
            .frame(egui::Frame::NONE)
            .resizable(false)
            .show(ctx, |ui| {
                for path in state.recent_files.recent_files.clone().into_values().rev() {
                    if ui.add(recent_file_button(path.clone(), arrow)).clicked() {
                        state.open_file(path.clone());
                    }
                }
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
                        fill: colors::BLACK_BLACK_600,
                        stroke: egui::Stroke::NONE,
                        inner_margin: egui::Margin::same(24),
                        outer_margin: egui::Margin::symmetric(0, 0),
                        corner_radius: CornerRadius::same(8),
                        shadow: egui::Shadow {
                            color: colors::PRIMARY_SMOKE.opacity(0.2),
                            spread: 3,
                            blur: 32,
                            offset: [0, 5],
                        },
                    })
                    .backdrop_color(colors::SURFACE_SECONDARY.opacity(0.85))
                    .show(ctx, |ui| {
                        ui.add(
                            egui::Label::new(RichText::new("Connect to IP Address").size(18.))
                                .selectable(false),
                        );
                        ui.add_space(5.);
                        ui.add(
                            egui::Label::new(
                                RichText::new("Enter the IP address of a DB or Sim")
                                    .color(colors::PRIMARY_ONYX_5)
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

                        style.visuals.widgets.active.bg_fill = colors::SURFACE_SECONDARY;
                        style.visuals.widgets.open.bg_fill = colors::SURFACE_SECONDARY;
                        style.visuals.widgets.inactive.bg_fill = colors::SURFACE_SECONDARY;
                        style.visuals.widgets.hovered.bg_fill = colors::SURFACE_SECONDARY;
                        ui.add(
                            egui::Label::new(
                                RichText::new("IP:PORT")
                                    .color(colors::PRIMARY_ONYX_5)
                                    .size(12.),
                            )
                            .selectable(false),
                        );
                        ui.add_space(5.);
                        ui.add(
                            egui::TextEdit::singleline(&mut addr).margin(egui::Margin::same(16)),
                        );
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
                                                .color(colors::HYPERBLUE_DEFAULT)
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
                                if ui.add(EButton::green("CONNECT").width(156.)).clicked() {
                                    let socket_addr = match addr.parse() {
                                        Ok(addr) => addr,
                                        Err(err) => {
                                            *state.modal_state = ModalState::ConnectToIp {
                                                addr,
                                                error: Some(ConnectError::SocketParse(err)),
                                                connecting: None,
                                            };
                                            return;
                                        }
                                    };
                                    let status = state.connect(socket_addr, false);
                                    *state.modal_state = ModalState::ConnectToIp {
                                        addr,
                                        error: None,
                                        connecting: Some(status),
                                    };
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
                        colors::PRIMARY_SMOKE
                    } else if response.hovered() {
                        colors::PRIMARY_SMOKE.opacity(0.75)
                    } else {
                        colors::SURFACE_SECONDARY
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
                    colors::PRIMARY_CREAME,
                );

                response
            },
        )
        .inner
    }
}

fn recent_file_button(
    path: PathBuf,
    arrow: egui::TextureId,
) -> impl FnOnce(&mut egui::Ui) -> egui::Response {
    move |ui| {
        ui.allocate_ui_with_layout(
            egui::vec2(322., 61.),
            egui::Layout::left_to_right(egui::Align::Center),
            move |ui| {
                let mut name_font_id = egui::TextStyle::Button.resolve(ui.style());
                name_font_id.size = 12.0;
                let mut path_font_id = egui::TextStyle::Button.resolve(ui.style());
                path_font_id.size = 10.0;
                let response =
                    ui.allocate_rect(ui.max_rect(), egui::Sense::CLICK | egui::Sense::HOVER);

                egui::Image::new(SizedTexture::new(arrow, egui::vec2(24., 24.)))
                    .tint(colors::WHITE)
                    .paint_at(
                        ui,
                        egui::Rect::from_center_size(
                            ui.max_rect().max
                                - egui::vec2(7.0 + 12.0, ui.max_rect().height() / 2.0),
                            egui::vec2(20., 20.),
                        ),
                    );

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

                ui.painter().text(
                    ui.max_rect().min + egui::vec2(16.0, 24.0),
                    egui::Align2::LEFT_CENTER,
                    name,
                    name_font_id,
                    colors::PRIMARY_CREAME,
                );

                ui.painter().text(
                    ui.max_rect().min + egui::vec2(16.0, 16.0 + 24.0),
                    egui::Align2::LEFT_CENTER,
                    path_display,
                    path_font_id,
                    colors::PRIMARY_CREAME_6,
                );

                ui.painter().rect_filled(
                    egui::Rect::from_min_max(
                        egui::Pos2::new(ui.max_rect().min.x, ui.max_rect().max.y - 1.0),
                        egui::Pos2::new(ui.max_rect().max.x, ui.max_rect().max.y),
                    ),
                    CornerRadius::ZERO,
                    if response.hovered() {
                        colors::PRIMARY_CREAME_6.opacity(0.05)
                    } else {
                        colors::PRIMARY_CREAME_6.opacity(0.1)
                    },
                );

                let overlay_rect = egui::Rect::from_min_max(
                    ui.max_rect().min,
                    ui.max_rect().max - egui::vec2(0., 1.),
                );
                if response.is_pointer_button_down_on() {
                    ui.painter().rect_filled(
                        overlay_rect,
                        CornerRadius::ZERO,
                        colors::PRIMARY_SMOKE.opacity(0.75),
                    );
                } else if response.hovered() {
                    ui.painter().rect_filled(
                        overlay_rect,
                        CornerRadius::ZERO,
                        colors::PRIMARY_SMOKE.opacity(0.3),
                    );
                }

                response
            },
        )
        .inner
    }
}

fn dirs() -> directories::ProjectDirs {
    directories::ProjectDirs::from("systems", "elodin", "editor").unwrap()
}

fn recent_files() -> RecentFiles {
    let recent_file_path = dirs().data_dir().join("recent_files.toml");
    let Ok(contents) = std::fs::read_to_string(recent_file_path) else {
        return RecentFiles::default();
    };
    toml::from_str(&contents).unwrap_or_default()
}

fn save_recent_files(paths: &RecentFiles) {
    let dirs = dirs();
    let data_dir = dirs.data_dir();
    let _ = std::fs::create_dir_all(data_dir);
    let recent_file_path = data_dir.join("recent_files.toml");
    let Ok(contents) = toml::to_string(&paths) else {
        return;
    };
    let _ = std::fs::write(recent_file_path, contents);
}
