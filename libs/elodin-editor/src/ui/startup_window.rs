use bevy::{
    camera::RenderTarget,
    ecs::system::SystemParam,
    prelude::*,
    window::{EnabledButtons, PrimaryWindow, WindowRef, WindowResolution},
};
use bevy_egui::{EguiContexts, EguiTextureHandle};
use egui::{Color32, CornerRadius, RichText, Stroke, load::SizedTexture};
use hifitime::Epoch;
use impeller_bevy::{
    ConnectionAddr, ConnectionStatus, CurrentStreamId, MsgPacketRx, MsgPacketTx, PacketRx,
    PacketTx, ThreadConnectionStatus,
};
#[cfg(not(target_family = "wasm"))]
use impeller_bevy::{spawn_msg_tcp_connect, spawn_tcp_connect};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::{
    collections::BTreeMap,
    net::{Ipv6Addr, SocketAddr, ToSocketAddrs},
    path::PathBuf,
    sync::Arc,
};

use crate::{VERSION, create_egui_context};
#[cfg(not(target_family = "wasm"))]
use crate::{dirs, ui::base_window};

use super::{
    button::EButton,
    colors::{self, ColorExt, get_scheme},
    images,
    theme::corner_radius_sm,
    widgets::{RootWidgetSystem, RootWidgetSystemExt},
};
use crate::ui::widgets::SystemStateExt;

#[derive(Component)]
pub struct StartupWindow;

/// Wasm draws the startup screen on the primary canvas. A second window is not
/// visible there: `Window::visible` is a no-op, and page CSS stretches every canvas.
#[derive(Resource)]
pub struct StartupActive;

/// Initial `?db=` attempt. While this exists the editor chrome stays hidden and
/// a connecting or failure dialog is shown instead.
#[derive(Resource, Clone)]
#[cfg_attr(not(target_family = "wasm"), allow(dead_code))]
pub struct UrlConnect {
    addr: String,
    failed: bool,
}

#[derive(Resource)]
struct MainWindowRevealDelay {
    frames_remaining: u8,
}

const MAIN_WINDOW_REVEAL_FRAMES: u8 = 2;

fn queue_main_window_reveal(commands: &mut Commands) {
    commands.insert_resource(MainWindowRevealDelay {
        frames_remaining: MAIN_WINDOW_REVEAL_FRAMES,
    });
}

fn create_startup_window(
    mut commands: Commands,
    status: Res<ThreadConnectionStatus>,
    connection_addr: Option<Res<ConnectionAddr>>,
    primary: Query<&Window, With<PrimaryWindow>>,
) {
    #[cfg(not(target_family = "wasm"))]
    let _ = connection_addr;
    #[cfg(not(target_family = "wasm"))]
    commands.insert_resource(recent_files());
    #[cfg(target_family = "wasm")]
    commands.insert_resource(RecentItems::default());
    if status.status() == ConnectionStatus::NoConnection {
        #[cfg(target_family = "wasm")]
        commands.insert_resource(StartupActive);
        #[cfg(not(target_family = "wasm"))]
        {
            let egui_context = create_egui_context();

            let mut window = commands.spawn((
                Window {
                    title: "Elodin".to_owned(),
                    resolution: WindowResolution::new(730, 470),
                    resize_constraints: WindowResizeConstraints {
                        min_width: 730.0,
                        min_height: 470.0,
                        max_width: 730.0,
                        max_height: 470.0,
                    },
                    enabled_buttons: EnabledButtons {
                        minimize: false,
                        maximize: false,
                        close: true,
                    },
                    ..base_window()
                },
                StartupWindow,
                egui_context,
                Camera2d,
            ));

            let camera = Camera::default();

            window.insert((camera, RenderTarget::Window(WindowRef::Entity(window.id()))));
        }
    } else if primary.single().is_ok() {
        queue_main_window_reveal(&mut commands);
    }
    #[cfg(target_family = "wasm")]
    if status.status() == ConnectionStatus::Connecting
        && let Some(addr) = connection_addr.as_ref()
    {
        commands.insert_resource(UrlConnect {
            addr: addr.endpoint().authority().to_string(),
            failed: false,
        });
    }
}

fn reveal_main_window(
    mut commands: Commands,
    reveal_delay: Option<ResMut<MainWindowRevealDelay>>,
    mut main_window: Query<&mut Window, (With<PrimaryWindow>, Without<StartupWindow>)>,
) {
    let Some(mut reveal_delay) = reveal_delay else {
        return;
    };
    if reveal_delay.frames_remaining > 0 {
        reveal_delay.frames_remaining -= 1;
        return;
    }

    if let Ok(mut window) = main_window.single_mut() {
        window.visible = true;
    }
    commands.remove_resource::<MainWindowRevealDelay>();
}

pub fn add_layouts(world: &mut World) {
    #[cfg(not(target_family = "wasm"))]
    world.add_root_widget_with::<StartupLayout, With<StartupWindow>>("startup_layout", ());
    #[cfg(target_family = "wasm")]
    if world.contains_resource::<StartupActive>() {
        world.add_root_widget::<StartupLayout>("startup_layout");
    }
}

#[cfg(target_family = "wasm")]
fn url_connect_feedback(world: &mut World) {
    if world.contains_resource::<StartupActive>() {
        return;
    }
    let Some(mut prompt) = world.get_resource::<UrlConnect>().cloned() else {
        return;
    };
    match world
        .get_resource::<ThreadConnectionStatus>()
        .map(|status| status.status())
    {
        Some(ConnectionStatus::Success) => {
            world.remove_resource::<UrlConnect>();
            return;
        }
        Some(ConnectionStatus::Error) if !prompt.failed => {
            impeller_bevy::suspend_ws_reconnect(world);
            prompt.failed = true;
            world.insert_resource(prompt.clone());
        }
        _ => {}
    }

    world.egui_context_scope::<_, With<PrimaryWindow>>(|world, ctx| {
        if !prompt.failed {
            ctx.request_repaint();
        }
        let dismissed = egui::Modal::new(egui::Id::new("url_connect"))
            .frame(connect_dialog_frame())
            .backdrop_color(get_scheme().bg_secondary.opacity(0.85))
            .show(&ctx, |ui| {
                if prompt.failed {
                    ui.add(
                        egui::Label::new(RichText::new("Couldn't connect").size(18.))
                            .selectable(false),
                    );
                    ui.add_space(8.);
                    ui.add(
                        egui::Label::new(
                            RichText::new(format!("No database is reachable at {}.", prompt.addr))
                                .color(colors::REDDISH_DEFAULT)
                                .size(12.),
                        )
                        .selectable(false),
                    );
                    ui.add_space(24.);
                    ui.add(EButton::green("OK").width(280.)).clicked()
                } else {
                    ui.add(
                        egui::Label::new(RichText::new("Connecting").size(18.)).selectable(false),
                    );
                    ui.add_space(8.);
                    ui.add(
                        egui::Label::new(
                            RichText::new(format!("Trying to reach {}.", prompt.addr))
                                .color(get_scheme().text_tertiary)
                                .size(12.),
                        )
                        .selectable(false),
                    );
                    ui.add_space(24.);
                    ui.add(EButton::gray("CANCEL").width(280.)).clicked()
                }
            });
        if dismissed.inner {
            return_to_connection_screen(world, &prompt.addr);
        }
    });
}

#[cfg(target_family = "wasm")]
fn return_to_connection_screen(world: &mut World, addr: &str) {
    impeller_bevy::suspend_ws_reconnect(world);
    if let Some(status) = world.get_resource::<ThreadConnectionStatus>() {
        status.set_status(ConnectionStatus::NoConnection);
    }
    if let Some(mut recent) = world.get_resource_mut::<RecentItems>()
        && let Ok(now) = Epoch::now()
    {
        recent.push(now, RecentItem::Addr(addr.to_string()));
    }
    world.remove_resource::<UrlConnect>();
    world.insert_resource(StartupActive);
}

#[cfg(target_family = "wasm")]
fn connect_dialog_frame() -> egui::Frame {
    egui::Frame {
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
    }
}

pub struct StartupPlugin;

impl Plugin for StartupPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, create_startup_window);
        #[cfg(target_family = "wasm")]
        app.add_systems(Update, url_connect_feedback.before(add_layouts));
        app.add_systems(Update, (add_layouts, reveal_main_window));
    }
}

#[derive(SystemParam)]
pub struct StartupLayout<'w, 's> {
    contexts: EguiContexts<'w, 's>,
    window: Query<'w, 's, Entity, With<StartupWindow>>,
    images: Local<'s, images::Images>,
    modal_state: Local<'s, ModalState>,
    packet_tx: ResMut<'w, PacketTx>,
    packet_rx: ResMut<'w, PacketRx>,
    msg_packet_tx: ResMut<'w, MsgPacketTx>,
    msg_packet_rx: ResMut<'w, MsgPacketRx>,
    current_stream_id: ResMut<'w, CurrentStreamId>,
    status: ResMut<'w, ThreadConnectionStatus>,
    recent_files: ResMut<'w, RecentItems>,
    commands: Commands<'w, 's>,
    #[cfg(target_family = "wasm")]
    ws_connect: MessageWriter<'w, impeller_bevy::WsConnect>,
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
        connecting: bool,
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
    #[cfg(not(target_family = "wasm"))]
    fn connect(&mut self, addr: SocketAddr, reconnect: bool) {
        let (packet_tx, packet_rx, outgoing_packet_rx, incoming_packet_tx) =
            impeller_bevy::channels::channels();
        let (msg_tx, msg_rx, msg_outgoing_rx, msg_incoming_tx) =
            impeller_bevy::channels::msg_channels();
        let stream_id = fastrand::u64(..);
        spawn_msg_tcp_connect(addr, msg_outgoing_rx, msg_incoming_tx);
        let status = spawn_tcp_connect(
            addr,
            outgoing_packet_rx,
            incoming_packet_tx,
            stream_id,
            reconnect,
            true,
        );
        self.commands.insert_resource(ConnectionAddr::new(addr));
        self.commands
            .insert_resource(impeller_bevy::ComponentFilteredStream::default());
        *self.current_stream_id = CurrentStreamId(stream_id);
        *self.packet_tx = packet_tx;
        *self.packet_rx = packet_rx;
        *self.msg_packet_tx = msg_tx;
        *self.msg_packet_rx = msg_rx;
        *self.status = status;
    }

    #[cfg(target_family = "wasm")]
    fn connect(&mut self, endpoint: impeller_bevy::DbEndpoint, _reconnect: bool) {
        self.ws_connect.write(impeller_bevy::WsConnect(endpoint));
    }

    fn switch_to_main(&mut self) {
        #[cfg(not(target_family = "wasm"))]
        {
            let e = self.window.single_mut();
            self.commands
                .entity(e.expect("Window entity should exist"))
                .despawn();
        }
        #[cfg(target_family = "wasm")]
        self.commands.remove_resource::<StartupActive>();
        queue_main_window_reveal(&mut self.commands);
    }

    #[cfg(not(target_family = "wasm"))]
    fn open_file(&mut self, file: PathBuf) {
        self.recent_files.push(
            hifitime::Epoch::now().unwrap(),
            RecentItem::File(file.clone()),
        );
        save_recent_files(&self.recent_files);

        #[cfg(not(target_os = "windows"))]
        {
            let cache_dir = dirs().cache_dir().to_owned();
            std::thread::spawn(move || {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(crate::run::run_recipe(
                    cache_dir,
                    file,
                    stellarator::util::CancelToken::default(),
                ))
                .unwrap();
            });
        }
        self.connect(SocketAddr::new(Ipv6Addr::UNSPECIFIED.into(), 2240), true);
        self.switch_to_main();
    }

    fn start_connect(&mut self, addr: String) {
        #[cfg(target_family = "wasm")]
        {
            let endpoint = match impeller_bevy::DbEndpoint::parse(&addr) {
                Ok(endpoint) => endpoint,
                Err(err) => {
                    *self.modal_state = ModalState::ConnectToIp {
                        addr,
                        error: Some(ConnectError::SocketParse(Arc::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            err,
                        )))),
                        connecting: false,
                    };
                    return;
                }
            };
            self.recent_files.push(
                hifitime::Epoch::now().unwrap(),
                RecentItem::Addr(addr.clone()),
            );
            self.connect(endpoint, false);
            *self.modal_state = ModalState::ConnectToIp {
                addr,
                error: None,
                connecting: true,
            };
            return;
        }
        #[cfg(not(target_family = "wasm"))]
        let socket_addr = match addr.to_socket_addrs().map(|mut x| x.next()) {
            Ok(Some(addr)) => addr,
            Ok(None) => {
                *self.modal_state = ModalState::ConnectToIp {
                    addr,
                    error: Some(ConnectError::ResolutionFailed),
                    connecting: false,
                };
                return;
            }

            Err(err) => {
                *self.modal_state = ModalState::ConnectToIp {
                    addr,
                    error: Some(ConnectError::SocketParse(Arc::new(err))),
                    connecting: false,
                };
                return;
            }
        };
        #[cfg(not(target_family = "wasm"))]
        {
            self.recent_files.push(
                hifitime::Epoch::now().unwrap(),
                RecentItem::Addr(addr.clone()),
            );
            save_recent_files(&self.recent_files);

            self.connect(socket_addr, false);
            *self.modal_state = ModalState::ConnectToIp {
                addr,
                error: None,
                connecting: true,
            };
        }
    }

    fn default_connect_addr(&self) -> String {
        #[cfg(target_family = "wasm")]
        if let Some(RecentItem::Addr(addr)) = self.recent_files.recent_files.values().next_back() {
            return addr.clone();
        }
        if cfg!(target_family = "wasm") || cfg!(target_os = "windows") {
            "127.0.0.1:2240".to_string()
        } else {
            "[::]:2240".to_string()
        }
    }

    fn left_pane(
        &mut self,
        ui: &mut egui::Ui,
        logo_full: egui::TextureId,
        #[allow(unused_variables)] folder: egui::TextureId,
        icon_ip_addr: egui::TextureId,
    ) {
        let avail = ui.available_size();
        let column = if cfg!(target_family = "wasm") {
            egui::vec2(avail.x.min(408.0), avail.y.min(470.0))
        } else {
            egui::vec2(408.0, 470.0)
        };
        ui.allocate_ui_with_layout(column, egui::Layout::top_down(egui::Align::Center), |ui| {
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
                RichText::new(format!("VERSION {}", VERSION)).color(get_scheme().text_secondary),
            );
            #[cfg(target_family = "wasm")]
            {
                ui.add_space(24.0);
                ui.label(
                    RichText::new("Open with ?db=host:port, or connect below.")
                        .color(get_scheme().text_secondary)
                        .size(12.),
                );
                ui.add_space(28.0);
            }
            #[cfg(not(target_family = "wasm"))]
            ui.add_space(75.0);
            #[cfg(not(any(target_os = "windows", target_family = "wasm")))]
            {
                if ui
                    .add(startup_button("Open Existing Project", folder))
                    .clicked()
                    && let Some(file) = rfd::FileDialog::new()
                        .add_filter("python", &["py"])
                        .add_filter("s10", &["toml"])
                        .pick_file()
                {
                    self.open_file(file);
                }
                ui.add_space(10.0);
            }
            if ui
                .add(startup_button("Connect to IP Address", icon_ip_addr))
                .clicked()
            {
                *self.modal_state = ModalState::ConnectToIp {
                    addr: self.default_connect_addr(),
                    error: None,
                    connecting: false,
                }
            }
        });
    }

    fn recent_pane(&mut self, ui: &mut egui::Ui, arrow: egui::TextureId) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            for item in self.recent_files.recent_files.clone().into_values().rev() {
                if ui.add(recent_item_button(item.clone(), arrow)).clicked() {
                    match item {
                        #[cfg(not(target_family = "wasm"))]
                        RecentItem::File(path) => {
                            self.open_file(path.clone());
                        }
                        #[cfg(target_family = "wasm")]
                        RecentItem::File(_) => {}
                        RecentItem::Addr(addr) => {
                            self.start_connect(addr);
                        }
                    }
                }
            }
        });
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
        let mut state = state.params_mut(world);
        let logo_full = state
            .contexts
            .add_image(EguiTextureHandle::Weak(state.images.logo_full.id()));
        let folder = state
            .contexts
            .add_image(EguiTextureHandle::Weak(state.images.icon_folder.id()));

        let arrow = state.contexts.add_image(EguiTextureHandle::Weak(
            state.images.icon_chevron_right.id(),
        ));
        let icon_ip_addr = state
            .contexts
            .add_image(EguiTextureHandle::Weak(state.images.icon_ip_addr.id()));

        #[cfg(not(target_family = "wasm"))]
        {
            super::utils::show_central_panel(
                egui::CentralPanel::default().frame(egui::Frame::NONE),
                ctx,
                |ui| state.left_pane(ui, logo_full, folder, icon_ip_addr),
            );
            super::utils::show_panel(
                egui::Panel::right("right")
                    .exact_size(322.0)
                    .frame(egui::Frame::NONE.fill(get_scheme().bg_secondary))
                    .resizable(false),
                ctx,
                |ui| state.recent_pane(ui, arrow),
            );
        }
        #[cfg(target_family = "wasm")]
        {
            super::utils::show_central_panel(
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE.fill(get_scheme().bg_secondary)),
                ctx,
                |ui| {
                    let avail = ui.available_rect_before_wrap();
                    let show_recent = avail.width() >= 762.0;
                    let size = if show_recent {
                        egui::vec2(730.0, 470.0)
                    } else {
                        egui::vec2(
                            440.0_f32.min(avail.width() - 24.0).max(280.0),
                            470.0_f32.min(avail.height() - 24.0).max(320.0),
                        )
                    };
                    let card = egui::Rect::from_center_size(avail.center(), size);
                    ui.painter().rect_filled(
                        card,
                        egui::CornerRadius::same(8),
                        get_scheme().bg_primary,
                    );
                    ui.painter().rect_stroke(
                        card,
                        egui::CornerRadius::same(8),
                        egui::Stroke::new(1.0_f32, get_scheme().text_tertiary),
                        egui::StrokeKind::Inside,
                    );
                    let (left, right) = if show_recent {
                        let right = egui::Rect::from_min_max(
                            egui::pos2(card.max.x - 322.0, card.min.y),
                            card.max,
                        );
                        ui.painter().rect_filled(
                            right,
                            egui::CornerRadius::ZERO,
                            get_scheme().bg_secondary,
                        );
                        (
                            egui::Rect::from_min_max(card.min, egui::pos2(right.min.x, card.max.y)),
                            Some(right),
                        )
                    } else {
                        (card, None)
                    };
                    if let Some(right) = right {
                        ui.scope_builder(egui::UiBuilder::new().max_rect(right), |ui| {
                            state.recent_pane(ui, arrow);
                        });
                    }
                    ui.scope_builder(egui::UiBuilder::new().max_rect(left), |ui| {
                        state.left_pane(ui, logo_full, folder, icon_ip_addr);
                    });
                },
            );
        }
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
                            Stroke::new(0.0_f32, Color32::TRANSPARENT);
                        style.visuals.widgets.active.bg_stroke =
                            Stroke::new(0.0_f32, Color32::TRANSPARENT);
                        style.visuals.widgets.hovered.fg_stroke =
                            Stroke::new(0.0_f32, Color32::TRANSPARENT);
                        style.visuals.widgets.hovered.bg_stroke =
                            Stroke::new(0.0_f32, Color32::TRANSPARENT);
                        style.visuals.widgets.open.fg_stroke =
                            Stroke::new(0.0_f32, Color32::TRANSPARENT);
                        style.visuals.widgets.open.bg_stroke =
                            Stroke::new(0.0_f32, Color32::TRANSPARENT);

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
                            connecting: false,
                        };
                        if connecting {
                            match state.status.status() {
                                ConnectionStatus::NoConnection | ConnectionStatus::Connecting => {
                                    ui.add(
                                        egui::Label::new(
                                            RichText::new("Connecting")
                                                .color(get_scheme().highlight)
                                                .size(12.),
                                        )
                                        .selectable(false),
                                    );
                                    ui.ctx().request_repaint();
                                    *state.modal_state = ModalState::ConnectToIp {
                                        addr: addr.clone(),
                                        error: None,
                                        connecting: true,
                                    };
                                }
                                ConnectionStatus::Success => {
                                    state.switch_to_main();
                                }
                                ConnectionStatus::Error => {
                                    *state.modal_state = ModalState::ConnectToIp {
                                        addr: addr.clone(),
                                        error: Some(ConnectError::Connection),
                                        connecting: false,
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
                        #[cfg(not(target_family = "wasm"))]
                        if let Some(dirs) = directories::UserDirs::new()
                            && let Some(home_dir) = dirs.home_dir().to_str()
                        {
                            path_display = path_display.replace(home_dir, "~");
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

#[cfg(not(target_family = "wasm"))]
fn recent_files() -> RecentItems {
    let recent_file_path = dirs().data_dir().join("recent_items.postcard");
    let Ok(contents) = std::fs::read(recent_file_path) else {
        return RecentItems::default();
    };
    postcard::from_bytes(&contents).unwrap_or_default()
}

#[cfg(not(target_family = "wasm"))]
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
