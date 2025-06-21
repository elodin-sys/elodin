use std::{net::SocketAddr, time::Instant};

use bevy::{
    ecs::system::{Local, SystemParam},
    log::warn,
    prelude::{Commands, Component, Entity, Query, Res, Resource},
};
use egui::{CornerRadius, RichText};
use impeller2_bevy::{ConnectionAddr, ConnectionStatus, ThreadConnectionStatus};
use impeller2_cli::mlua::MultiValue;

use super::{
    button::EButton,
    colors::{ColorExt, get_scheme},
    widgets::WidgetSystem,
};

#[derive(Resource)]
pub struct LuaActor {
    tx: flume::Sender<(String, flume::Sender<Result<String, String>>)>,
}

impl LuaActor {
    pub fn spawn(addr: SocketAddr) -> Self {
        let (cmd_tx, cmd_rx) =
            flume::unbounded::<(String, flume::Sender<Result<String, String>>)>();
        stellarator::struc_con::stellar(move || async move {
            let lua = impeller2_cli::lua().unwrap();
            let client = match impeller2_cli::Client::connect(addr).await {
                Ok(c) => c,
                Err(err) => {
                    warn!(?err, "lua client couldn't connect");
                    return;
                }
            };
            if let Err(err) = lua.globals().set("client", client) {
                warn!(?err, "error spawning lua client");
                return;
            }
            loop {
                if let Ok((cmd, res_tx)) = cmd_rx.recv() {
                    match lua.load(&cmd).eval_async::<MultiValue>().await {
                        Ok(values) => {
                            let val = values
                                .iter()
                                .map(|value| format!("{:#?}", value))
                                .collect::<Vec<_>>()
                                .join("\t");
                            let _ = res_tx.send(Ok(val));
                        }
                        Err(err) => {
                            let _ = res_tx.send(Err(err.to_string()));
                        }
                    }
                }
            }
        });
        LuaActor { tx: cmd_tx }
    }

    pub fn send(&self, cmd: String) -> flume::Receiver<Result<String, String>> {
        let (tx, rx) = flume::bounded(1);
        let _ = self.tx.send((cmd, tx));
        rx
    }
}

#[derive(Component)]
pub struct ActionTile {
    pub button_name: String,
    pub lua: String,
    pub status: Status,
}

#[derive(Clone, Default)]
pub enum Status {
    #[default]
    Pending,
    Sent {
        sent: Instant,
        rx: flume::Receiver<Result<String, String>>,
    },
    Completed(Result<String, String>),
}

#[derive(SystemParam)]
pub struct ActionTileWidget<'w, 's> {
    action_tiles: Query<'w, 's, &'static mut ActionTile>,
    lua: Option<Res<'w, LuaActor>>,
}

impl WidgetSystem for ActionTileWidget<'_, '_> {
    type Args = Entity;

    type Output = ();

    fn ui_system(
        world: &mut bevy::prelude::World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) -> Self::Output {
        let mut state = state.get_mut(world);
        let mut tile = state.action_tiles.get_mut(args).unwrap();
        egui::Frame::NONE
            .inner_margin(egui::Margin::same(32))
            .show(ui, |ui| {
                let button = ui.add(match &tile.status {
                    Status::Pending | Status::Completed(Ok(_)) => EButton::green(&tile.button_name),
                    Status::Sent { .. } => EButton::green(&tile.button_name).loading(true),
                    Status::Completed(Err(_)) => EButton::red(&tile.button_name),
                });

                if button.clicked() {
                    if let Some(lua) = state.lua {
                        let rx = lua.send(tile.lua.clone());
                        tile.status = Status::Sent {
                            rx,
                            sent: Instant::now(),
                        };
                    }
                }
                ui.add_space(32.0);
                match tile.status.clone() {
                    Status::Pending => {}
                    Status::Sent { sent, rx } => {
                        if sent.elapsed() > std::time::Duration::from_secs(10) {
                            tile.status = Status::Pending;
                        }
                        if let Ok(res) = rx.try_recv() {
                            tile.status = Status::Completed(res);
                        }
                    }
                    Status::Completed(Err(err)) => {
                        let modal = egui::Modal::new(egui::Id::new("connect_to_ip"))
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
                            .show(ui.ctx(), |ui| {
                                ui.add(
                                    egui::Label::new(
                                        RichText::new("Error Sending Action").size(18.),
                                    )
                                    .selectable(false),
                                );
                                ui.add_space(5.);
                                ui.colored_label(get_scheme().error, err);
                            });
                        if modal.should_close() {
                            tile.status = Status::Pending
                        }
                    }
                    _ => {}
                }
            });
    }
}

pub fn spawn_lua_actor(
    lua: Option<Res<LuaActor>>,
    addr: Option<Res<ConnectionAddr>>,
    status: Res<ThreadConnectionStatus>,
    mut last_status: Local<Option<ConnectionStatus>>,
    mut commands: Commands,
) {
    let status = status.status();
    if *last_status == Some(status) {
        *last_status = Some(status);
        return;
    }
    *last_status = Some(status);
    if let Some(addr) = addr {
        if lua.is_none() && status == ConnectionStatus::Success {
            commands.insert_resource(LuaActor::spawn(addr.0));
        }
    }
    if lua.is_some() {
        match status {
            impeller2_bevy::ConnectionStatus::NoConnection
            | impeller2_bevy::ConnectionStatus::Error => {
                commands.remove_resource::<LuaActor>();
            }
            _ => {}
        }
    }
}
