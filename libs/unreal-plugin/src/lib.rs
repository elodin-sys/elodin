use bevy::prelude::*;
use bevy::transform::components::Transform;
use impeller::{
    bevy::{ImpellerSubscribePlugin, Subscriptions},
    bevy_sync::SyncPlugin,
    client::MsgPair,
    server::handle_socket,
    well_known::{self, WorldPos},
};
use std::net::SocketAddr;
use tokio::net::TcpStream;
use unreal_api::{
    core::UnrealEntityCommandsExt,
    ffi,
    module::{InitUserModule, UserModule},
    Component,
};

pub struct ElodinModule;

#[derive(Default, Debug, Component)]
#[uuid = "8d2df877-499b-46f3-9660-bd2e1867af0d"]
pub struct CameraMarker;

#[allow(clippy::type_complexity)]
fn spawn_camera(
    query: Query<(Entity, &Transform), (With<well_known::Camera>, Without<CameraMarker>)>,
    mut commands: Commands,
) {
    for (entity, transform) in query.iter() {
        commands
            .entity(entity)
            .insert(CameraMarker)
            .insert_actor(ffi::ActorClass::CameraActor, *transform)
            .set_view_target();
    }
}

impl InitUserModule for ElodinModule {
    fn init() -> Self {
        Self {}
    }
}

impl UserModule for ElodinModule {
    fn init(&self, app: &mut App) {
        let addr = "127.0.0.1:2240".parse().expect("failed to parse address");
        let (sub, bevy_tx) = ImpellerSubscribePlugin::pair();
        app.add_plugins(SimClient { addr, bevy_tx })
            .add_plugins(SyncPlugin {
                plugin: sub,
                subscriptions: Subscriptions::default(),
                enable_pbr: false,
            });
        app.add_systems(Update, sync_pos);
        app.add_systems(Update, spawn_camera);
    }
}

unreal_api::implement_unreal_module!(ElodinModule);

#[derive(Clone)]
struct SimClient {
    addr: SocketAddr,
    bevy_tx: flume::Sender<MsgPair>,
}

impl Plugin for SimClient {
    fn build(&self, _: &mut App) {
        let c = self.clone();
        std::thread::spawn(move || {
            println!("spawn tokio");
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio runtime failed to start");
            rt.block_on(async move {
                loop {
                    let Ok(socket) = TcpStream::connect(c.addr).await else {
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        continue;
                    };
                    let (rx_socket, tx_socket) = socket.into_split();

                    if let Err(err) = handle_socket(
                        c.bevy_tx.clone(),
                        tx_socket,
                        rx_socket,
                        std::iter::empty(),
                        std::iter::empty(),
                    )
                    .await
                    {
                        println!("socket error {err:?}");
                    }
                }
            });
        });
    }
}

pub fn sync_pos(mut query: Query<(&mut Transform, &WorldPos)>) {
    query.iter_mut().for_each(|(mut transform, pos)| {
        let WorldPos { pos, att } = pos;
        let [x, y, z] = pos.parts().map(|x| x.into_buf());
        let [i, j, k, w] = att.parts().map(|x| x.into_buf());
        *transform = bevy::prelude::Transform {
            translation: Vec3::new(x as f32, y as f32, z as f32),
            rotation: Quat::from_xyzw(i as f32, j as f32, k as f32, w as f32),
            ..Default::default()
        }
    });
}
