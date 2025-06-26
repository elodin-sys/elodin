use elodin_db::{DB, State, handle_conn};
use impeller2::types::{ComponentId, Timestamp};
use impeller2_wkt::{ComponentMetadata, EntityMetadata};
use nox_ecs::Error;
use std::{
    sync::{Arc, atomic},
    time::{Duration, Instant},
};
use stellarator::struc_con::{Joinable, Thread};
use tracing::warn;

use crate::{Compiled, World, WorldExec};

pub struct Server {
    db: elodin_db::Server,
    world: WorldExec<Compiled>,
}

impl Server {
    pub fn new(db: elodin_db::Server, world: WorldExec<Compiled>) -> Self {
        Self { db, world }
    }

    pub async fn run(self) -> Result<(), Error> {
        tracing::info!("running server");
        self.run_with_cancellation(|| false).await
    }

    pub async fn run_with_cancellation(
        self,
        is_cancelled: impl Fn() -> bool + 'static,
    ) -> Result<(), Error> {
        tracing::info!("running server with cancellation");
        let Self { db, mut world } = self;
        let elodin_db::Server { listener, db } = db;
        let start_time = Timestamp::now();
        init_db(&db, &mut world.world, start_time)?;
        let tick_db = db.clone();
        let stream: Thread<Option<Result<(), Error>>> =
            stellarator::struc_con::stellar(move || async move {
                let mut handles = vec![];
                loop {
                    let stream = listener.accept().await?;
                    handles.push(stellarator::spawn(handle_conn(stream, db.clone())).drop_guard());
                }
            });
        let tick = stellarator::spawn(tick(tick_db, world, is_cancelled, start_time));
        futures_lite::future::race(async { stream.join().await.unwrap().unwrap() }, async {
            tick.await
                .map_err(|_| stellarator::Error::JoinFailed)
                .map_err(Error::from)
        })
        .await
    }
}

pub fn init_db(
    db: &elodin_db::DB,
    world: &mut World,
    start_timestamp: Timestamp,
) -> Result<(), elodin_db::Error> {
    tracing::info!("initializing db");
    for (id, asset) in world.assets.iter().enumerate() {
        db.insert_asset(id as u64, &asset.inner)?;
    }
    db.with_state_mut(|state| {
        for (component_id, (schema, component_metadata)) in world.metadata.component_map.iter() {
            let Some(column) = world.host.get(component_id) else {
                continue;
            };
            let size = schema.size();
            let entity_ids =
                bytemuck::try_cast_slice::<_, u64>(column.entity_ids.as_slice()).unwrap();
            for (i, entity_id) in entity_ids.iter().enumerate() {
                let offset = i * size;
                let entity_id = impeller2::types::EntityId(*entity_id);
                let entity_metadata = world
                    .metadata
                    .entity_metadata
                    .entry(entity_id)
                    .or_insert_with(|| EntityMetadata {
                        entity_id,
                        name: format!("entity{}", entity_id),
                        metadata: Default::default(),
                    });
                let pair_name = format!("{}.{}", entity_metadata.name, component_metadata.name);
                let pair_id = ComponentId::new(&pair_name);
                let pair_metadata = ComponentMetadata {
                    component_id: pair_id,
                    name: pair_name,
                    metadata: component_metadata.metadata.clone(),
                    asset: component_metadata.asset,
                };

                state.set_component_metadata(pair_metadata, &db.path)?;
                state.insert_component(pair_id, schema.clone(), &db.path)?;
                let component = state.get_component(pair_id).unwrap();
                let buf = &column.buffer[offset..offset + size];
                component.time_series.push_buf(start_timestamp, buf)?;
            }
        }
        for entity_metadata in world.entity_metadata().values() {
            state.set_component_metadata(
                ComponentMetadata {
                    component_id: ComponentId::new(&entity_metadata.name),
                    name: entity_metadata.name.clone(),
                    metadata: entity_metadata.metadata.clone(),
                    asset: false,
                },
                &db.path,
            )?;
        }
        Ok::<_, elodin_db::Error>(())
    })?;

    db.time_step.store(
        world.metadata.run_time_step.0.as_nanos() as u64,
        atomic::Ordering::SeqCst,
    );
    let default_stream_time_step = Duration::from_secs_f64(
        world.metadata.sim_time_step.0.as_secs_f64() / world.metadata.default_playback_speed,
    );
    db.default_stream_time_step.store(
        default_stream_time_step.as_nanos() as u64,
        atomic::Ordering::SeqCst,
    );
    let _ = db.save_db_state();

    Ok(())
}

pub fn copy_db_to_world(state: &State, world: &mut WorldExec<Compiled>) {
    let world = &mut world.world;
    for (component_id, (schema, _)) in world.metadata.component_map.iter() {
        let Some(column) = world.host.get_mut(component_id) else {
            continue;
        };
        let entity_ids = bytemuck::try_cast_slice::<_, u64>(column.entity_ids.as_slice()).unwrap();
        let size = schema.size();
        for (i, entity_id) in entity_ids.iter().enumerate() {
            let offset = i * size;
            let entity_id = impeller2::types::EntityId(*entity_id);
            let Some(entity_metadata) = world.metadata.entity_metadata.get(&entity_id) else {
                continue;
            };
            let Some((_, component_metadata)) = world.metadata.component_map.get(component_id)
            else {
                continue;
            };

            let pair_name = format!("{}.{}", entity_metadata.name, component_metadata.name);
            let pair_id = ComponentId::new(&pair_name);
            let Some(component) = state.get_component(pair_id) else {
                continue;
            };
            let (_, head) = component.time_series.latest().unwrap();
            column.buffer[offset..offset + size].copy_from_slice(head);
        }
    }
}

pub fn commit_world_head(
    state: &State,
    world: &mut WorldExec<Compiled>,
    timestamp: Timestamp,
) -> Result<(), Error> {
    for (component_id, (schema, _)) in world.world.metadata.component_map.iter() {
        let Some(column) = world.world.host.get_mut(component_id) else {
            continue;
        };
        let entity_ids = bytemuck::try_cast_slice::<_, u64>(column.entity_ids.as_slice()).unwrap();
        let size = schema.size();
        for (i, entity_id) in entity_ids.iter().enumerate() {
            let offset = i * size;
            let entity_id = impeller2::types::EntityId(*entity_id);
            let Some(entity_metadata) = world.world.metadata.entity_metadata.get(&entity_id) else {
                continue;
            };
            let Some((_, component_metadata)) =
                world.world.metadata.component_map.get(component_id)
            else {
                continue;
            };
            let pair_name = format!("{}.{}", entity_metadata.name, component_metadata.name);
            let pair_id = ComponentId::new(&pair_name);
            let Some(component) = state.get_component(pair_id) else {
                continue;
            };
            let buf = &column.buffer[offset..offset + size];
            component.time_series.push_buf(timestamp, buf)?;
        }
    }
    Ok(())
}

async fn tick(
    db: Arc<DB>,
    mut world: WorldExec<Compiled>,
    is_cancelled: impl Fn() -> bool + 'static,
    mut timestamp: Timestamp,
) {
    let mut start = Instant::now();
    let mut tick = 0;
    while db.recording_cell.wait().await {
        if tick >= world.world.max_tick() {
            db.recording_cell.set_playing(false);
            world.world.metadata.max_tick = u64::MAX;
        }
        db.with_state(|state| copy_db_to_world(state, &mut world));
        if let Err(err) = world.run() {
            warn!(?err, "error ticking world");
        }
        db.with_state(|state| {
            if let Err(err) = commit_world_head(state, &mut world, timestamp) {
                warn!(?err, "error committing head");
            }
        });
        db.last_updated.store(timestamp);
        let time_step = db.time_step().max(Duration::from_micros(100));
        let sleep_time = time_step.saturating_sub(start.elapsed());
        if is_cancelled() {
            return;
        }
        stellarator::sleep(sleep_time).await;
        let now = Instant::now();
        while start < now {
            start += time_step;
        }
        tick += 1;
        timestamp += world.world.sim_time_step().0;
    }
}
