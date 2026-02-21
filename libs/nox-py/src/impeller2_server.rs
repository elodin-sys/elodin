use crate::Error;
use bytemuck;
use elodin_db::{AtomicTimestampExt, DB, State, handle_conn};
use impeller2::types::{ComponentId, Timestamp};
use impeller2_wkt::{ComponentMetadata, EntityMetadata};
use std::{
    collections::HashSet,
    sync::{
        Arc,
        atomic::{self, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
use stellarator::struc_con::{Joinable, Thread};
use stellarator::util::CancelToken;
use tracing::{info, warn};

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
        let cancel_token = CancelToken::new();
        self.run_with_cancellation(
            || false,
            |_, _, _, _, _| {},
            |_, _, _, _, _| {},
            false,
            None,
            cancel_token,
        )
        .await
    }

    pub async fn run_with_cancellation(
        self,
        is_cancelled: impl Fn() -> bool + 'static,
        pre_step: impl Fn(u64, &Arc<DB>, &Arc<AtomicU64>, Timestamp, Timestamp) + 'static,
        post_step: impl Fn(u64, &Arc<DB>, &Arc<AtomicU64>, Timestamp, Timestamp) + 'static,
        interactive: bool,
        start_timestamp: Option<Timestamp>,
        cancel_token: CancelToken,
    ) -> Result<(), Error> {
        tracing::info!("running server with cancellation");
        let Self { db, mut world } = self;
        let elodin_db::Server { listener, db } = db;
        let start_time = start_timestamp.unwrap_or_else(Timestamp::now);
        init_db(&db, &mut world.world, start_time)?;
        let tick_db = db.clone();
        let tick_counter = Arc::new(AtomicU64::new(0));
        let stream: Thread<Option<Result<(), Error>>> =
            stellarator::struc_con::stellar(move || async move {
                loop {
                    let stream = listener.accept().await?;
                    let conn_db = db.clone();
                    stellarator::struc_con::stellar(move || handle_conn(stream, conn_db));
                }
            });
        let tick = stellarator::spawn(tick(
            tick_db,
            tick_counter,
            world,
            is_cancelled,
            pre_step,
            post_step,
            start_time,
            interactive,
            cancel_token,
        ));
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
    db.set_earliest_timestamp(start_timestamp)?;
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
                };

                state.set_component_metadata(pair_metadata, &db.path)?;
                state.insert_component(pair_id, schema.clone(), &db.path)?;
                let component = state.get_component(pair_id).unwrap();
                let buf = &column.buffer[offset..offset + size];
                component.time_series.push_buf(start_timestamp, buf)?;
            }
            if let Some(path) = &world.metadata.schematic_path {
                state
                    .db_config
                    .set_schematic_path(path.to_string_lossy().to_string());
            }
            if let Some(content) = &world.metadata.schematic {
                state.db_config.set_schematic_content(content.clone());
            }
        }
        for entity_metadata in world.entity_metadata().values() {
            state.set_component_metadata(
                ComponentMetadata {
                    component_id: ComponentId::new(&entity_metadata.name),
                    name: entity_metadata.name.clone(),
                    metadata: entity_metadata.metadata.clone(),
                },
                &db.path,
            )?;
        }
        Ok::<_, elodin_db::Error>(())
    })?;

    const PLAYBACK_FREQUENCY: f64 = 60.0;
    let default_stream_time_step =
        Duration::from_secs_f64(world.metadata.default_playback_speed / PLAYBACK_FREQUENCY);
    db.default_stream_time_step.store(
        default_stream_time_step.as_nanos() as u64,
        atomic::Ordering::SeqCst,
    );
    db.with_state_mut(|state| {
        state.db_config.default_stream_time_step = default_stream_time_step;
    });
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

        let mut component_changed = false;

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

            let pair_id = ComponentId::from_pair(&entity_metadata.name, &component_metadata.name);
            let Some(component) = state.get_component(pair_id) else {
                continue;
            };
            let Some((_, head)) = component.time_series.latest() else {
                continue;
            };

            let current_value = &column.buffer[offset..offset + size];
            if current_value != head {
                component_changed = true;
            }

            column.buffer[offset..offset + size].copy_from_slice(head);
        }

        if component_changed {
            world.dirty_components.insert(*component_id);
        }
    }
}

pub type PairId = ComponentId;

pub fn get_pair_ids(
    world: &WorldExec<Compiled>,
    components: &[ComponentId],
) -> Result<Vec<PairId>, Error> {
    let mut results = vec![];
    for component_id in components {
        if let Some((_schema, component_metadata)) =
            world.world.metadata.component_map.get(component_id)
        {
            let Some(column) = world.world.host.get(component_id) else {
                continue;
            };
            let entity_ids =
                bytemuck::try_cast_slice::<_, u64>(column.entity_ids.as_slice()).unwrap();
            for entity_id in entity_ids.iter() {
                let entity_id = impeller2::types::EntityId(*entity_id);
                let Some(entity_metadata) = world.world.metadata.entity_metadata.get(&entity_id)
                else {
                    continue;
                };
                let pair_id =
                    ComponentId::from_pair(&entity_metadata.name, &component_metadata.name);
                results.push(pair_id);
            }
        }
    }
    Ok(results)
}

pub fn commit_world_head(
    state: &State,
    world: &mut WorldExec<Compiled>,
    timestamp: Timestamp,
    exclusions: Option<&HashSet<ComponentId>>,
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
            let Some((_schema, component_metadata)) =
                world.world.metadata.component_map.get(component_id)
            else {
                continue;
            };

            if let Some(exclusions) = exclusions
                && exclusions.contains(component_id)
            {
                continue;
            }

            let pair_id = ComponentId::from_pair(&entity_metadata.name, &component_metadata.name);
            let Some(component) = state.get_component(pair_id) else {
                continue;
            };
            let buf = &column.buffer[offset..offset + size];
            component.time_series.push_buf(timestamp, buf)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn tick(
    db: Arc<DB>,
    tick_counter: Arc<AtomicU64>,
    mut world: WorldExec<Compiled>,
    is_cancelled: impl Fn() -> bool + 'static,
    pre_step: impl Fn(u64, &Arc<DB>, &Arc<AtomicU64>, Timestamp, Timestamp) + 'static,
    post_step: impl Fn(u64, &Arc<DB>, &Arc<AtomicU64>, Timestamp, Timestamp) + 'static,
    start_timestamp: Timestamp,
    interactive: bool,
    cancel_token: CancelToken,
) {
    let external_controls: HashSet<ComponentId> = external_controls(&world).collect();
    let wait_for_write: Vec<ComponentId> = wait_for_write(&world).collect();
    let wait_for_write_pair_ids: Vec<PairId> = get_pair_ids(&world, &wait_for_write).unwrap();
    let mut wait_for_write_pair_ids = collect_timestamps(&db, &wait_for_write_pair_ids);
    let run_time_step: Option<Duration> = world
        .world
        .metadata
        .run_time_step
        .map(|time_step| time_step.0);
    let time_step = world.world.sim_time_step().0;
    #[rustfmt::skip]
    let should_cancel = || { is_cancelled() || cancel_token.is_cancelled() };
    let mut next_tick_deadline: Option<Instant> = None;
    loop {
        if should_cancel() {
            return;
        }
        if !db.recording_cell.is_playing() {
            next_tick_deadline = None;
            let _ = futures_lite::future::race(db.recording_cell.wait(), async {
                cancel_token.wait().await;
                false
            })
            .await;
            if should_cancel() {
                return;
            }
            continue;
        }
        let start = Instant::now();
        if let Some(rts) = run_time_step {
            next_tick_deadline.get_or_insert(start + rts);
        }
        let tick = tick_counter.load(Ordering::SeqCst);
        let tick_nanos = time_step.as_nanos() * (tick as u128);
        let tick_duration = Duration::from_nanos(tick_nanos.min(u64::MAX as u128) as u64);
        let timestamp = start_timestamp + tick_duration;
        if tick >= world.world.max_tick() {
            db.recording_cell.set_playing(false);
            world.world.metadata.max_tick = u64::MAX;
            if !interactive {
                return;
            } else {
                info!(
                    "Simulation stopped; it reached its max_tick {}.",
                    world.world.max_tick()
                );
            }
        }
        pre_step(tick, &db, &tick_counter, timestamp, start_timestamp);

        if tick_counter.load(Ordering::SeqCst) < tick {
            db.last_updated.store(Timestamp(i64::MIN));
            next_tick_deadline = None;
            continue;
        }

        db.with_state(|state| copy_db_to_world(state, &mut world));
        if let Err(err) = world.run() {
            warn!(?err, "error ticking world");
        }
        db.with_state(|state| {
            if let Err(err) =
                commit_world_head(state, &mut world, timestamp, Some(&external_controls))
            {
                warn!(?err, "error committing head");
            }
        });
        db.last_updated.update_max(timestamp);
        while !wait_for_write_pair_ids.is_empty()
            && !timestamps_changed(&db, &mut wait_for_write_pair_ids).unwrap_or(false)
        {
            stellarator::sleep(Duration::from_millis(1)).await;
            if should_cancel() {
                return;
            }
        }
        if should_cancel() {
            return;
        }
        post_step(tick, &db, &tick_counter, timestamp, start_timestamp);
        if let Some(deadline) = next_tick_deadline.as_mut() {
            let now = Instant::now();
            if *deadline > now {
                stellarator::sleep(*deadline - now).await;
            }
            *deadline += run_time_step.unwrap();
        }

        if tick_counter.load(Ordering::SeqCst) == tick {
            tick_counter.fetch_add(1, Ordering::SeqCst);
        }
    }
}

pub fn external_controls(world: &WorldExec<Compiled>) -> impl Iterator<Item = ComponentId> + '_ {
    world
        .world
        .metadata
        .component_map
        .iter()
        .filter(|(_, (_, component_metadata))| {
            component_metadata
                .metadata
                .get("external_control")
                .map(|s| s == "true")
                .unwrap_or(false)
        })
        .map(|(component_id, _)| *component_id)
}

pub fn wait_for_write(world: &WorldExec<Compiled>) -> impl Iterator<Item = ComponentId> + '_ {
    world
        .world
        .metadata
        .component_map
        .iter()
        .filter(|(_component_id, (_schema, component_metadata))| {
            component_metadata
                .metadata
                .get("wait_for_write")
                .map(|s| s == "true")
                .unwrap_or(false)
        })
        .map(|(component_id, _)| *component_id)
}

pub fn collect_timestamps(db: &DB, components: &[ComponentId]) -> Vec<(ComponentId, Timestamp)> {
    db.with_state(|state| {
        let mut timestamps = vec![];
        for component_id in components.iter() {
            if let Some(component) = state.get_component(*component_id) {
                if let Some((timestamp, _)) = component.time_series.latest() {
                    timestamps.push((*component_id, *timestamp));
                } else {
                    timestamps.push((*component_id, Timestamp(i64::MIN)));
                }
            } else {
                timestamps.push((*component_id, Timestamp(i64::MIN)));
            }
        }
        timestamps
    })
}

pub fn timestamps_changed(db: &DB, components: &mut [(PairId, Timestamp)]) -> Option<bool> {
    if components.is_empty() {
        return None;
    }
    db.with_state(|state| {
        let mut changed = None;
        for (component_id, timestamp) in components.iter_mut() {
            if let Some(component) = state.get_component(*component_id)
                && let Some((curr_timestamp, _)) = component.time_series.latest()
            {
                if *timestamp != *curr_timestamp {
                    changed = Some(true);
                    *timestamp = *curr_timestamp;
                } else if changed.is_none() {
                    changed = Some(false);
                }
            }
        }
        changed
    })
}
