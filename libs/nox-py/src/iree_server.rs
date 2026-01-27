//! IREE-compatible server for running simulations with the Elodin DB.
//!
//! This module provides a simplified server that works with IREEWorldExec
//! instead of the XLA-based WorldExec.

use crate::iree_exec::IREEWorldExec;
use crate::Error;
use elodin_db::DB;
use impeller2::types::{ComponentId, Timestamp};
use impeller2_wkt::{ComponentMetadata, EntityMetadata};
use nox_ecs::World;
use pyo3::Python;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use stellarator::struc_con::{Joinable, Thread};
use tracing::warn;

/// IREE-compatible server for running simulations.
pub struct IREEServer {
    db: elodin_db::Server,
    world: IREEWorldExec,
}

impl IREEServer {
    pub fn new(db: elodin_db::Server, world: IREEWorldExec) -> Self {
        Self { db, world }
    }

    /// Simple run method without cancellation support.
    /// Currently unused but kept for API completeness and future use.
    #[allow(dead_code)]
    pub async fn run(self) -> Result<(), Error> {
        tracing::info!("running IREE server");
        self.run_with_cancellation(
            || false,
            |_, _, _, _, _| {},
            |_, _, _, _, _| {},
            false,
            None,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn run_with_cancellation(
        self,
        is_cancelled: impl Fn() -> bool + 'static,
        pre_step: impl Fn(u64, &Arc<DB>, &Arc<AtomicU64>, Timestamp, Timestamp) + 'static,
        post_step: impl Fn(u64, &Arc<DB>, &Arc<AtomicU64>, Timestamp, Timestamp) + 'static,
        interactive: bool,
        start_timestamp: Option<Timestamp>,
    ) -> Result<(), Error> {
        tracing::info!("running IREE server with cancellation");
        let Self { db, mut world } = self;
        let elodin_db::Server { listener, db } = db;
        let start_time = start_timestamp.unwrap_or_else(Timestamp::now);
        init_db_iree(&db, &mut world.world, start_time)?;
        let tick_db = db.clone();
        let tick_counter = Arc::new(AtomicU64::new(0));
        let stream: Thread<Option<Result<(), Error>>> =
            stellarator::struc_con::stellar(move || async move {
                let mut handles = vec![];
                loop {
                    let stream = listener.accept().await?;
                    handles.push(
                        stellarator::spawn(elodin_db::handle_conn(stream, db.clone())).drop_guard(),
                    );
                }
            });
        let tick = stellarator::spawn(tick_iree(
            tick_db,
            tick_counter,
            world,
            is_cancelled,
            pre_step,
            post_step,
            start_time,
            interactive,
        ));
        futures_lite::future::race(async { stream.join().await.unwrap().unwrap() }, async {
            tick.await
                .map_err(|_| stellarator::Error::JoinFailed)
                .map_err(nox_ecs::Error::from)
                .map_err(Error::from)
        })
        .await
    }
}

/// Initialize the database with world state for IREE execution.
pub fn init_db_iree(
    db: &elodin_db::DB,
    world: &mut World,
    start_timestamp: Timestamp,
) -> Result<(), elodin_db::Error> {
    tracing::info!("initializing db for IREE");
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

    let default_stream_time_step = Duration::from_secs_f64(
        world.metadata.sim_time_step.0.as_secs_f64() / world.metadata.default_playback_speed,
    );
    db.default_stream_time_step.store(
        default_stream_time_step.as_nanos() as u64,
        std::sync::atomic::Ordering::SeqCst,
    );
    let _ = db.save_db_state();

    Ok(())
}

/// Commit world state to the database for IREE execution.
pub fn commit_world_head_iree(
    state: &elodin_db::State,
    world: &mut IREEWorldExec,
    timestamp: Timestamp,
    exclusions: Option<&HashSet<ComponentId>>,
) -> Result<(), nox_ecs::Error> {
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

/// Copy database state to world for IREE execution.
pub fn copy_db_to_world_iree(state: &elodin_db::State, world: &mut IREEWorldExec) {
    let world_inner = &mut world.world;
    for (component_id, (schema, _)) in world_inner.metadata.component_map.iter() {
        let Some(column) = world_inner.host.get_mut(component_id) else {
            continue;
        };
        let entity_ids = bytemuck::try_cast_slice::<_, u64>(column.entity_ids.as_slice()).unwrap();
        let size = schema.size();

        let mut component_changed = false;

        for (i, entity_id) in entity_ids.iter().enumerate() {
            let offset = i * size;
            let entity_id = impeller2::types::EntityId(*entity_id);
            let Some(entity_metadata) = world_inner.metadata.entity_metadata.get(&entity_id) else {
                continue;
            };
            let Some((_, component_metadata)) = world_inner.metadata.component_map.get(component_id)
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
            world_inner.dirty_components.insert(*component_id);
        }
    }
}

/// Collect external control component IDs for IREE execution.
pub fn external_controls_iree(world: &IREEWorldExec) -> impl Iterator<Item = ComponentId> + '_ {
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

#[allow(clippy::too_many_arguments)]
async fn tick_iree(
    db: Arc<DB>,
    tick_counter: Arc<AtomicU64>,
    mut world: IREEWorldExec,
    is_cancelled: impl Fn() -> bool + 'static,
    pre_step: impl Fn(u64, &Arc<DB>, &Arc<AtomicU64>, Timestamp, Timestamp) + 'static,
    post_step: impl Fn(u64, &Arc<DB>, &Arc<AtomicU64>, Timestamp, Timestamp) + 'static,
    start_timestamp: Timestamp,
    interactive: bool,
) {
    let external_controls: HashSet<ComponentId> = external_controls_iree(&world).collect();
    let run_time_step: Option<Duration> = world
        .world
        .metadata
        .run_time_step
        .map(|time_step| time_step.0);
    let time_step = world.world.sim_time_step().0;
    
    while db.recording_cell.wait().await {
        let start = Instant::now();
        let tick = tick_counter.load(Ordering::SeqCst);
        let tick_nanos = time_step.as_nanos() * (tick as u128);
        let tick_duration = Duration::from_nanos(tick_nanos.min(u64::MAX as u128) as u64);
        let timestamp = start_timestamp + tick_duration;
        
        if tick >= world.world.max_tick() {
            db.recording_cell.set_playing(false);
            world.world.metadata.max_tick = u64::MAX;
            if !interactive {
                return;
            }
        }
        
        pre_step(tick, &db, &tick_counter, timestamp, start_timestamp);

        if tick_counter.load(Ordering::SeqCst) < tick {
            continue;
        }

        db.with_state(|state| copy_db_to_world_iree(state, &mut world));
        
        // Run the IREE computation
        if let Err(err) = Python::with_gil(|py| world.run(py)) {
            warn!(?err, "error ticking IREE world");
        }
        
        db.with_state(|state| {
            if let Err(err) =
                commit_world_head_iree(state, &mut world, timestamp, Some(&external_controls))
            {
                warn!(?err, "error committing head");
            }
        });
        
        db.last_updated.store(timestamp);
        
        if is_cancelled() {
            return;
        }
        
        post_step(tick, &db, &tick_counter, timestamp, start_timestamp);
        
        if let Some(run_time_step) = run_time_step.as_ref()
            && let Some(sleep_time) = run_time_step.checked_sub(start.elapsed())
        {
            stellarator::sleep(sleep_time).await;
        }

        if tick_counter.load(Ordering::SeqCst) == tick {
            tick_counter.fetch_add(1, Ordering::SeqCst);
        }
    }
}
