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

use crate::World;
use crate::exec::WorldExec;

pub struct Server {
    db: elodin_db::Server,
    world: WorldExec,
}

impl Server {
    pub fn new(db: elodin_db::Server, world: WorldExec) -> Self {
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
        init_db(&db, world.world_mut(), start_time)?;
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
        }
        if let Some(path) = &world.metadata.schematic_path {
            state
                .db_config
                .set_schematic_path(path.to_string_lossy().to_string());
        }
        if let Some(content) = &world.metadata.schematic {
            state.db_config.set_schematic_content(content.clone());
        }
        if !world.metadata.sensor_cameras.is_empty() {
            match serde_json::to_string(&world.metadata.sensor_cameras) {
                Ok(json) => {
                    state
                        .db_config
                        .metadata
                        .insert("sensor_cameras".to_string(), json);
                }
                Err(e) => {
                    tracing::warn!("Failed to serialize sensor_cameras metadata: {e}");
                }
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

pub fn copy_db_to_world(state: &State, world: &mut WorldExec) {
    let world = world.world_mut();
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

pub fn get_pair_ids(world: &WorldExec, components: &[ComponentId]) -> Result<Vec<PairId>, Error> {
    let w = world.world();
    let mut results = vec![];
    for component_id in components {
        if let Some((_schema, component_metadata)) = w.metadata.component_map.get(component_id) {
            let Some(column) = w.host.get(component_id) else {
                continue;
            };
            let entity_ids =
                bytemuck::try_cast_slice::<_, u64>(column.entity_ids.as_slice()).unwrap();
            for entity_id in entity_ids.iter() {
                let entity_id = impeller2::types::EntityId(*entity_id);
                let Some(entity_metadata) = w.metadata.entity_metadata.get(&entity_id) else {
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

pub fn commit_world_head_unified(
    state: &State,
    exec: &mut crate::exec::WorldExec,
    timestamp: Timestamp,
    exclusions: Option<&HashSet<ComponentId>>,
) -> Result<(), Error> {
    commit_world_head_for_world(state, exec.world_mut(), timestamp, exclusions)
}

fn commit_world_head_for_world(
    state: &State,
    world: &mut crate::world::World,
    timestamp: Timestamp,
    exclusions: Option<&HashSet<ComponentId>>,
) -> Result<(), Error> {
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
            let Some((_schema, component_metadata)) =
                world.metadata.component_map.get(component_id)
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
    mut world: WorldExec,
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
    let generate_real_time = world.world().metadata.generate_real_time;
    let configured_ticks_per_telemetry = world.world().ticks_per_telemetry();
    let time_step = world.world().sim_time_step().0;
    let detailed_timing = std::env::var("ELODIN_DETAILED_TIMING").is_ok();
    if detailed_timing {
        world.profiler_mut().detailed_timing = true;
        info!("detailed timing enabled (ELODIN_DETAILED_TIMING)");
    }
    #[rustfmt::skip]
    let should_cancel = || { is_cancelled() || cancel_token.is_cancelled() };
    let mut next_tick_deadline: Option<Instant> = None;
    let mut last_behind_warning: Option<Instant> = None;
    let mut last_timing_log: Option<Instant> = None;
    // Always-on per-phase timing summary. Dropping `metrics` prints
    // the one-block stdout summary on any exit (cancel, max_tick,
    // panic unwind). See `crate::tick_metrics` for the format.
    let mut metrics = crate::tick_metrics::TickMetrics::new();
    // Feed the summary header its rate context: simulation_rate is
    // 1 / sim_time_step; telemetry_rate is simulation_rate divided
    // by ticks_per_telemetry. The summary uses these to annotate
    // the steps line when rates differ.
    {
        let simulation_rate_hz = 1.0 / time_step.as_secs_f64();
        let telemetry_rate_hz = simulation_rate_hz / (configured_ticks_per_telemetry.max(1) as f64);
        metrics.set_rates(simulation_rate_hz, telemetry_rate_hz);
    }
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
        if generate_real_time {
            // Deadline is set after effective_batch is known for this iteration.
        }
        let tick = tick_counter.load(Ordering::SeqCst);
        let tick_nanos = time_step.as_nanos() * (tick as u128);
        let tick_duration = Duration::from_nanos(tick_nanos.min(u64::MAX as u128) as u64);
        let timestamp = start_timestamp + tick_duration;
        let max_tick = world.world().max_tick();
        let effective_batch = configured_ticks_per_telemetry.min(max_tick.saturating_sub(tick));
        let effective_batch = effective_batch.max(1);
        let effective_batch_time_step =
            Duration::from_secs_f64(time_step.as_secs_f64() * effective_batch as f64);
        if generate_real_time {
            next_tick_deadline.get_or_insert(start + effective_batch_time_step);
        }
        let end_tick = tick.saturating_add(effective_batch.saturating_sub(1));
        let end_tick_nanos = time_step.as_nanos() * (end_tick as u128);
        let end_tick_duration = Duration::from_nanos(end_tick_nanos.min(u64::MAX as u128) as u64);
        let batch_end_timestamp = start_timestamp + end_tick_duration;
        if tick >= max_tick {
            db.recording_cell.set_playing(false);
            world.world_mut().metadata.max_tick = u64::MAX;
            if !interactive {
                return;
            } else {
                info!(
                    "Simulation stopped; it reached its max_tick {}.",
                    world.world().max_tick()
                );
            }
        }
        let pre_step_start = Instant::now();
        pre_step(tick, &db, &tick_counter, timestamp, start_timestamp);
        metrics.observe_pre_step(pre_step_start.elapsed());

        if tick_counter.load(Ordering::SeqCst) < tick {
            db.last_updated.store(Timestamp(i64::MIN));
            next_tick_deadline = None;
            continue;
        }

        let copy_in_start = Instant::now();
        db.with_state(|state| copy_db_to_world(state, &mut world));
        let copy_in_elapsed = copy_in_start.elapsed();
        metrics.observe_copy_db_to_world(copy_in_elapsed);
        let copy_in_ms = if detailed_timing {
            copy_in_elapsed.as_secs_f64() * 1000.0
        } else {
            0.0
        };
        // Temporarily override so the kernel runs the right number of batched ticks.
        world.world_mut().metadata.ticks_per_telemetry = effective_batch;
        let run_start = Instant::now();
        if let Err(err) = world.run() {
            warn!(?err, "error ticking world");
        }
        let run_elapsed = run_start.elapsed();
        metrics.observe_world_run(run_elapsed);
        let run_ms = if detailed_timing {
            run_elapsed.as_secs_f64() * 1000.0
        } else {
            0.0
        };
        world.world_mut().metadata.ticks_per_telemetry = configured_ticks_per_telemetry;
        let commit_start = Instant::now();
        db.with_state(|state| {
            if let Err(err) = commit_world_head_unified(
                state,
                &mut world,
                batch_end_timestamp,
                Some(&external_controls),
            ) {
                warn!(?err, "error committing head");
            }
        });
        let commit_elapsed = commit_start.elapsed();
        metrics.observe_commit(commit_elapsed);
        let commit_ms = if detailed_timing {
            commit_elapsed.as_secs_f64() * 1000.0
        } else {
            0.0
        };
        db.last_updated.update_max(batch_end_timestamp);
        if detailed_timing {
            let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
            tracing::trace!(wall_ms, copy_in_ms, run_ms, commit_ms, "server tick phases",);
        }
        let wait_for_write_start = Instant::now();
        while !wait_for_write_pair_ids.is_empty()
            && !timestamps_changed(&db, &mut wait_for_write_pair_ids).unwrap_or(false)
        {
            stellarator::sleep(Duration::from_millis(1)).await;
            if should_cancel() {
                metrics.observe_wait_for_write(wait_for_write_start.elapsed());
                return;
            }
        }
        metrics.observe_wait_for_write(wait_for_write_start.elapsed());
        if should_cancel() {
            return;
        }
        // Called with end_tick (not batch start) because the world state now
        // reflects the last tick of the batch.
        let post_step_start = Instant::now();
        post_step(
            end_tick,
            &db,
            &tick_counter,
            batch_end_timestamp,
            start_timestamp,
        );
        metrics.observe_post_step(post_step_start.elapsed());
        if generate_real_time && let Some(deadline) = next_tick_deadline.as_mut() {
            let pacing_start = Instant::now();
            let now = Instant::now();
            if now > *deadline {
                let should_warn = last_behind_warning
                    .map(|last| now.duration_since(last) >= Duration::from_secs(1))
                    .unwrap_or(true);
                if should_warn {
                    let behind_ms = now.duration_since(*deadline).as_secs_f64() * 1000.0;
                    let target_ms = effective_batch_time_step.as_secs_f64() * 1000.0;
                    let factor = if target_ms > 0.0 {
                        (behind_ms + target_ms) / target_ms
                    } else {
                        0.0
                    };
                    info!(
                        "simulation cannot achieve real-time; {:.2}ms behind target {:.2}ms/tick ({:.2}x behind)",
                        behind_ms, target_ms, factor
                    );
                    last_behind_warning = Some(now);
                }
            }
            // Cap deadline drift: if we're more than 2 tick periods behind,
            // reset the deadline to now. This prevents a burst of fast ticks
            // (visible as timeline stutter in the editor) when a one-time
            // delay occurs, such as the initial render bridge connection.
            if now > *deadline + effective_batch_time_step * 2 {
                *deadline = now;
            }
            if *deadline > now {
                stellarator::sleep(*deadline - now).await;
            }
            *deadline += effective_batch_time_step;
            metrics.observe_real_time_pacing(pacing_start.elapsed());
        }
        if detailed_timing {
            let now = Instant::now();
            let should_log = last_timing_log
                .map(|last| now.duration_since(last) >= Duration::from_secs(2))
                .unwrap_or(true);
            if should_log {
                let profile = world.profile();
                let h2d = profile.get("h2d_upload").copied().unwrap_or(0.0);
                let kernel = profile.get("kernel_invoke").copied().unwrap_or(0.0);
                let d2h = profile.get("d2h_download").copied().unwrap_or(0.0);
                let history = profile.get("add_to_history").copied().unwrap_or(0.0);
                let tick_total = profile.get("tick").copied().unwrap_or(0.0);
                let rtf = profile.get("real_time_factor").copied().unwrap_or(0.0);
                info!(
                    "tick breakdown: total={:.3}ms (h2d={:.3}ms kernel={:.3}ms d2h={:.3}ms history={:.3}ms) RTF={:.2}x",
                    tick_total, h2d, kernel, d2h, history, rtf,
                );
                last_timing_log = Some(now);
            }
        }

        if tick_counter.load(Ordering::SeqCst) == tick {
            tick_counter.fetch_add(effective_batch, Ordering::SeqCst);
        }
        // One completed simulation cycle; `effective_batch` is the
        // number of sim steps `world.run()` executed inside it.
        metrics.cycles = metrics.cycles.saturating_add(1);
        metrics.steps = metrics.steps.saturating_add(effective_batch);
        // Record the full wall time of this loop iteration (pre_step
        // through real_time_pacing). Feeds the "mean cycle" value
        // in the summary header.
        metrics.observe_total_cycle(start.elapsed());
    }
}

pub fn external_controls(world: &WorldExec) -> impl Iterator<Item = ComponentId> + '_ {
    world
        .world()
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

pub fn wait_for_write(world: &WorldExec) -> impl Iterator<Item = ComponentId> + '_ {
    world
        .world()
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
