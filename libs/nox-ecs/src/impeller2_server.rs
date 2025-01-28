use elodin_db::MetadataExt;
use elodin_db::{handle_conn, DB};
use nox_ecs::Error;
use smallvec::SmallVec;
use std::{
    io,
    sync::{atomic, Arc},
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
        self.run_with_cancellation(|| false).await
    }

    pub async fn run_with_cancellation(
        self,
        is_cancelled: impl Fn() -> bool + 'static,
    ) -> Result<(), Error> {
        let Self { db, mut world } = self;
        let elodin_db::Server { listener, db } = db;
        init_db(&db, &mut world.world)?;
        let tick_db = db.clone();
        let stream: Thread<Option<Result<(), Error>>> =
            stellarator::struc_con::stellar(move || async move {
                let mut handles = vec![];
                loop {
                    let stream = listener.accept().await?;
                    handles.push(stellarator::spawn(handle_conn(stream, db.clone())).drop_guard());
                }
            });
        let tick = stellarator::spawn(tick(tick_db, world, is_cancelled));
        futures_lite::future::race(async { stream.join().await.unwrap().unwrap() }, async {
            tick.await
                .map_err(|_| stellarator::Error::JoinFailed)
                .map_err(Error::from)
        })
        .await
    }
}

pub fn init_db(db: &elodin_db::DB, world: &mut World) -> Result<(), elodin_db::Error> {
    for (id, asset) in world.assets.iter().enumerate() {
        db.assets.insert(id as u64, &asset.inner)?;
    }
    for (component_id, (schema, _)) in world.metadata.component_map.iter() {
        let shape: SmallVec<[usize; 4]> = schema.shape.iter().map(|&x| x as usize).collect();
        let Some(column) = world.host.get_mut(component_id) else {
            continue;
        };
        let component_id = impeller2::types::ComponentId(component_id.0);
        let db_component = db.components.entry(component_id).or_try_insert_with(|| {
            elodin_db::Component::try_create(component_id, schema.prim_type, &shape, &db.path)
        })?;
        let size = schema.size();
        let entity_ids = bytemuck::try_cast_slice::<_, u64>(column.entity_ids.as_slice()).unwrap();
        for (i, entity_id) in entity_ids.iter().enumerate() {
            let offset = i * size;
            let entity_id = impeller2::types::EntityId(*entity_id);
            let path = db
                .path
                .join(component_id.to_string())
                .join(entity_id.to_string());
            let start_tick = db.latest_tick.load(atomic::Ordering::SeqCst);
            let entity = match elodin_db::Entity::create(
                path,
                start_tick,
                entity_id,
                db_component.schema.clone(),
            ) {
                Ok(entity) => entity,
                Err(elodin_db::Error::Io(err)) if err.kind() == io::ErrorKind::AlreadyExists => {
                    continue;
                }
                Err(err) => return Err(err),
            };
            {
                let mut writer = entity.writer.lock().unwrap();
                writer
                    .head_mut(size)?
                    .copy_from_slice(&column.buffer[offset..offset + size]);
            }
            db_component.entities.insert(entity_id, entity);
        }
    }
    let entity_metadata_path = db.path.join("entity_metadata");
    std::fs::create_dir_all(&entity_metadata_path)?;
    for (id, metadata) in world.entity_metadata().iter() {
        let path = entity_metadata_path.join(id.to_string());
        db.entity_metadata.insert(*id, metadata.clone());
        if let Err(err) = metadata.write(&path) {
            warn!(?err, "failed to write metadata");
        }
    }
    for (component_id, (_, metadata)) in world.component_map().iter() {
        if let Some(component) = db.components.get_mut(component_id) {
            if let Err(err) =
                metadata.write(db.path.join(component_id.to_string()).join("metadata"))
            {
                warn!(?err, "failed to write metadata");
            }
            component.metadata.store(Arc::new(metadata.clone()));
        }
    }
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

pub fn copy_db_to_world(db: &DB, world: &mut WorldExec<Compiled>) {
    for (component_id, (schema, _)) in world.world.metadata.component_map.iter() {
        let Some(component) = db
            .components
            .get(&impeller2::types::ComponentId(component_id.0))
        else {
            continue;
        };

        let Some(column) = world.world.host.get_mut(component_id) else {
            continue;
        };
        let entity_ids = bytemuck::try_cast_slice::<_, u64>(column.entity_ids.as_slice()).unwrap();
        let size = schema.size();
        for (i, entity_id) in entity_ids.iter().enumerate() {
            let offset = i * size;
            let Some(db_buf) = component
                .entities
                .get(&impeller2::types::EntityId(*entity_id))
            else {
                continue;
            };
            let writer = db_buf.writer.lock().unwrap();
            let head = writer.head().unwrap();
            column.buffer[offset..offset + size].copy_from_slice(head);
        }
    }
}

pub fn commit_world_head(db: &DB, world: &mut WorldExec<Compiled>) {
    for (component_id, (schema, _)) in world.world.metadata.component_map.iter() {
        let Some(component) = db
            .components
            .get(&impeller2::types::ComponentId(component_id.0))
        else {
            continue;
        };

        let Some(column) = world.world.host.get_mut(component_id) else {
            continue;
        };
        let entity_ids = bytemuck::try_cast_slice::<_, u64>(column.entity_ids.as_slice()).unwrap();
        let size = schema.size();
        for (i, entity_id) in entity_ids.iter().enumerate() {
            let offset = i * size;
            let Some(db_buf) = component
                .entities
                .get(&impeller2::types::EntityId(*entity_id))
            else {
                continue;
            };
            let buf = &column.buffer[offset..offset + size];
            let mut writer = db_buf.writer.lock().unwrap();
            if let Err(err) = writer.commit_head(buf) {
                warn!(?err, "error committing head");
            }
        }
    }
}

async fn tick(
    db: Arc<DB>,
    mut world: WorldExec<Compiled>,
    is_cancelled: impl Fn() -> bool + 'static,
) {
    let mut start = Instant::now();
    let mut tick = 0;
    while db.recording_cell.wait().await {
        if tick >= world.world.max_tick() {
            db.recording_cell.set_playing(false);
            world.world.metadata.max_tick = u64::MAX;
        }
        copy_db_to_world(&db, &mut world);
        if let Err(err) = world.run() {
            warn!(?err, "error ticking world");
        }
        commit_world_head(&db, &mut world);
        db.latest_tick.fetch_add(1, atomic::Ordering::Release);
        db.tick_waker.wake_all();
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
    }
}
