use impeller2::types::PrimType;
use impeller_db::{handle_conn, Error, DB};
use smallvec::SmallVec;
use std::{
    sync::{atomic, Arc},
    time::{Duration, Instant},
};
use tracing::warn;

use crate::{Compiled, WorldExec};

pub struct Server {
    db: impeller_db::Server,
    world: WorldExec<Compiled>,
}

impl Server {
    pub fn new(db: impeller_db::Server, world: WorldExec<Compiled>) -> Self {
        Self { db, world }
    }

    pub async fn run(self) -> Result<(), Error> {
        let Self { db, mut world } = self;
        let impeller_db::Server {
            listener,
            db,
            time_step,
        } = db;
        init_db(&db, &mut world)?;
        stellarator::spawn(tick(time_step, db.clone(), world));
        loop {
            let stream = listener.accept().await?;
            stellarator::spawn(handle_conn(stream, db.clone()));
        }
    }
}

fn init_db(
    db: &impeller_db::DB,
    world: &mut WorldExec<Compiled>,
) -> Result<(), impeller_db::Error> {
    for (component_id, (archetype, metadata)) in world.world.component_map.iter() {
        let shape: SmallVec<[usize; 4]> = metadata
            .component_type
            .shape
            .iter()
            .map(|&x| x as usize)
            .collect();
        let prim_type = match metadata.component_type.primitive_ty {
            impeller::PrimitiveTy::U8 => PrimType::U8,
            impeller::PrimitiveTy::U16 => PrimType::U16,
            impeller::PrimitiveTy::U32 => PrimType::U32,
            impeller::PrimitiveTy::U64 => PrimType::U64,
            impeller::PrimitiveTy::I8 => PrimType::I8,
            impeller::PrimitiveTy::I16 => PrimType::I16,
            impeller::PrimitiveTy::I32 => PrimType::I32,
            impeller::PrimitiveTy::I64 => PrimType::I64,
            impeller::PrimitiveTy::Bool => PrimType::Bool,
            impeller::PrimitiveTy::F32 => PrimType::F32,
            impeller::PrimitiveTy::F64 => PrimType::F64,
        };
        let Some(world_buf) = world.world.host.get_mut(component_id) else {
            continue;
        };
        let component_id = impeller2::types::ComponentId(component_id.0);
        let db_component = db.components.entry(component_id).or_try_insert_with(|| {
            impeller_db::Component::try_create(component_id, prim_type, &shape, &db.path)
        })?;
        let Some(entity_ids) = world.world.entity_ids.get(archetype) else {
            continue;
        };

        let size = metadata.component_type.size();
        let entity_ids = bytemuck::try_cast_slice::<_, u64>(entity_ids.as_slice()).unwrap();
        for (i, entity_id) in entity_ids.iter().enumerate() {
            let offset = i * size;
            let entity_id = impeller2::types::EntityId(*entity_id);
            let path = db
                .path
                .join(component_id.to_string())
                .join(entity_id.to_string());
            let start_tick = db.latest_tick.load(atomic::Ordering::SeqCst);
            let entity = impeller_db::Entity::create(
                path,
                start_tick,
                entity_id,
                db_component.schema.clone(),
            )?;
            {
                let mut writer = entity.writer.lock().unwrap();
                writer
                    .head_mut(size)?
                    .copy_from_slice(&world_buf[offset..offset + size]);
            }
            db_component.entities.insert(entity_id, entity);
        }
    }
    Ok(())
}

fn copy_db_to_world(db: &DB, world: &mut WorldExec<Compiled>) {
    for (component_id, (archetype, metadata)) in world.world.component_map.iter() {
        let Some(component) = db
            .components
            .get(&impeller2::types::ComponentId(component_id.0))
        else {
            continue;
        };
        let Some(entity_ids) = world.world.entity_ids.get(archetype) else {
            continue;
        };

        let Some(world_buf) = world.world.host.get_mut(component_id) else {
            continue;
        };
        let entity_ids = bytemuck::try_cast_slice::<_, u64>(entity_ids.as_slice()).unwrap();
        let size = metadata.component_type.size();
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
            world_buf[offset..offset + size].copy_from_slice(head);
        }
    }
}

fn commit_world_head(db: &DB, world: &mut WorldExec<Compiled>) {
    for (component_id, (archetype, metadata)) in world.world.component_map.iter() {
        let Some(component) = db
            .components
            .get(&impeller2::types::ComponentId(component_id.0))
        else {
            continue;
        };
        let Some(entity_ids) = world.world.entity_ids.get(archetype) else {
            continue;
        };

        let Some(world_buf) = world.world.host.get_mut(component_id) else {
            continue;
        };
        let entity_ids = bytemuck::try_cast_slice::<_, u64>(entity_ids.as_slice()).unwrap();
        let size = metadata.component_type.size();
        for (i, entity_id) in entity_ids.iter().enumerate() {
            let offset = i * size;
            let Some(db_buf) = component
                .entities
                .get(&impeller2::types::EntityId(*entity_id))
            else {
                continue;
            };
            let mut writer = db_buf.writer.lock().unwrap();
            if let Err(err) = writer.commit_head(&world_buf[offset..offset + size]) {
                warn!(?err, "error committing head");
            }
        }
    }
}

async fn tick(time_step: Duration, db: Arc<DB>, mut world: WorldExec<Compiled>) {
    let mut start = Instant::now();
    loop {
        copy_db_to_world(&db, &mut world);
        if let Err(err) = world.run() {
            warn!(?err, "error ticking world");
        }
        commit_world_head(&db, &mut world);
        db.latest_tick.fetch_add(1, atomic::Ordering::Release);
        db.tick_waker.wake_all();
        let sleep_time = time_step.saturating_sub(start.elapsed());
        stellarator::sleep(sleep_time).await;
        start += time_step;
    }
}
