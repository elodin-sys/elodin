use elodin_db::{DB, State, handle_conn};
use impeller2::types::{ComponentId, Timestamp};
use impeller2_wkt::{ComponentMetadata, EntityMetadata, SetComponentMetadata, UdpUnicast, Stream, StreamBehavior};
use impeller2_stellar::Client;
use nox_ecs::Error;
use std::{
    net::SocketAddr,
    sync::{Arc, atomic},
    time::{Duration, Instant},
};
use stellarator::struc_con::{Joinable, Thread};
use tracing::{info, warn};

use crate::{Compiled, World, WorldExec};

pub struct Server {
    db: elodin_db::Server,
    world: WorldExec<Compiled>,
    mirror_addr: Option<SocketAddr>,
}

impl Server {
    /// Create a new server with an embedded database
    pub fn new(db: elodin_db::Server, world: WorldExec<Compiled>) -> Self {
        Self {
            db,
            world,
            mirror_addr: None,
        }
    }
    
    /// Create a new server with an embedded database that mirrors to an external database
    pub fn with_mirror(db: elodin_db::Server, world: WorldExec<Compiled>, mirror_addr: SocketAddr) -> Self {
        Self {
            db,
            world,
            mirror_addr: Some(mirror_addr),
        }
    }

    pub async fn run(self) -> Result<(), Error> {
        tracing::info!("running server");
        self.run_with_cancellation(|| false).await
    }

    pub async fn run_with_cancellation(
        self,
        is_cancelled: impl Fn() -> bool + 'static,
    ) -> Result<(), Error> {
        let Self { db, mut world, mirror_addr } = self;
        let elodin_db::Server { listener, db } = db;
        let start_time = Timestamp::now();
        
        // Initialize embedded database as normal
        init_db(&db, &mut world.world, start_time)?;
        
        // If mirroring is enabled, set it up
        if let Some(mirror_addr) = mirror_addr {
            info!("setting up mirror to external database at {}", mirror_addr);
            let listener_addr = listener.local_addr()?;
            let db_clone = db.clone();
            stellarator::spawn(async move {
                if let Err(e) = setup_mirror(listener_addr, mirror_addr, db_clone).await {
                    warn!("failed to setup mirror: {:?}", e);
                }
            });
        }
        
        // Run the server normally
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

        // Track if any values changed for this component
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

            let pair_name = format!("{}.{}", entity_metadata.name, component_metadata.name);
            let pair_id = ComponentId::new(&pair_name);

            let Some(component) = state.get_component(pair_id) else {
                continue;
            };
            let (_, head) = component.time_series.latest().unwrap();

            // Check if the value has changed
            let current_value = &column.buffer[offset..offset + size];
            if current_value != head {
                component_changed = true;
            }

            column.buffer[offset..offset + size].copy_from_slice(head);
        }

        // Mark component as dirty if any value changed
        // This ensures it gets copied to client buffers for GPU execution
        if component_changed {
            world.dirty_components.insert(*component_id);
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

            // Skip writing back external control components
            if component_metadata
                .metadata
                .get("external_control")
                .map(|s| s == "true")
                .unwrap_or(false)
            {
                continue;
            }

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
        let time_step = world.world.metadata.run_time_step.0;
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

/// Set up mirroring from the embedded database to an external database
/// 
/// This mirrors the approach used in downlink.lua:
/// 1. Connect to both databases
/// 2. Get metadata from the embedded database and send to external database
/// 3. Tell the embedded database to stream to the external database
async fn setup_mirror(
    source_addr: SocketAddr,
    mirror_addr: SocketAddr,
    _db: Arc<DB>,
) -> Result<(), Error> {
    info!("configuring mirror: {} -> {}", source_addr, mirror_addr);
    
    // Wait a moment for the embedded database to be fully ready
    stellarator::sleep(Duration::from_millis(100)).await;
    
    // Connect to both databases
    let mut source_client = Client::connect(source_addr)
        .await
        .map_err(|e| Error::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
    
    let mut mirror_client = Client::connect(mirror_addr)
        .await
        .map_err(|e| Error::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
    
    // Get metadata from source database using DumpMetadata message
    use impeller2_wkt::DumpMetadata;
    let metadata_resp = source_client.request(&DumpMetadata).await
        .map_err(|e| Error::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e))))?;
    
    info!("dumped {} components from source database", metadata_resp.component_metadata.len());
    
    // Send component metadata to mirror database
    for component_metadata in metadata_resp.component_metadata {
        let msg = SetComponentMetadata(component_metadata);
        let (result, _) = mirror_client.send(&msg).await;
        result.map_err(|e| Error::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e))))?;
    }
    
    info!("metadata sent to mirror database");
    
    // Tell the embedded database to stream to the mirror via UDP (forward direction)
    let forward_stream = UdpUnicast {
        stream: Stream {
            behavior: StreamBehavior::RealTime,
            id: 1,
        },
        addr: mirror_addr.to_string(),
    };
    
    let (result, _) = source_client.send(&forward_stream).await;
    result.map_err(|e| Error::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e))))?;
    
    info!("forward streaming configured: embedded → external");
    
    // Tell the external database to stream BACK to the embedded database (reverse direction)
    // This enables bidirectional control - values written to external DB flow back to simulation
    let reverse_stream = UdpUnicast {
        stream: Stream {
            behavior: StreamBehavior::RealTime,
            id: 2,
        },
        addr: source_addr.to_string(),
    };
    
    let (result, _) = mirror_client.send(&reverse_stream).await;
    result.map_err(|e| Error::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e))))?;
    
    info!("reverse streaming configured: external → embedded");
    info!("bidirectional mirroring active: {} ↔ {}", source_addr, mirror_addr);
    Ok(())
}
