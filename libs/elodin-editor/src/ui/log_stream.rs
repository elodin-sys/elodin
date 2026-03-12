use bevy::ecs::query::QueryData;
use bevy::ecs::system::{In, InRef, SystemParam};
use bevy::prelude::{Commands, Component, Entity, Query, Res, World};
use egui::{self, Color32, RichText, ScrollArea};
use impeller2::types::{OwnedPacket, Timestamp};
use impeller2_bevy::{CommandsExt, CurrentStreamId, PacketGrantR};
use impeller2_wkt::{
    CurrentTimestamp, ErrorResponse, FixedRateMsgStream, FixedRateOp, GetMsgs, LogEntry, MsgBatch,
};
use std::collections::BTreeMap;
use std::time::Instant;

use super::PaneName;

const FRAMES_BEFORE_CONNECT: u32 = 5;
const MAX_LOG_ENTRIES: usize = 10_000;
const STREAM_RECOVERY_TIMEOUT_SECS: f32 = 5.0;

#[derive(Clone)]
pub struct LogStreamPane {
    pub entity: Entity,
    pub name: PaneName,
}

#[derive(Clone, Copy)]
pub struct LogStreamWidgetArgs {
    pub entity: Entity,
}

#[derive(Component, Default)]
pub struct LogStreamState {
    pub msg_id: [u8; 2],
    pub msg_name: String,
    pub connection_state: ConnectionState,
}

pub enum ConnectionState {
    WaitingToConnect { frames_waited: u32 },
    Active,
}

impl Default for ConnectionState {
    fn default() -> Self {
        Self::WaitingToConnect { frames_waited: 0 }
    }
}

#[derive(Component)]
pub struct LogCache {
    pub entries: BTreeMap<Timestamp, Vec<LogEntry>>,
    pub total_count: usize,
    pub last_stream_activity: Option<Instant>,
    auto_scroll: bool,
    filter_level: u8,
}

impl Default for LogCache {
    fn default() -> Self {
        Self {
            entries: BTreeMap::new(),
            total_count: 0,
            last_stream_activity: None,
            auto_scroll: true,
            filter_level: 0,
        }
    }
}

impl LogCache {
    fn insert(&mut self, ts: Timestamp, entry: LogEntry) {
        let bucket = self.entries.entry(ts).or_default();
        // Backfill and live stream can overlap in replay DBs; suppress exact duplicates.
        if bucket.iter().any(|existing| existing == &entry) {
            return;
        }
        bucket.push(entry);
        self.total_count += 1;

        while self.total_count > MAX_LOG_ENTRIES {
            let Some(oldest_ts) = self.entries.keys().next().copied() else {
                break;
            };

            let mut remove_bucket = false;
            if let Some(bucket) = self.entries.get_mut(&oldest_ts) {
                if !bucket.is_empty() {
                    bucket.remove(0);
                    self.total_count -= 1;
                }
                remove_bucket = bucket.is_empty();
            }

            if remove_bucket {
                self.entries.remove(&oldest_ts);
            }
        }
    }
}

trait LogEntryUiExt {
    fn level_str(&self) -> &'static str;
    fn level_color(&self) -> Color32;
}

impl LogEntryUiExt for LogEntry {
    fn level_str(&self) -> &'static str {
        match self.level {
            0 => "TRACE",
            1 => "DEBUG",
            2 => "INFO",
            3 => "WARN",
            4 => "ERROR",
            _ => "???",
        }
    }

    fn level_color(&self) -> Color32 {
        match self.level {
            0 => Color32::from_rgb(128, 128, 128),
            1 => Color32::from_rgb(100, 149, 237),
            2 => Color32::from_rgb(144, 238, 144),
            3 => Color32::from_rgb(255, 200, 50),
            4 => Color32::from_rgb(255, 80, 80),
            _ => Color32::WHITE,
        }
    }
}

fn parse_log_entry(data: &[u8]) -> Option<LogEntry> {
    postcard::from_bytes::<LogEntry>(data).ok()
}

fn send_stream_request(commands: &mut Commands, entity: Entity, msg_id: [u8; 2], stream_id: u64) {
    commands.send_msg_req_reply_raw(
        FixedRateMsgStream {
            msg_id,
            fixed_rate: FixedRateOp {
                stream_id,
                behavior: Default::default(),
            },
        },
        move |InRef(pkt): InRef<OwnedPacket<PacketGrantR>>, mut caches: Query<&mut LogCache>| {
            if let OwnedPacket::Msg(msg_buf) = pkt
                && let Some(timestamp) = msg_buf.timestamp
                && let Ok(mut cache) = caches.get_mut(entity)
                && let Some(entry) = parse_log_entry(&msg_buf.buf)
            {
                cache.insert(timestamp, entry);
                cache.last_stream_activity = Some(Instant::now());
            }
            false
        },
    );
}

fn send_backfill_request(
    commands: &mut Commands,
    entity: Entity,
    msg_id: [u8; 2],
    start_from: Timestamp,
) {
    commands.send_msg_req_reply(
        GetMsgs {
            msg_id,
            range: start_from..Timestamp(i64::MAX),
            limit: Some(500),
        },
        move |In(result): In<Result<MsgBatch, ErrorResponse>>, mut caches: Query<&mut LogCache>| {
            match result {
                Ok(batch) => {
                    if let Ok(mut cache) = caches.get_mut(entity) {
                        for (ts, data) in &batch.data {
                            if let Some(entry) = parse_log_entry(data) {
                                cache.insert(*ts, entry);
                            }
                        }
                    }
                }
                Err(e) => {
                    bevy::log::warn!("log backfill error: {}", e);
                }
            }
            true
        },
    );
}

pub fn connect_streams(
    mut query: Query<(Entity, &mut LogStreamState, &mut LogCache)>,
    mut commands: Commands,
    stream_id: Res<CurrentStreamId>,
) {
    for (entity, mut state, mut cache) in &mut query {
        match &mut state.connection_state {
            ConnectionState::WaitingToConnect { frames_waited } => {
                *frames_waited += 1;
                if *frames_waited >= FRAMES_BEFORE_CONNECT {
                    let msg_id = state.msg_id;
                    send_backfill_request(&mut commands, entity, msg_id, Timestamp(i64::MIN));
                    send_stream_request(&mut commands, entity, msg_id, stream_id.0);
                    state.connection_state = ConnectionState::Active;
                }
            }
            ConnectionState::Active => {
                if let Some(last) = cache.last_stream_activity
                    && last.elapsed().as_secs_f32() > STREAM_RECOVERY_TIMEOUT_SECS
                {
                    let msg_id = state.msg_id;
                    let start = cache
                        .entries
                        .keys()
                        .next_back()
                        .map(|timestamp| Timestamp(timestamp.0 + 1))
                        .unwrap_or(Timestamp(i64::MIN));
                    send_backfill_request(&mut commands, entity, msg_id, start);
                    send_stream_request(&mut commands, entity, msg_id, stream_id.0);
                    cache.last_stream_activity = None;
                }
            }
        }
    }
}

#[derive(QueryData)]
#[query_data(mutable)]
pub struct LogWidgetQuery {
    state: &'static LogStreamState,
    cache: &'static mut LogCache,
}

#[derive(SystemParam)]
pub struct LogStreamWidget<'w, 's> {
    query: Query<'w, 's, LogWidgetQuery>,
    current_time: Res<'w, CurrentTimestamp>,
}

impl super::widgets::WidgetSystem for LogStreamWidget<'_, '_> {
    type Args = LogStreamWidgetArgs;
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        LogStreamWidgetArgs { entity }: Self::Args,
    ) -> Self::Output {
        let mut state = state.get_mut(world);
        let current_ts = state.current_time.0;
        let Ok(LogWidgetQueryItem {
            state: log_state,
            mut cache,
        }) = state.query.get_mut(entity)
        else {
            return;
        };

        ui.vertical(|ui| {
            let filtered: Vec<_> = cache
                .entries
                .range(..=current_ts)
                .flat_map(|(ts, entries)| entries.iter().map(move |entry| (*ts, entry.clone())))
                .filter(|(_, entry)| entry.level >= cache.filter_level)
                .collect();

            // Toolbar
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!("{} ", log_state.msg_name))
                        .color(Color32::from_rgb(180, 180, 180))
                        .small(),
                );
                ui.separator();

                let levels = ["TRACE", "DEBUG", "INFO", "WARN", "ERROR"];
                let level_colors = [
                    Color32::from_rgb(128, 128, 128),
                    Color32::from_rgb(100, 149, 237),
                    Color32::from_rgb(144, 238, 144),
                    Color32::from_rgb(255, 200, 50),
                    Color32::from_rgb(255, 80, 80),
                ];
                for (i, (name, color)) in levels.iter().zip(level_colors.iter()).enumerate() {
                    let active = cache.filter_level <= i as u8;
                    let label = if active {
                        RichText::new(*name).small().color(*color)
                    } else {
                        RichText::new(*name)
                            .small()
                            .color(Color32::from_rgb(80, 80, 80))
                    };
                    if ui.selectable_label(active, label).clicked() {
                        cache.filter_level = if active { ((i as u8) + 1).min(4) } else { i as u8 };
                    }
                }

                ui.separator();
                let scroll_label = if cache.auto_scroll {
                    "Auto-scroll ON"
                } else {
                    "Auto-scroll OFF"
                };
                if ui
                    .selectable_label(cache.auto_scroll, RichText::new(scroll_label).small())
                    .clicked()
                {
                    cache.auto_scroll = !cache.auto_scroll;
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        RichText::new(format!("{} entries", filtered.len()))
                            .small()
                            .color(Color32::from_rgb(120, 120, 120)),
                    );
                });
            });

            ui.separator();

            // Log entries
            let row_height = 18.0;

            ScrollArea::vertical()
                .auto_shrink([false, false])
                .stick_to_bottom(cache.auto_scroll)
                .show_rows(ui, row_height, filtered.len(), |ui, row_range| {
                    for i in row_range {
                        if let Some((ts, entry)) = filtered.get(i) {
                            ui.horizontal(|ui| {
                                let ts_secs = ts.0 as f64 / 1_000_000.0;
                                ui.label(
                                    RichText::new(format!("{ts_secs:>12.3}"))
                                        .monospace()
                                        .small()
                                        .color(Color32::from_rgb(120, 120, 120)),
                                );

                                let badge = RichText::new(format!(" {:5} ", entry.level_str()))
                                    .monospace()
                                    .small()
                                    .color(entry.level_color());
                                ui.label(badge);

                                ui.label(
                                    RichText::new(&entry.message)
                                        .monospace()
                                        .small()
                                        .color(Color32::from_rgb(220, 220, 220)),
                                );
                            });
                        }
                    }
                });
        });
    }
}
