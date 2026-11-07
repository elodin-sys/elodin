use super::*;
use impeller_bevy::browser_cache::{self, BrowserCacheScope, CacheRead};
use std::collections::{BTreeMap, BTreeSet};

const CHUNK_US: i64 = 250_000;

#[derive(Clone)]
struct Page {
    scope: String,
    generation: u64,
    block: i64,
    start: i64,
    end: i64,
    limit: usize,
    key: String,
}

struct Pending {
    page: Page,
    read: CacheRead,
    started: Instant,
}

pub(super) struct BrowserVideoState {
    scope: String,
    generation: u64,
    center: i64,
    updated: Instant,
    complete: BTreeSet<i64>,
    loaded: BTreeMap<i64, (Timestamp, Timestamp)>,
    cursors: BTreeMap<i64, i64>,
    pending: Option<Pending>,
    network: Option<(u64, String, Instant)>,
    timeouts: usize,
}

impl BrowserVideoState {
    pub(super) fn busy(&self) -> bool {
        self.pending.is_some() || self.network.is_some()
    }

    fn new(scope: String, center: i64) -> Self {
        Self {
            scope,
            generation: 0,
            center,
            updated: Instant::now(),
            complete: BTreeSet::new(),
            loaded: BTreeMap::new(),
            cursors: BTreeMap::new(),
            pending: None,
            network: None,
            timeouts: 0,
        }
    }

    fn move_to(&mut self, center: i64) {
        let now = Instant::now();
        if browser_cache::is_seek(
            self.center,
            center,
            now.duration_since(self.updated),
            1_000_000,
        ) {
            self.generation = self.generation.wrapping_add(1);
        }
        self.updated = now;
        self.center = center;
        self.complete
            .retain(|time| time.abs_diff(center) <= 3_000_000);
        self.cursors
            .retain(|time, _| time.abs_diff(center) <= 3_000_000);
        self.loaded
            .retain(|time, _| time.abs_diff(center) <= 3_000_000);
    }
}

fn accept(cache: &mut VideoFrameCache, page: &Page, batch: MsgBatch) -> bool {
    let Some(state) = cache.browser.as_ref() else {
        return false;
    };
    if state.scope != page.scope || state.generation != page.generation {
        return false;
    }
    if batch.data.len() > page.limit
        || batch
            .data
            .iter()
            .any(|(_, bytes)| bytes.len() > cache.max_raw_bytes)
        || batch.data.windows(2).any(|pair| pair[0].0 > pair[1].0)
    {
        return false;
    }
    let count = batch.data.len();
    let mut last = page.start;
    let mut first_loaded = None;
    let mut last_loaded = None;
    for (time, data) in batch.data {
        // GetMsgs includes a predecessor, and can return its first frame before the log starts.
        last = last.max(time.0);
        if time.0 >= page.end {
            continue;
        }
        first_loaded.get_or_insert(time);
        last_loaded = Some(time);
        let anchor = cache.eviction_pivot.unwrap_or(time);
        cache.insert_raw(time, data, anchor);
    }
    let state = cache.browser.as_mut().expect("checked browser state");
    if let (Some(first), Some(last)) = (first_loaded, last_loaded) {
        state
            .loaded
            .entry(page.block)
            .and_modify(|bounds| {
                bounds.0 = bounds.0.min(first);
                bounds.1 = bounds.1.max(last);
            })
            .or_insert((first, last));
    }
    if count < page.limit || last >= page.end.saturating_sub(1) {
        state.complete.insert(page.block);
        state.cursors.remove(&page.block);
    } else {
        state.cursors.insert(page.block, last.saturating_add(1));
    }
    true
}

pub(super) fn update(
    entity: Entity,
    stream: &mut VideoStream,
    cache: &mut VideoFrameCache,
    timestamp: Timestamp,
    scope: &BrowserCacheScope,
    slots: &mut usize,
    commands: &mut Commands,
) {
    let Some(scope) = scope.key.as_ref() else {
        cache.clear_raw_keeping(None);
        cache.browser = None;
        return;
    };
    let center = browser_cache::window_start(timestamp.0, CHUNK_US);
    if cache
        .browser
        .as_ref()
        .is_none_or(|state| &state.scope != scope)
    {
        cache.clear_raw_keeping(None);
        cache.browser = Some(BrowserVideoState::new(scope.clone(), center));
    }
    let state = cache.browser.as_mut().expect("browser state initialized");
    state.move_to(center);
    let evicted: Vec<_> = state
        .loaded
        .iter()
        .filter_map(|(&block, &(first, last))| {
            (!cache.raw_frames.contains_key(&first) || !cache.raw_frames.contains_key(&last))
                .then_some(block)
        })
        .collect();
    for block in evicted {
        state.complete.remove(&block);
        state.cursors.remove(&block);
        state.loaded.remove(&block);
    }
    if state
        .network
        .as_ref()
        .is_some_and(|(_, _, started)| started.elapsed() >= browser_cache::REQUEST_TIMEOUT)
    {
        state.network = None;
        state.generation = state.generation.wrapping_add(1);
        state.timeouts += 1;
        if state.timeouts >= 3 {
            let error = "Camera requests timed out. Reopen the recording.".to_owned();
            stream.state = StreamState::Error(error.clone());
            commands.write_message(crate::viewer::ViewerEvent::Failed(error));
            return;
        }
    }
    let ready = state.pending.as_mut().and_then(|pending| {
        pending
            .read
            .poll()
            .map(|bytes| (pending.page.clone(), bytes))
    });
    let expired = state
        .pending
        .as_ref()
        .is_some_and(|pending| pending.started.elapsed().as_secs() >= 3);
    if let Some((page, bytes)) = ready {
        state.pending = None;
        if page.generation == state.generation && page.scope == state.scope {
            let accepted = bytes
                .as_deref()
                .and_then(browser_cache::decode::<MsgBatch>)
                .is_some_and(|batch| accept(cache, &page, batch));
            if !accepted {
                send(entity, stream.msg_id, page, cache, commands);
            }
        }
    } else if expired {
        let pending = cache
            .browser
            .as_mut()
            .expect("browser state")
            .pending
            .take()
            .expect("expired read");
        if pending.page.generation == cache.browser.as_ref().expect("browser state").generation {
            send(entity, stream.msg_id, pending.page, cache, commands);
        }
    }
    let state = cache.browser.as_mut().expect("browser state");
    if state.busy() || *slots == 0 {
        return;
    }
    // Prefer the current frame, then forward playback, then backward seek context.
    let mut offsets = vec![0, 1, 2, 3, 4, -1, -2];
    if cache.is_h264 {
        offsets.extend((-120..-2).rev());
    }
    let Some(block) = offsets
        .into_iter()
        .map(|offset| center.saturating_add(offset * CHUNK_US))
        .find(|block| !state.complete.contains(block))
    else {
        return;
    };
    let start = state.cursors.get(&block).copied().unwrap_or(block);
    let end = block.saturating_add(CHUNK_US);
    let limit = backfill_frame_limit(stream.raw_dims, cache.is_h264).max(2);
    let key = format!(
        "video/{}/{}/{start}/{end}/{limit}",
        stream.msg_id[0], stream.msg_id[1]
    );
    let page = Page {
        scope: scope.clone(),
        generation: state.generation,
        block,
        start,
        end,
        limit,
        key,
    };
    let read = browser_cache::lookup(scope, &page.key);
    state.pending = Some(Pending {
        page,
        read,
        started: Instant::now(),
    });
    *slots -= 1;
}

fn send(
    entity: Entity,
    msg_id: [u8; 2],
    page: Page,
    cache: &mut VideoFrameCache,
    commands: &mut Commands,
) {
    cache.browser.as_mut().expect("browser state").network =
        Some((page.generation, page.key.clone(), Instant::now()));
    let request = GetMsgs {
        msg_id,
        range: Timestamp(page.start)..Timestamp(page.end.saturating_sub(1)),
        limit: Some(page.limit),
    };
    commands.send_msg_req_reply(
        request,
        move |In(result): In<Result<MsgBatch, ErrorResponse>>,
              mut caches: Query<(&mut VideoFrameCache, &mut VideoStream)>,
              mut commands: Commands,
              active_scope: Res<BrowserCacheScope>| {
            let Ok((mut cache, mut stream)) = caches.get_mut(entity) else {
                return true;
            };
            let Some(state) = cache.browser.as_mut() else {
                return true;
            };
            if active_scope.key.as_ref() != Some(&page.scope) || state.scope != page.scope {
                return true;
            }
            if state.network.as_ref().is_some_and(|(generation, key, _)| {
                *generation == page.generation && key == &page.key
            }) {
                state.network = None;
            }
            if state.generation != page.generation {
                return true;
            }
            if let Ok(batch) = result {
                state.timeouts = 0;
                let encoded = postcard::to_allocvec(&batch).ok();
                if accept(&mut cache, &page, batch) {
                    if let Some(bytes) = encoded {
                        browser_cache::put(&page.scope, &page.key, &bytes);
                    }
                    return true;
                }
            }
            let error = "Camera data could not load. Reopen the recording.".to_owned();
            stream.state = StreamState::Error(error.clone());
            commands.write_message(crate::viewer::ViewerEvent::Failed(error));
            true
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_seek_responses_cannot_change_the_new_window() {
        let mut cache = VideoFrameCache::for_raw();
        cache.browser = Some(BrowserVideoState::new("session".into(), 0));
        let page = Page {
            scope: "session".into(),
            generation: 0,
            block: 0,
            start: 0,
            end: CHUNK_US,
            limit: 4,
            key: "first".into(),
        };
        cache.browser.as_mut().unwrap().move_to(10_000_000);
        accept(
            &mut cache,
            &page,
            MsgBatch {
                data: vec![(Timestamp(10), vec![1; 4])],
            },
        );
        assert!(cache.raw_frames.is_empty());
        assert!(cache.browser.as_ref().unwrap().complete.is_empty());
    }

    #[test]
    fn byte_limits_keep_frames_near_the_playhead() {
        let mut cache = VideoFrameCache::for_raw();
        cache.max_raw_bytes = 12;
        cache.browser = Some(BrowserVideoState::new("recording".into(), 20));
        cache.last_decoded_ts = Some(Timestamp(20));
        cache.insert_raw(Timestamp(10), vec![1; 8], Timestamp(20));
        cache.insert_raw(Timestamp(20), vec![2; 8], Timestamp(20));
        assert_eq!(cache.raw_frames.len(), 1);
        assert!(cache.raw_frames.contains_key(&Timestamp(20)));
        cache.insert_raw(Timestamp(30), vec![3; 16], Timestamp(20));
        assert_eq!(cache.raw_frames.len(), 1);
    }
    #[test]
    fn decoded_images_obey_bytes_when_frame_sizes_differ() {
        let mut cache = VideoFrameCache::for_raw();
        cache.max_decoded_bytes = 12;
        cache.eviction_pivot = Some(Timestamp(20));
        for (time, size) in [(0, 4), (10, 8), (20, 8)] {
            let mut image = Image::default();
            image.data = Some(vec![0; size]);
            cache.insert_decoded(Timestamp(time), image);
        }
        assert_eq!(
            cache
                .decoded_frames
                .values()
                .map(|image| image.data.as_ref().unwrap().len())
                .sum::<usize>(),
            8
        );
        assert!(cache.decoded_frames.contains_key(&Timestamp(20)));
    }

    #[test]
    fn camera_pages_keep_a_cursor_and_ignore_the_upper_boundary() {
        let mut cache = VideoFrameCache::for_raw();
        cache.browser = Some(BrowserVideoState::new("session".into(), 0));
        let page = Page {
            scope: "session".into(),
            generation: 0,
            block: 0,
            start: 0,
            end: CHUNK_US,
            limit: 2,
            key: "page".into(),
        };
        assert!(accept(
            &mut cache,
            &page,
            MsgBatch {
                data: vec![(Timestamp(1), vec![1]), (Timestamp(2), vec![2])]
            }
        ));
        assert_eq!(cache.browser.as_ref().unwrap().cursors.get(&0), Some(&3));
        assert!(!cache.browser.as_ref().unwrap().complete.contains(&0));
        assert!(accept(
            &mut cache,
            &page,
            MsgBatch {
                data: vec![(Timestamp(CHUNK_US), vec![3])]
            }
        ));
        assert!(!cache.raw_frames.contains_key(&Timestamp(CHUNK_US)));
        let next = Page { start: 3, ..page };
        assert!(accept(
            &mut cache,
            &next,
            MsgBatch {
                data: vec![(Timestamp(3), vec![3])]
            }
        ));
        assert!(cache.browser.as_ref().unwrap().complete.contains(&0));
    }
    #[test]
    fn message_pages_accept_the_database_predecessor() {
        let mut cache = VideoFrameCache::for_raw();
        cache.browser = Some(BrowserVideoState::new("session".into(), 0));
        let page = Page {
            scope: "session".into(),
            generation: 0,
            block: 0,
            start: 10,
            end: 30,
            limit: 2,
            key: "page".into(),
        };
        assert!(accept(
            &mut cache,
            &page,
            MsgBatch {
                data: vec![(Timestamp(5), vec![5]), (Timestamp(15), vec![15])]
            }
        ));
        assert_eq!(cache.browser.as_ref().unwrap().cursors.get(&0), Some(&16));
        let next = Page { start: 16, ..page };
        assert!(accept(
            &mut cache,
            &next,
            MsgBatch {
                data: vec![(Timestamp(15), vec![15]), (Timestamp(25), vec![25])]
            }
        ));
        assert_eq!(cache.browser.as_ref().unwrap().cursors.get(&0), Some(&26));
        assert_eq!(cache.raw_frames.len(), 3);
    }
}
