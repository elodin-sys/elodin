use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::error::Error;
use crate::nal;

const MAX_READY: usize = 8;

struct Ready {
    pts_us: i64,
    frame: web_sys::VideoFrame,
}

struct Inner {
    /// PTS values submitted since the last reset, in decode order.
    /// An output whose timestamp is not the front of this queue is from a
    /// superseded seek and is closed.
    submitted: VecDeque<i64>,
    queue: VecDeque<Ready>,
    /// Drop outputs with a PTS strictly below this. Seek intermediates never
    /// occupy a `VideoFrame` slot.
    discard_before: Option<i64>,
    open: usize,
    decoded: u64,
    dropped: u64,
    error: Option<Error>,
    /// Non-zero while a `flush()` promise is outstanding.
    flush_inflight: u64,
    flush_generation: u64,
    /// Set when `flush()` runs. The next chunk after a flush has to be a keyframe.
    needs_keyframe: bool,
}

pub struct Decoder {
    video: web_sys::VideoDecoder,
    codec: String,
    width: u32,
    height: u32,
    /// SPS and PPS from the first keyframe. Samples stay Annex-B with the
    /// parameter sets in band: Safari rejects every `description` (avcC)
    /// configuration, while both Safari and Chrome decode Annex-B without one.
    parameter_sets: Option<(Vec<u8>, Vec<u8>)>,
    /// Safari does not emit the last queued frame until `flush()`.
    pub drain_with_flush: bool,
    inner: Rc<RefCell<Inner>>,
    _output: Closure<dyn FnMut(web_sys::VideoFrame)>,
    _error: Closure<dyn FnMut(JsValue)>,
}

pub struct Popped {
    pub pts_us: i64,
    pub frame: web_sys::VideoFrame,
}

impl Decoder {
    /// `codec` may be empty. The first keyframe's SPS then supplies `avc1.PPCCLL`.
    pub fn new(codec: &str) -> Result<Self, Error> {
        let inner = Rc::new(RefCell::new(Inner {
            submitted: VecDeque::new(),
            queue: VecDeque::new(),
            discard_before: None,
            open: 0,
            decoded: 0,
            dropped: 0,
            error: None,
            flush_inflight: 0,
            flush_generation: 0,
            needs_keyframe: false,
        }));
        let output_inner = inner.clone();
        let output = Closure::wrap(Box::new(move |frame: web_sys::VideoFrame| {
            on_output(&output_inner, frame);
        }) as Box<dyn FnMut(web_sys::VideoFrame)>);
        let error_inner = inner.clone();
        let error = Closure::wrap(Box::new(move |err: JsValue| {
            let mut inner = error_inner.borrow_mut();
            if inner.error.is_none() {
                inner.error = Some(Error::from(err));
            }
        }) as Box<dyn FnMut(JsValue)>);
        let init = web_sys::VideoDecoderInit::new(
            error.as_ref().unchecked_ref(),
            output.as_ref().unchecked_ref(),
        );
        let video = web_sys::VideoDecoder::new(&init)?;
        Ok(Self {
            video,
            codec: codec.to_string(),
            width: 0,
            height: 0,
            parameter_sets: None,
            drain_with_flush: crate::is_safari(),
            inner,
            _output: output,
            _error: error,
        })
    }

    /// Container dimensions. Omitted from the decoder config when unset.
    pub fn set_coded_size(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    /// Safari rejects `decode()` while `flush()` is unresolved.
    pub fn is_blocked(&self) -> bool {
        self.inner.borrow().flush_inflight != 0
    }

    /// `flush()` leaves the decoder configured, and the next chunk must be a keyframe.
    pub fn needs_keyframe(&self) -> bool {
        self.inner.borrow().needs_keyframe
    }

    pub fn decode_queue_size(&self) -> u32 {
        self.video.decode_queue_size()
    }

    pub fn open_frames(&self) -> usize {
        self.inner.borrow().open
    }

    pub fn decoded(&self) -> u64 {
        self.inner.borrow().decoded
    }

    pub fn dropped(&self) -> u64 {
        self.inner.borrow().dropped
    }

    pub fn error(&self) -> Option<Error> {
        self.inner.borrow().error.clone()
    }

    pub fn set_discard_before(&self, pts_us: Option<i64>) {
        self.inner.borrow_mut().discard_before = pts_us;
    }

    /// Abort pending outputs and reconfigure. In-flight frames are closed.
    pub fn reset(&mut self) -> Result<(), Error> {
        {
            let mut inner = self.inner.borrow_mut();
            inner.submitted.clear();
            inner.discard_before = None;
            inner.flush_generation = inner.flush_generation.wrapping_add(1).max(1);
            inner.flush_inflight = 0;
            inner.needs_keyframe = false;
            while let Some(ready) = inner.queue.pop_front() {
                ready.frame.close();
                inner.open = inner.open.saturating_sub(1);
            }
        }
        if self.video.state() == web_sys::CodecState::Closed {
            self.rebuild()?;
        } else if self.video.state() != web_sys::CodecState::Unconfigured
            && self.video.reset().is_err()
        {
            self.rebuild()?;
        }
        if self.parameter_sets.is_some() {
            self.configure()?;
        }
        self.inner.borrow_mut().error = None;
        Ok(())
    }

    /// Release frames the decoder is still holding. A high-profile stream
    /// keeps the last sample until the next one, or until `flush`. The next
    /// sample after this returns has to be a keyframe.
    pub fn flush(&self) -> Result<(), Error> {
        if self.video.state() != web_sys::CodecState::Configured {
            return Ok(());
        }
        {
            let inner = self.inner.borrow();
            if inner.flush_inflight != 0 || inner.submitted.is_empty() {
                return Ok(());
            }
        }
        let generation = {
            let mut inner = self.inner.borrow_mut();
            inner.flush_generation = inner.flush_generation.wrapping_add(1).max(1);
            inner.flush_inflight = inner.flush_generation;
            inner.needs_keyframe = true;
            inner.flush_generation
        };
        let promise = self.video.flush();
        let inner = self.inner.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let result = JsFuture::from(promise).await;
            let mut inner = inner.borrow_mut();
            if inner.flush_inflight != generation {
                return;
            }
            inner.flush_inflight = 0;
            if let Err(err) = result
                && inner.error.is_none()
            {
                inner.error = Some(Error::from(err));
            }
        });
        Ok(())
    }

    pub fn submit(
        &mut self,
        pts_us: i64,
        annexb: &[u8],
        key: bool,
        duration_us: f64,
    ) -> Result<(), Error> {
        if self.video.state() == web_sys::CodecState::Closed {
            self.rebuild()?;
        }
        let sample = self.prepare_sample(annexb, key)?;
        if self.video.state() != web_sys::CodecState::Configured {
            self.configure()?;
        }
        let chunk_type = if key {
            web_sys::EncodedVideoChunkType::Key
        } else {
            web_sys::EncodedVideoChunkType::Delta
        };
        let chunk = {
            let bytes = js_sys::Uint8Array::from(sample.as_slice());
            let init = web_sys::EncodedVideoChunkInit::new_with_u8_array(&bytes, 0, chunk_type);
            init.set_timestamp_f64(pts_us as f64);
            // Safari drops chunks whose duration is missing.
            init.set_duration_f64(duration_us.max(1.0));
            web_sys::EncodedVideoChunk::new(&init)?
        };
        self.inner.borrow_mut().submitted.push_back(pts_us);
        if let Err(err) = self.video.decode(&chunk) {
            self.inner.borrow_mut().submitted.pop_back();
            return Err(err.into());
        }
        Ok(())
    }

    /// Pop the frame with this PTS, closing anything older still in the queue.
    pub fn pop_pts(&mut self, pts_us: i64) -> Option<web_sys::VideoFrame> {
        let mut inner = self.inner.borrow_mut();
        loop {
            let front = inner.queue.front()?;
            if front.pts_us < pts_us {
                let ready = inner.queue.pop_front().unwrap();
                ready.frame.close();
                inner.open = inner.open.saturating_sub(1);
                inner.dropped += 1;
                continue;
            }
            if front.pts_us > pts_us {
                return None;
            }
            let ready = inner.queue.pop_front().unwrap();
            inner.open = inner.open.saturating_sub(1);
            return Some(ready.frame);
        }
    }

    /// Latest ready frame at or before `playhead_us`. Older ready frames are
    /// closed and counted as dropped.
    pub fn pop_upto(&mut self, playhead_us: i64) -> Option<Popped> {
        let mut inner = self.inner.borrow_mut();
        let mut chosen: Option<Ready> = None;
        while inner
            .queue
            .front()
            .is_some_and(|ready| ready.pts_us <= playhead_us)
        {
            let ready = inner.queue.pop_front().unwrap();
            if let Some(previous) = chosen.replace(ready) {
                previous.frame.close();
                inner.open = inner.open.saturating_sub(1);
                inner.dropped += 1;
            }
        }
        chosen.map(|ready| {
            inner.open = inner.open.saturating_sub(1);
            Popped {
                pts_us: ready.pts_us,
                frame: ready.frame,
            }
        })
    }

    pub fn close_frame(&self, frame: web_sys::VideoFrame) {
        frame.close();
    }

    fn rebuild(&mut self) -> Result<(), Error> {
        let init = web_sys::VideoDecoderInit::new(
            self._error.as_ref().unchecked_ref(),
            self._output.as_ref().unchecked_ref(),
        );
        self.video = web_sys::VideoDecoder::new(&init)?;
        Ok(())
    }

    fn prepare_sample(&mut self, annexb: &[u8], key: bool) -> Result<Vec<u8>, Error> {
        let nals = nal::annexb_nals(annexb);
        if nals.is_empty() {
            return Err(Error::EmptyAccessUnit);
        }
        let sps = nals
            .iter()
            .find(|nal| nal::nal_type(nal) == Some(7))
            .copied();
        let pps = nals
            .iter()
            .find(|nal| nal::nal_type(nal) == Some(8))
            .copied();
        if self.parameter_sets.is_none() {
            let (Some(sps), Some(pps)) = (sps, pps) else {
                return Err(Error::MissingParameterSets);
            };
            if self.codec.is_empty() {
                self.codec = nal::avc1_codec_string(sps).ok_or(Error::ShortSps)?;
            }
            self.parameter_sets = Some((sps.to_vec(), pps.to_vec()));
        }
        let Some((cached_sps, cached_pps)) = self.parameter_sets.as_ref() else {
            return Err(Error::MissingParameterSets);
        };
        if !nals
            .iter()
            .any(|nal| matches!(nal::nal_type(nal), Some(1..=5)))
        {
            return Err(Error::NoPictureNal);
        }
        if key && !nals.iter().any(|nal| nal::nal_type(nal) == Some(5)) {
            return Err(Error::NoIdr);
        }
        // Without a `description`, every keyframe has to carry its own
        // parameter sets. Access unit delimiters are dropped.
        let mut sample: Vec<&[u8]> = Vec::with_capacity(nals.len() + 2);
        if key && (sps.is_none() || pps.is_none()) {
            sample.push(cached_sps);
            sample.push(cached_pps);
        }
        sample.extend(nals.into_iter().filter(|nal| nal::nal_type(nal) != Some(9)));
        Ok(nal::nals_to_annexb(&sample))
    }

    fn configure(&self) -> Result<(), Error> {
        if self.codec.is_empty() {
            return Err(Error::UnknownCodec);
        }
        let config = web_sys::VideoDecoderConfig::new(&self.codec);
        if self.width > 0 && self.height > 0 {
            config.set_coded_width(self.width);
            config.set_coded_height(self.height);
        }
        config.set_optimize_for_latency(true);
        self.video.configure(&config)?;
        if self.video.state() != web_sys::CodecState::Configured {
            return Err(Error::ConfigureFailed);
        }
        Ok(())
    }
}

fn on_output(inner: &Rc<RefCell<Inner>>, frame: web_sys::VideoFrame) {
    let reported = frame.timestamp().round() as i64;
    let mut inner = inner.borrow_mut();
    // Outputs are in decode order. Safari sometimes reports timestamp 0, so a
    // non-zero timestamp that is not the one we submitted is a stale frame
    // and must not consume the queue.
    let Some(&expected) = inner.submitted.front() else {
        frame.close();
        return;
    };
    if reported != 0 && reported != expected {
        frame.close();
        return;
    }
    inner.submitted.pop_front();
    let pts_us = expected;
    inner.decoded += 1;
    if inner.discard_before.is_some_and(|until| pts_us < until) {
        frame.close();
        return;
    }
    while inner.queue.len() >= MAX_READY {
        if let Some(old) = inner.queue.pop_front() {
            old.frame.close();
            inner.open = inner.open.saturating_sub(1);
            inner.dropped += 1;
        }
    }
    inner.open += 1;
    inner.queue.push_back(Ready { pts_us, frame });
}
