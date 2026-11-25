//! `globalThis.elodin.captureScreenshot()` resolves to a PNG `Blob` of the editor window,
//! or rejects with an `Error` when readback or encoding fails.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bevy::prelude::*;
use bevy::render::view::window::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::window::PrimaryWindow;
use js_sys::{Function, Promise};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

/// Readback normally finishes within a few frames. After this, the screenshot
/// entity is still sitting there with no [`ScreenshotCaptured`].
const READBACK_TIMEOUT_SECS: f32 = 5.0;

pub struct WebScreenshotPlugin;

impl Plugin for WebScreenshotPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, install)
            .add_systems(Update, dispatch);
    }
}

struct Inflight {
    reject: Function,
    settled: Arc<AtomicBool>,
    entity: Entity,
    deadline: f32,
}

struct Bridge {
    queue: Rc<RefCell<VecDeque<(Function, Function)>>>,
    inflight: RefCell<Option<Inflight>>,
    _capture: Closure<dyn FnMut() -> Promise>,
}

fn install(world: &mut World) {
    let queue = Rc::new(RefCell::new(VecDeque::new()));
    let queued = Rc::clone(&queue);
    let capture = Closure::wrap(Box::new(move || {
        let queued = Rc::clone(&queued);
        Promise::new(&mut |resolve, reject| {
            queued.borrow_mut().push_back((resolve, reject));
        })
    }) as Box<dyn FnMut() -> Promise>);
    if let Err(err) = publish(capture.as_ref().unchecked_ref()) {
        error!("captureScreenshot was not installed: {err:?}");
    }
    world.insert_non_send(Bridge {
        queue,
        inflight: RefCell::new(None),
        _capture: capture,
    });
}

fn publish(capture: &Function) -> Result<(), JsValue> {
    let global = js_sys::global();
    let key = JsValue::from_str("elodin");
    let existing = js_sys::Reflect::get(&global, &key)?;
    let elodin = if existing.is_object() {
        existing
    } else {
        let object = js_sys::Object::new().into();
        js_sys::Reflect::set(&global, &key, &object)?;
        object
    };
    js_sys::Reflect::set(&elodin, &JsValue::from_str("captureScreenshot"), capture)?;
    Ok(())
}

fn dispatch(
    mut commands: Commands,
    bridge: NonSend<Bridge>,
    windows: Query<(), With<PrimaryWindow>>,
    screenshots: Query<(), With<Screenshot>>,
    time: Res<Time>,
) {
    // One primary-window capture at a time. Bevy despawns a second screenshot of
    // the same target without `ScreenshotCaptured`, and a skipped readback never
    // emits that event. Either way the popped promise has to reject.
    if !ready_for_capture(&mut commands, &bridge, &screenshots, &time) {
        return;
    }
    if windows.is_empty() {
        return;
    }
    let Some((resolve, reject)) = bridge.queue.borrow_mut().pop_front() else {
        return;
    };
    let settled = Arc::new(AtomicBool::new(false));
    let mut screenshot = commands.spawn(Screenshot::primary_window());
    let entity = screenshot.id();
    let on_capture = Arc::clone(&settled);
    let reject_capture = reject.clone();
    screenshot.observe(move |captured: On<ScreenshotCaptured>| {
        if on_capture.swap(true, Ordering::AcqRel) {
            return;
        }
        deliver(&captured.image, resolve.clone(), reject_capture.clone());
    });
    let on_remove = Arc::clone(&settled);
    let reject_remove = reject.clone();
    screenshot.observe(move |_remove: On<Remove, Screenshot>| {
        reject_lost(&reject_remove, &on_remove);
    });
    *bridge.inflight.borrow_mut() = Some(Inflight {
        reject,
        settled,
        entity,
        deadline: time.elapsed_secs() + READBACK_TIMEOUT_SECS,
    });
}

/// Settles a finished or lost capture. `true` when another capture may start.
fn ready_for_capture(
    commands: &mut Commands,
    bridge: &Bridge,
    screenshots: &Query<(), With<Screenshot>>,
    time: &Time,
) -> bool {
    enum Hold {
        Idle,
        Wait,
        Done,
        Lost { despawn: bool },
    }
    let hold = {
        let inflight = bridge.inflight.borrow();
        match inflight.as_ref() {
            None => Hold::Idle,
            Some(capture) if capture.settled.load(Ordering::Acquire) => Hold::Done,
            Some(capture) => {
                let alive = screenshots.contains(capture.entity);
                let timed_out = time.elapsed_secs() >= capture.deadline;
                if alive && !timed_out {
                    Hold::Wait
                } else {
                    Hold::Lost { despawn: alive }
                }
            }
        }
    };
    match hold {
        Hold::Idle => true,
        Hold::Wait => false,
        Hold::Done => {
            bridge.inflight.borrow_mut().take();
            true
        }
        Hold::Lost { despawn } => {
            if let Some(capture) = bridge.inflight.borrow_mut().take() {
                if despawn {
                    commands.entity(capture.entity).try_despawn();
                }
                reject_lost(&capture.reject, &capture.settled);
            }
            true
        }
    }
}

fn reject_lost(reject: &Function, settled: &AtomicBool) {
    if settled.swap(true, Ordering::AcqRel) {
        return;
    }
    warn!("screenshot readback lost");
    reject_with(reject, "screenshot readback lost");
}

fn deliver(image: &Image, resolve: Function, reject: Function) {
    let rgba = match image.clone().try_into_dynamic() {
        Ok(dynamic) => dynamic.to_rgba8(),
        Err(err) => {
            reject_with(&reject, &err.to_string());
            return;
        }
    };
    let (width, height) = (rgba.width(), rgba.height());
    if width == 0 || height == 0 {
        reject_with(&reject, "screenshot is empty");
        return;
    }
    let promise = match png_blob(width, height, rgba.as_raw()) {
        Ok(promise) => promise,
        Err(err) => {
            reject_js(&reject, err);
            return;
        }
    };
    wasm_bindgen_futures::spawn_local(async move {
        match JsFuture::from(promise).await {
            Ok(blob) => {
                let _ = resolve.call1(&JsValue::UNDEFINED, &blob);
            }
            Err(err) => reject_js(&reject, err),
        }
    });
}

fn png_blob(width: u32, height: u32, pixels: &[u8]) -> Result<Promise, JsValue> {
    let image = web_sys::ImageData::new_with_u8_clamped_array_and_sh(
        wasm_bindgen::Clamped(pixels),
        width,
        height,
    )?;
    let canvas = web_sys::OffscreenCanvas::new(width, height)?;
    let context = canvas
        .get_context("2d")?
        .ok_or_else(|| JsValue::from_str("2d context unavailable"))?;
    let context: web_sys::OffscreenCanvasRenderingContext2d = context.dyn_into()?;
    context.put_image_data(&image, 0, 0)?;
    let options = web_sys::ImageEncodeOptions::new();
    options.set_type("image/png");
    canvas.convert_to_blob_with_options(&options)
}

fn reject_with(reject: &Function, message: &str) {
    let _ = reject.call1(&JsValue::UNDEFINED, &js_sys::Error::new(message));
}

fn reject_js(reject: &Function, value: JsValue) {
    if value.is_instance_of::<js_sys::Error>() {
        let _ = reject.call1(&JsValue::UNDEFINED, &value);
        return;
    }
    let message = value
        .as_string()
        .unwrap_or_else(|| "screenshot failed".to_owned());
    reject_with(reject, &message);
}
