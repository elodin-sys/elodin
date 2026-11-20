//! `globalThis.elodin.captureScreenshot()` resolves to a PNG `Blob` of the editor window.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use bevy::prelude::*;
use bevy::render::view::window::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::window::PrimaryWindow;
use js_sys::{Function, Promise};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

pub struct WebScreenshotPlugin;

impl Plugin for WebScreenshotPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, install)
            .add_systems(Update, dispatch);
    }
}

struct Bridge {
    queue: Rc<RefCell<VecDeque<(Function, Function)>>>,
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
) {
    if windows.is_empty() {
        return;
    }
    // Bevy drops a second screenshot of the same window spawned in one frame.
    let Some((resolve, reject)) = bridge.queue.borrow_mut().pop_front() else {
        return;
    };
    commands.spawn(Screenshot::primary_window()).observe(
        move |captured: On<ScreenshotCaptured>| {
            deliver(&captured.image, resolve.clone(), reject.clone());
        },
    );
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
