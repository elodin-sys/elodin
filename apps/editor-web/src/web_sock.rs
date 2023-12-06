// based on https://github.com/rerun-io/ewebsock
// simplified and spedup using thingbuf

use thingbuf::mpsc::blocking::{Receiver, Sender};
use tracing::{debug, error, warn};
use web_sys::BinaryType;

#[allow(clippy::needless_pass_by_value)]
fn string_from_js_value(s: wasm_bindgen::JsValue) -> String {
    s.as_string().unwrap_or(format!("{:#?}", s))
}

pub fn connect(url: impl Into<String>) -> anyhow::Result<(WsSender, Receiver<Msg, MsgRecycle>)> {
    let (tx, rx) = thingbuf::mpsc::blocking::with_recycle(256, MsgRecycle);
    //let (ws_receiver, on_event) = WsReceiver::new();
    let ws_sender = ws_connect_impl(url.into(), tx)?;
    Ok((ws_sender, rx))
}

pub type WsReceiver = Receiver<Msg, MsgRecycle>;

/// This is how you send messages to the server.
///
/// When this is dropped, the connection is closed.
pub struct WsSender {
    ws: Option<web_sys::WebSocket>,
}

impl Drop for WsSender {
    fn drop(&mut self) {
        if let Err(err) = self.close() {
            warn!("Failed to close WebSocket: {err:?}");
        }
    }
}

impl WsSender {
    /// Send the message to the server.
    pub fn send(&mut self, msg: Vec<u8>) {
        if let Some(ws) = &mut self.ws {
            let result = ws.send_with_u8_array(&msg);
            if let Err(err) = result.map_err(string_from_js_value) {
                error!("Failed to send: {:?}", err);
            }
        }
    }

    /// Close the conenction.
    ///
    /// This is called automatically when the sender is dropped.
    pub fn close(&mut self) -> anyhow::Result<()> {
        if let Some(ws) = self.ws.take() {
            debug!("Closing WebSocket");
            ws.close()
                .map_err(string_from_js_value)
                .map_err(|err| anyhow::anyhow!("{}", err))
        } else {
            Ok(())
        }
    }
}

#[derive(Default)]
pub struct Msg {
    pub msg_type: MsgType,
    pub buf: Vec<u8>,
}

pub struct MsgRecycle;

impl thingbuf::recycling::Recycle<Msg> for MsgRecycle {
    fn new_element(&self) -> Msg {
        Msg {
            msg_type: MsgType::None,
            buf: Vec::with_capacity(256),
        }
    }

    fn recycle(&self, element: &mut Msg) {
        element.buf.clear();
        element.msg_type = MsgType::None;
    }
}

#[derive(Default, PartialEq, Eq)]
pub enum MsgType {
    #[default]
    None,
    Buf,
    Open,
    Close,
    Error,
}

pub(crate) fn ws_connect_impl(
    url: String,
    tx: Sender<Msg, MsgRecycle>,
) -> anyhow::Result<WsSender> {
    // Based on https://rustwasm.github.io/wasm-bindgen/examples/websockets.html

    use wasm_bindgen::closure::Closure;
    use wasm_bindgen::JsCast as _;

    // Connect to an server
    let ws = web_sys::WebSocket::new(&url)
        .map_err(string_from_js_value)
        .map_err(|err| anyhow::anyhow!("{}", err))?;

    // For small binary messages, like CBOR, Arraybuffer is more efficient than Blob handling
    ws.set_binary_type(BinaryType::Arraybuffer);

    // onmessage callback
    {
        let tx = tx.clone();
        let onmessage_callback = Closure::wrap(Box::new(move |e: web_sys::MessageEvent| {
            // Handle difference Text/Binary,...
            if let Ok(abuf) = e.data().dyn_into::<js_sys::ArrayBuffer>() {
                let array = js_sys::Uint8Array::new(&abuf);
                let mut send_ref = tx.send_ref().unwrap();
                let len = array.length() as usize;
                send_ref.buf.resize(len, 0);
                send_ref.msg_type = MsgType::Buf;
                array.copy_to(&mut send_ref.buf[..len]);
                drop(array);
                drop(abuf);
            } else {
                debug!("Unknown websocket message received: {:?}", e.data());
            }
        }) as Box<dyn FnMut(web_sys::MessageEvent)>);

        // set message event handler on WebSocket
        ws.set_onmessage(Some(onmessage_callback.as_ref().unchecked_ref()));

        // forget the callback to keep it alive
        onmessage_callback.forget();
    }

    {
        let tx = tx.clone();
        let onerror_callback = Closure::wrap(Box::new(move |error_event: web_sys::ErrorEvent| {
            error!(
                "error event: {}: {:?}",
                error_event.message(),
                error_event.error()
            );
            if let Err(err) = tx.send(Msg {
                msg_type: MsgType::Error,
                buf: error_event.message().as_bytes().to_vec(),
            }) {
                error!("error occured send {:?}", err);
            }
        }) as Box<dyn FnMut(web_sys::ErrorEvent)>);
        ws.set_onerror(Some(onerror_callback.as_ref().unchecked_ref()));
        onerror_callback.forget();
    }

    {
        let tx = tx.clone();
        let onopen_callback = Closure::wrap(Box::new(move |_| {
            if let Err(err) = tx.send(Msg {
                msg_type: MsgType::Open,
                buf: vec![],
            }) {
                error!("error occured send {:?}", err);
            }
        }) as Box<dyn FnMut(wasm_bindgen::JsValue)>);
        ws.set_onopen(Some(onopen_callback.as_ref().unchecked_ref()));
        onopen_callback.forget();
    }

    {
        let tx = tx.clone();
        let onclose_callback = Closure::wrap(Box::new(move |_| {
            if let Err(err) = tx.send(Msg {
                msg_type: MsgType::Close,
                buf: vec![],
            }) {
                error!("error occured send {:?}", err);
            }
        }) as Box<dyn FnMut(wasm_bindgen::JsValue)>);
        ws.set_onclose(Some(onclose_callback.as_ref().unchecked_ref()));
        onclose_callback.forget();
    }

    Ok(WsSender { ws: Some(ws) })
}
