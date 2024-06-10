use bytes::Buf;
use conduit::{
    client::{Demux, MsgPair},
    Packet, Payload,
};
use tracing::{debug, error};
use wasm_bindgen::prelude::*;
use web_sys::{BinaryType, ErrorEvent};

#[allow(clippy::needless_pass_by_value)]
fn string_from_js_value(s: wasm_bindgen::JsValue) -> String {
    s.as_string().unwrap_or(format!("{:#?}", s))
}

macro_rules! console_error {
    ($($t:tt)*) => (web_sys::console::error_1(&format_args!($($t)*).to_string().into()))
}

pub(crate) fn spawn_wasm(url: String, bevy_tx: flume::Sender<MsgPair>) -> anyhow::Result<()> {
    use wasm_bindgen::JsCast as _;

    let ws = web_sys::WebSocket::new(&url)
        .map_err(string_from_js_value)
        .map_err(|err| anyhow::anyhow!("{}", err))?;

    // set this to use Arraybuffer as the data type.
    // The alternative, blob, has caused my entire computer to run out of file-descriptors, Dan's computer
    // actually blue-screened, and overall it is just terrible.
    ws.set_binary_type(BinaryType::Arraybuffer);

    let (ws_ready_tx, ws_ready_rx) = flume::bounded::<()>(0);
    let (out_tx, out_rx) = flume::unbounded::<Packet<Payload<bytes::Bytes>>>();

    let bevy_tx = bevy_tx.clone();
    let mut demux = Demux::default();
    let onmessage_callback = Closure::<dyn FnMut(_)>::new(move |e: web_sys::MessageEvent| {
        if let Ok(abuf) = e.data().dyn_into::<js_sys::ArrayBuffer>() {
            let array = js_sys::Uint8Array::new(&abuf);
            let len = array.length() as usize;
            let mut buf = bytes::BytesMut::zeroed(len);
            array.copy_to(&mut buf);
            let buf = buf.freeze();
            let packet = match Packet::parse(buf) {
                Ok(p) => p,
                Err(err) => {
                    error!(?err, "error parsing packet");
                    return;
                }
            };
            let Ok(msg) = demux.handle(packet) else {
                return;
            };
            bevy_tx
                .send(MsgPair {
                    msg,
                    tx: Some(out_tx.downgrade()),
                })
                .unwrap();
        } else {
            debug!("Unknown websocket message received: {:?}", e.data());
        }
    });
    // set message event handler on WebSocket
    ws.set_onmessage(Some(onmessage_callback.as_ref().unchecked_ref()));
    // forget the callback to keep it alive
    onmessage_callback.forget();

    let onerror_callback = Closure::<dyn FnMut(_)>::new(move |e: ErrorEvent| {
        console_error!("error event: {:?}", e);
    });
    ws.set_onerror(Some(onerror_callback.as_ref().unchecked_ref()));
    onerror_callback.forget();

    let onopen_callback = Closure::<dyn FnMut()>::new(move || {
        ws_ready_tx.send(()).unwrap();
    });
    ws.set_onopen(Some(onopen_callback.as_ref().unchecked_ref()));
    onopen_callback.forget();

    let send_ws = ws.clone();
    wasm_bindgen_futures::spawn_local(async move {
        ws_ready_rx.recv_async().await.unwrap();
        while let Ok(msg) = out_rx.recv_async().await {
            let mut buf = bytes::BytesMut::new();
            msg.write(&mut buf).unwrap_throw();
            send_ws.send_with_u8_array(buf.chunk()).unwrap()
        }
    });

    Ok(())
}
