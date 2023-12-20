use elodin_conduit::{
    bevy::{Msg, SubscribeEvent},
    builder::Builder,
    parser::Parser,
    ComponentValue, SUB_COMPONENT_ID,
};
use tracing::{debug, error};
use wasm_bindgen::UnwrapThrowExt;
use web_sys::BinaryType;

#[allow(clippy::needless_pass_by_value)]
fn string_from_js_value(s: wasm_bindgen::JsValue) -> String {
    s.as_string().unwrap_or(format!("{:#?}", s))
}

pub(crate) fn spawn_wasm(url: String, bevy_tx: flume::Sender<Msg<'static>>) -> anyhow::Result<()> {
    // Based on https://rustwasm.github.io/wasm-bindgen/examples/websockets.html

    use wasm_bindgen::closure::Closure;
    use wasm_bindgen::JsCast as _;

    let ws = web_sys::WebSocket::new(&url)
        .map_err(string_from_js_value)
        .map_err(|err| anyhow::anyhow!("{}", err))?;

    // set this to use Arraybuffer as the data type.
    // The alternative, blob, has caused my entire computer to run out of file-descriptors, Dan's computer
    // actually blue-screened, and overall it is just terrible.
    ws.set_binary_type(BinaryType::Arraybuffer);

    let send_ws = ws.clone();
    let (out_tx, out_rx) = flume::unbounded();
    wasm_bindgen_futures::spawn_local(async move {
        let mut buf = vec![];
        while let Ok(Msg::Data(data)) = out_rx.recv_async().await {
            buf.clear();
            let mut builder = Builder::new(&mut buf, 0).unwrap_throw();
            builder.append_data(data).unwrap_throw();
            send_ws
                .send_with_u8_array(builder.into_buf())
                .unwrap_throw();
        }
    });

    {
        let bevy_tx = bevy_tx.clone();
        let onmessage_callback = Closure::wrap(Box::new(move |e: web_sys::MessageEvent| {
            if let Ok(abuf) = e.data().dyn_into::<js_sys::ArrayBuffer>() {
                let array = js_sys::Uint8Array::new(&abuf);
                let len = array.length() as usize;
                let mut buf = vec![0u8; len];
                array.copy_to(&mut buf);
                let Some(mut parser) = Parser::new(buf) else {
                    return;
                };
                if let Some(batch) = parser.parse_data_msg() {
                    let filters: Vec<_> = batch
                        .components
                        .iter()
                        .filter(|c| c.component_id == SUB_COMPONENT_ID)
                        .flat_map(|c| c.storage.iter())
                        .filter_map(|(_, c)| match c {
                            ComponentValue::Filter(filter) => Some(filter),
                            _ => None,
                        })
                        .cloned()
                        .collect();
                    if filters.is_empty() {
                        for data in batch.components.into_iter() {
                            if let Err(err) = bevy_tx.send(Msg::Data(data)) {
                                error!(?err, "error sending data");
                            }
                        }
                    } else if let Err(err) = bevy_tx.send(Msg::Subscribe(SubscribeEvent {
                        tx: out_tx.clone(),
                        filters,
                    })) {
                        error!(?err, "error sending data");
                    }
                }
            } else {
                debug!("Unknown websocket message received: {:?}", e.data());
            }
        }) as Box<dyn FnMut(web_sys::MessageEvent)>);

        // set message event handler on WebSocket
        ws.set_onmessage(Some(onmessage_callback.as_ref().unchecked_ref()));

        // forget the callback to keep it alive
        onmessage_callback.forget();
    }

    // {
    //     let tx = tx.clone();
    //     let onerror_callback = Closure::wrap(Box::new(move |error_event: web_sys::ErrorEvent| {
    //         error!(
    //             "error event: {}: {:?}",
    //             error_event.message(),
    //             error_event.error()
    //         );
    //         if let Err(err) = tx.send(Msg {
    //             msg_type: MsgType::Error,
    //             buf: error_event.message().as_bytes().to_vec(),
    //         }) {
    //             error!("error occured send {:?}", err);
    //         }
    //     }) as Box<dyn FnMut(web_sys::ErrorEvent)>);
    //     ws.set_onerror(Some(onerror_callback.as_ref().unchecked_ref()));
    //     onerror_callback.forget();
    // }

    // {
    //     let tx = tx.clone();
    //     let onopen_callback = Closure::wrap(Box::new(move |_| {
    //         if let Err(err) = tx.send(Msg {
    //             msg_type: MsgType::Open,
    //             buf: vec![],
    //         }) {
    //             error!("error occured send {:?}", err);
    //         }
    //     }) as Box<dyn FnMut(wasm_bindgen::JsValue)>);
    //     ws.set_onopen(Some(onopen_callback.as_ref().unchecked_ref()));
    //     onopen_callback.forget();
    // }

    // {
    //     let tx = tx.clone();
    //     let onclose_callback = Closure::wrap(Box::new(move |_| {
    //         if let Err(err) = tx.send(Msg {
    //             msg_type: MsgType::Close,
    //             buf: vec![],
    //         }) {
    //             error!("error occured send {:?}", err);
    //         }
    //     }) as Box<dyn FnMut(wasm_bindgen::JsValue)>);
    //     ws.set_onclose(Some(onclose_callback.as_ref().unchecked_ref()));
    //     onclose_callback.forget();
    // }

    Ok(())
}
