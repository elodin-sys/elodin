use anyhow::anyhow;
use bevy::prelude::App;
use ewebsock::{WsEvent, WsMessage, WsReceiver, WsSender};
use paracosm::sync::ClientTransport;
use paracosm_editor::EditorPlugin;
use std::{cell::RefCell, rc::Rc};
use tracing::error;

fn main() -> anyhow::Result<()> {
    let window = web_sys::window().ok_or_else(|| anyhow!("window missing"))?;
    let document = window
        .document()
        .ok_or_else(|| anyhow!("document missing"))?;
    let container = document
        .get_element_by_id("editor-container")
        .ok_or_else(|| anyhow!("missing editor container div"))?;
    let url = container
        .get_attribute("data-ws-url")
        .ok_or_else(|| anyhow!("data-ws-url required"))?;
    let transport = Transport::connect(&url)?;
    let code_transport = transport.clone();
    container
        .add_event_listener_with_callback("code-update", a.as_ref().unchecked_ref())
        .map_err(|_| anyhow!("failed to add event listener"))?;
    let mut app = App::new();
    app.add_plugins(EditorPlugin::<Transport>::new())
        .insert_non_send_resource(transport.clone());
    app.run();
    Ok(())
}

#[derive(Clone)]
struct Transport {
    tx: Rc<RefCell<WsSender>>,
    rx: Rc<WsReceiver>,
}

impl Transport {
    fn connect(url: &str) -> anyhow::Result<Self> {
        let (tx, rx) = ewebsock::connect(url).map_err(|err| anyhow!("websocket err: {}", err))?;
        let tx = Rc::new(RefCell::new(tx));
        let rx = Rc::new(rx);
        Ok(Transport { tx, rx })
    }
}

impl ClientTransport for Transport {
    fn try_recv_msg(&self) -> Option<paracosm::ClientMsg> {
        let event = self.rx.try_recv()?;
        let WsEvent::Message(WsMessage::Binary(buf)) = event else {
            return None;
        };
        let Ok(msg) = postcard::from_bytes(&buf) else {
            error!("error deserializing buf");
            return None;
        };
        Some(msg)
    }

    fn send_msg(&self, msg: paracosm::ServerMsg) {
        let Ok(buf) = postcard::to_allocvec(&msg) else {
            error!("error serializing msg");
            return;
        };
        let mut tx = self.tx.borrow_mut();
        tx.send(WsMessage::Binary(buf));
    }
}
