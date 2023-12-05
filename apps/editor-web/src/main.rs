use anyhow::anyhow;
use bevy::prelude::{App, In, IntoSystem, PostStartup};
use ewebsock::{WsEvent, WsMessage, WsReceiver, WsSender};
use paracosm::sync::ClientTransport;
use paracosm_editor::EditorPlugin;
use std::{cell::RefCell, rc::Rc};
use tracing::error;

fn main() -> anyhow::Result<()> {
    let url = get_url()?;
    let transport = Transport::connect(&url)?;
    let mut app = App::new();
    app.add_plugins(EditorPlugin::<Transport>::new())
        .add_systems(PostStartup, hide_loader.pipe(handle_error))
        .insert_non_send_resource(transport.clone());
    app.run();
    Ok(())
}

fn hide_loader() -> anyhow::Result<()> {
    let window = web_sys::window().ok_or_else(|| anyhow!("window missing"))?;
    let document = window
        .document()
        .ok_or_else(|| anyhow!("document missing"))?;
    let spinner = document
        .get_element_by_id("editor-spinner")
        .ok_or_else(|| anyhow!("missing editor spinner div"))?;
    let canvas = document
        .get_element_by_id("editor")
        .ok_or_else(|| anyhow!("missing editor canvas div"))?;
    spinner
        .set_attribute("style", "display: none;")
        .map_err(|e| anyhow!("set attr err {:?}", e))?;
    canvas
        .set_attribute("style", "display: block;")
        .map_err(|e| anyhow!("set attr err {:?}", e))?;
    Ok(())
}

fn handle_error(In(result): In<anyhow::Result<()>>) {
    if let Err(err) = result {
        error!(?err, "anyhow error")
    }
}

#[cfg(not(target_family = "wasm"))]
fn get_url() -> anyhow::Result<String> {
    use std::env;
    let args: Vec<String> = env::args().collect();
    Ok(args.into_iter().next().unwrap())
}

#[cfg(target_family = "wasm")]
fn get_url() -> anyhow::Result<String> {
    let window = web_sys::window().ok_or_else(|| anyhow!("window missing"))?;
    let document = window
        .document()
        .ok_or_else(|| anyhow!("document missing"))?;
    let container = document
        .get_element_by_id("editor-container")
        .ok_or_else(|| anyhow!("missing editor container div"))?;
    let url = container
        .get_attribute("data-ws-url")
        .ok_or_else(|| anyhow!("data-ws-url required"))?
        .to_string();
    Ok(url)
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
