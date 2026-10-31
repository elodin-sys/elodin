use super::*;
use crate::BrowserEndpoints;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen(inline_js = "
export function delay(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }
export async function request(path) {
  return await (await fetch('http://127.0.0.1:18766' + path, {credentials: 'include'})).text();
}
export function setCookie() { document.cookie = 'e2_session=test; path=/; SameSite=Lax'; }
")]
extern "C" {
    async fn delay(ms: u32);
    async fn request(path: &str) -> JsValue;
    #[wasm_bindgen(js_name = setCookie)]
    fn set_cookie();
}

fn endpoint(session: &str) -> DbEndpoint {
    BrowserEndpoints::new(
        &format!("ws://127.0.0.1:18766/{session}/main?route=one"),
        &format!("ws://127.0.0.1:18766/{session}/messages?route=two"),
        &format!("http://127.0.0.1:18766/{session}/assets?route=three"),
    )
    .unwrap()
    .into()
}

async fn wait_connected(world: &World) {
    for _ in 0..100 {
        if world.resource::<ThreadConnectionStatus>().status() == ConnectionStatus::Success {
            return;
        }
        delay(20).await;
    }
    panic!("Both browser WebSockets must connect");
}

/// Run with browser_server.mjs listening on 127.0.0.1:18766.
#[wasm_bindgen_test(async)]
async fn browser_routes_cookies_assets_reconnect_and_session_switch() {
    set_cookie();
    let mut world = World::new();
    let first = endpoint("first");
    install_connection(&mut world, Some(first.clone()), true);
    wait_connected(&world).await;
    assert_eq!(world.resource::<ConnectionAddr>().endpoint(), &first);
    assert!(first.socket_addr().is_none());
    let asset = first.asset_url("models/rocket%20one.glb").unwrap();
    let path = asset.strip_prefix("http://127.0.0.1:18766").unwrap();
    assert_eq!(
        request(path).await.as_string().unwrap(),
        "/first/assets/models/rocket%20one.glb?route=three"
    );

    // Losing either channel invalidates the pair and reconnects both channels.
    for channel in ["messages", "main"] {
        let stream = world.resource::<CurrentStreamId>().0;
        request(&format!("/control/close?channel={channel}")).await;
        delay(50).await;
        assert_eq!(
            world.resource::<ThreadConnectionStatus>().status(),
            ConnectionStatus::Error
        );
        ws_reconnect(&mut world);
        delay(300).await;
        ws_reconnect(&mut world);
        wait_connected(&world).await;
        assert_ne!(world.resource::<CurrentStreamId>().0, stream);
    }

    let stream = world.resource::<CurrentStreamId>().0;
    let second = endpoint("second");
    // Exercise the same queue/apply systems used by the public WsConnect message.
    world.init_resource::<Messages<WsConnect>>();
    world.write_message(WsConnect(second.clone()));
    use bevy::ecs::system::RunSystemOnce;
    world.run_system_once(queue_ws_connect).unwrap();
    apply_ws_connect(&mut world);
    wait_connected(&world).await;
    assert_eq!(world.resource::<ConnectionAddr>().endpoint(), &second);
    assert_ne!(world.resource::<CurrentStreamId>().0, stream);
    let requests = request("/control/requests").await.as_string().unwrap();
    for path in [
        "/first/main?route=one",
        "/first/messages?route=two",
        "/second/main?route=one",
        "/second/messages?route=two",
    ] {
        assert!(
            requests.contains(path),
            "missing request {path}: {requests}"
        );
    }
    assert!(requests.contains("e2_session=test"));
    suspend_ws_reconnect(&mut world);
}
