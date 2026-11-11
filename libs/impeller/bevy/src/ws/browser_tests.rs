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

#[wasm_bindgen(inline_js = "
export function replayHost(mode) {
  globalThis.replayMode = mode;
  globalThis.replayStores = 0;
  globalThis.elodinReplayCache = {
    async lookup(scope, request) {
      if (request[7] !== 0) throw Error('noncanonical request');
      if (globalThis.replayMode === 'reject') throw Error('private mode');
      if (globalThis.replayMode === 'timeout') return new Promise(() => {});
      if (globalThis.replayMode === 'miss') return undefined;
      return request.slice();
    },
    store(scope, request, reply) {
      if (request[7] !== 0 || reply[7] !== 0) throw Error('noncanonical store');
      globalThis.replayStores++;
    }
  };
}
export function storedReplies() { return globalThis.replayStores; }
")]
extern "C" {
    #[wasm_bindgen(js_name = replayHost)]
    fn replay_host(mode: &str);
    #[wasm_bindgen(js_name = storedReplies)]
    fn stored_replies() -> u32;
}

#[wasm_bindgen_test(async)]
async fn replay_patches_ids_falls_back_and_discards_old_generations() {
    set_cookie();
    replay_host("hit");
    let mut world = World::new();
    install_connection(&mut world, Some(endpoint("replay")), true);
    wait_connected(&world).await;
    while world
        .resource_mut::<crate::PacketRx>()
        .try_recv_pkt()
        .is_some()
    {}
    let (socket, hook) = {
        let sockets = world.get_non_send::<WsSockets>().unwrap();
        let live = sockets.main.as_ref().unwrap();
        (live.ws.clone(), live.replay.clone())
    };
    let frame = LenPacket::new(PacketTy::Msg, impeller_wkt::GetMsgs::ID, 0)
        .with_request_id(42)
        .inner;
    hook.state.borrow_mut().enabled = true;
    hook.send(&socket, frame.clone());
    delay(50).await;
    assert_eq!(
        world
            .resource_mut::<crate::PacketRx>()
            .try_recv_pkt()
            .unwrap()
            .req_id(),
        42
    );
    assert!(hook.state.borrow().pending.is_empty());

    for mode in ["miss", "reject", "timeout"] {
        replay_host(mode);
        hook.send(&socket, frame.clone());
        delay(if mode == "timeout" { 2100 } else { 50 }).await;
        assert!(hook.state.borrow().pending.contains_key(&42));
        hook.receive(&socket, &frame);
        assert_eq!(stored_replies(), 1);
        assert!(hook.state.borrow().pending.is_empty());
    }
    replay_host("hit");
    hook.send(&socket, frame);
    hook.generation.fetch_add(1, Ordering::SeqCst);
    delay(50).await;
    assert!(
        world
            .resource_mut::<crate::PacketRx>()
            .try_recv_pkt()
            .is_none()
    );
    suspend_ws_reconnect(&mut world);
    let _ = js_sys::Reflect::delete_property(
        &js_sys::global(),
        &JsValue::from_str("elodinReplayCache"),
    );
}
