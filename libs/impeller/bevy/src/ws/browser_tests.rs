use super::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen(inline_js = "
export function delay(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }
export function setCookie() { document.cookie = 'e2_session=test; path=/; SameSite=Lax'; }
")]
extern "C" {
    async fn delay(ms: u32);
    #[wasm_bindgen(js_name = setCookie)]
    fn set_cookie();
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
    install_connection(
        &mut world,
        Some(DbEndpoint::parse("127.0.0.1:18765").unwrap()),
        true,
    );
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
