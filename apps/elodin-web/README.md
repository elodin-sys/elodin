# Browser connections

This branch, `feat/wasm-parascom`, is experimental and will not merge as it stands.
Its changes require dedicated Elodin PRs with separate review and tests.

The editor accepts complete browser routes through three query parameters: `impeller`, `messages`, and `assets`.
Supply all three parameters together. Encode each parameter value with `URLSearchParams`.

```js
const query = new URLSearchParams({
  impeller: "wss://example.com/sessions/123/impeller?route=one",
  messages: "wss://example.com/sessions/123/impeller/msg?route=two",
  assets: "https://example.com/sessions/123/assets?route=three",
});
frame.src = `/editor/?${query}`;
```

Each endpoint must be an absolute URL without credentials or a fragment.
The asset base can include a path and routing parameters.
The editor preserves those parameters for DB assets and their relative dependencies, such as glTF textures.
Use percent escapes for encoded asset names. The editor also encodes spaces, query characters, and fragment characters in DB keys.

The browser supplies cookies for these requests. The host controls authentication and server permissions.
Elodin does not add authorization headers.

Browser routes select a read-only viewer by default.
This viewer blocks recording changes and uploads of schematics or layouts.
It also disables SQL panels and exact SQL statistics in Data Overview.
Typed telemetry and sparkline queries remain available.
Camera controls, local layout edits, and timeline navigation remain available.
Use `read_only=false` only when the server supports the full editor interface.

The standalone `?db=host:port` interface remains available.
It derives HTTP and WebSocket addresses from the database port and retains the full editor interface.
Do not combine `db` with browser routes.
Invalid, duplicate, or incomplete parameters produce `elodin-error` and stop startup.

## Rust interface

```rust,ignore
let endpoints = elodin_editor::BrowserEndpoints::new(main_ws, messages_ws, assets_http)?;
let plugin = elodin_editor::EditorPlugin::default()
    .with_browser_endpoints(endpoints)
    .with_viewer_capabilities(elodin_editor::ViewerCapabilities::READ_ONLY);
```

`BrowserEndpoints` validates the supplied routes. `DbEndpoint` identifies the connection with the complete routes, including their paths and routing parameters.
WASM callers can send `impeller_bevy::WsConnect(endpoints.into())` to replace the connection.
This operation replaces both channels and clears their packet queues.

## Lifecycle events

The editor emits `ViewerEvent` messages in Rust and custom events on its browser window.
The host can listen on the frame window when both pages share an origin.

`elodin-connecting` marks a connection or recording change.
`elodin-ready` follows the database handshake, schematic load, and initial telemetry backfill.
If the recording contains no requested telemetry, the editor does not wait for a backfill.
The event can repeat after a reconnect or recording change.
`elodin-error` reports connection, schematic, and asset failures. Its `detail` field contains an error string.

The host owns its loading screen and event handlers.
The read-only controls govern editor actions. The server must enforce its own access rules.

## Experimental schematic windows

Browser schematics show additional windows as named tabs inside the existing canvas.
WebGL cannot share the first canvas context with another window surface.
The tabs share the root scene and timeline. Desktop screen placement does not apply.

Window paths accept `db:` keys and paths relative to the containing schematic.
The resolver loads external, nested, and inline window layouts before it reports readiness.
Circular references, missing files, invalid KDL, excessive nesting, and excessive window counts produce load errors.
Paths must stay within the recording. Separate HTTP origins are outside this experiment.
The desktop loader retains its native windows.
The browser palette omits desktop window creation and screen arrangement commands.

This experiment covers schematic windows in the browser viewer.
The dedicated Elodin PR must also review full-editor window creation, tab navigation, scene sharing, and reconnect behavior.

## Tests

Run native tests from the repository root inside `nix develop`:

```sh
cargo test -p impeller-bevy --lib
cargo test -p elodin-editor --lib
cargo test -p elodin-web
cargo check -p elodin-web --target wasm32-unknown-unknown
```

The browser test needs Node.js, `wasm-bindgen-test-runner`, and Chrome with `chromedriver` on `PATH`.
Use a runner version that matches `wasm-bindgen` in `Cargo.lock`.
Start the fixture in a separate Nix shell from the repository root:

```sh
node libs/impeller/bevy/tests/browser_server.mjs
```

Run the browser test from the repository root:

```sh
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
  cargo test -p impeller-bevy --no-default-features --target wasm32-unknown-unknown --lib
```

The test covers custom routes, cookies, encoded asset names, reconnects after either channel closes, and session replacement on the same host.
