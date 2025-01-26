// If you want to use Phoenix channels, run `mix help phx.gen.channel`
// to get started and then uncomment the line below.
// import "./user_socket.js"

// You can include dependencies in two ways.
//
// The simplest option is to put them in assets/vendor and
// import them using relative paths:
//
//     import "../vendor/some-package.js"
//
// Alternatively, you can `npm install some-package --prefix assets` and import
// them using a path starting with the package name:
//
//     import "some-package"
//

// Include phoenix_html to handle method=PUT/DELETE in forms and buttons.
import "phoenix_html";
// Establish Phoenix Socket and LiveView configuration.
import { Socket } from "phoenix";
import { LiveSocket } from "phoenix_live_view";
import topbar from "../vendor/topbar";
import { CodeEditorHook } from "../../deps/live_monaco_editor/priv/static/live_monaco_editor.esm";

let Hooks = {};
Hooks.CodeEditorHook = CodeEditorHook;
Hooks.EditorWasmHook = {
  async mounted() {
    await waitForElm("#editor");
  },
};

Hooks.FireworksHook = {
  async mounted() {
    const container = document.querySelector("#fireworks");
    const fireworks = new Fireworks.Fireworks(container, {
      particles: 200,
      traceLength: 5,
      gravity: 2,
      mouse: {
        move: true,
      },
    });
    fireworks.start();
  },
};

// source: https://stackoverflow.com/a/61511955
function waitForElm(selector) {
  return new Promise((resolve) => {
    if (document.querySelector(selector)) {
      return resolve(document.querySelector(selector));
    }

    const observer = new MutationObserver((mutations) => {
      if (document.querySelector(selector)) {
        observer.disconnect();
        resolve(document.querySelector(selector));
      }
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true,
    });
  });
}

window.addEventListener("lme:editor_mounted", (ev) => {
  const hook = ev.detail.hook;
  window.addEventListener(
    "keydown",
    (event) => {
      if (event.keyCode === 13 && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        if (event.stopImmediatePropagation) {
          event.stopImmediatePropagation();
        } else {
          event.stopPropagation();
        }
        hook.pushEvent("update_code", {});
        return;
      }
    },
    true,
  );
});

let csrfToken = document
  .querySelector("meta[name='csrf-token']")
  .getAttribute("content");
let liveSocket = new LiveSocket("/live", Socket, {
  hooks: Hooks,
  params: { _csrf_token: csrfToken },
});

// Show progress bar on live navigation and form submits
topbar.config({ barColors: { 0: "#29d" }, shadowColor: "rgba(0, 0, 0, .3)" });
window.addEventListener("phx:page-loading-start", (_info) => topbar.show(300));
window.addEventListener("phx:page-loading-stop", (_info) => topbar.hide());

window.addEventListener("phx:copy", (event) => {
  navigator.clipboard.writeText(event.target.value).then(() => {});
});
window.addEventListener("phx:copy-inner", (event) => {
  navigator.clipboard.writeText(event.target.innerText).then(() => {});
});

// connect if there are any LiveViews on the page
liveSocket.connect();

liveSocket.getSocket().onClose((e) => {
  if (e.wasClean) {
    document.querySelector("#client-error").style.display = "none";
  }
});

// expose liveSocket on window for web console debug logs and latency simulation:
// >> liveSocket.enableDebug()
// >> liveSocket.enableLatencySim(1000)  // enabled for duration of browser session
// >> liveSocket.disableLatencySim()
window.liveSocket = liveSocket;
