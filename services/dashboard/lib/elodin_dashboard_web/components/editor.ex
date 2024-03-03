defmodule ElodinDashboardWeb.EditorComponents do
  use Phoenix.Component

  import ElodinDashboardWeb.IconComponents

  def console(assigns) do
    ~H"""
    <pre class={"#{if @hide, do: "h-0 hidden", else: "h-64"} whitespace-pre-wrap overflow-auto text-white bg-dark-matte p-2"}>
    <%= @logs %>
    </pre>
    """
  end

  def editor_wasm(assigns) do
    ~H"""
    <div
      phx-hook="EditorWasmHook"
      id="editor-container"
      class="bg-black-secondary h-full w-1/2 flex items-center justify-center"
      data-ws-url={@url}
      phx-update="ignore"
    >
      <.spinner id="editor-spinner" class="animate-spin w-16 h-16" />
      <canvas
        id="editor"
        style="display: none; width: 100%; height: 100%;"
        oncontextmenu="return false;"
      />
    </div>
    """
  end
end
