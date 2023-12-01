defmodule ParacosmDashboardWeb.EditorComponents do
  use Phoenix.Component

  alias Phoenix.LiveView.JS
  import ParacosmDashboardWeb.Gettext
  import ParacosmDashboardWeb.CoreComponents

  def console(assigns) do
    ~H"""
    <div class="w-full text-white overflow-auto h-[296px]">
      <div class="h-10 p-2 shadow-lg bg-orange flex items-center">
        <.icon name="hero-play-solid" class="h-6 w-6 bg-white" />
      </div>
      <pre class="whitespace-pre-wrap overflow-auto h-64 bg-dark-matte p-2">
      <%= @logs %>
      </pre>
    </div>
    """
  end

  def editor_wasm(assigns) do
    ~H"""
    <div
      phx-hook="EditorWasmHook"
      id="editor-container"
      class="bg-black h-full w-1/2"
      data-ws-url={@url}
      phx-update="ignore"
    >
      <canvas id="editor" />
    </div>
    """
  end
end
