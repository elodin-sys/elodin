defmodule ParacosmDashboardWeb.EditorComponents do
  use Phoenix.Component

  import ParacosmDashboardWeb.CoreComponents
  import ParacosmDashboardWeb.IconComponents

  def console(assigns) do
    ~H"""
    <div class="w-full text-white overflow-auto h-[296px]">
      <div class="h-[64px] p-2 shadow-lg bg-secondary-surface flex items-center">
        <.button class="px-elo-xl">Update Sim</.button>
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
      class="bg-secondary-surface h-full w-1/2 flex items-center justify-center"
      data-ws-url={@url}
      phx-update="ignore"
    >
      <.spinner id="editor-spinner" class="animate-spin w-16 h-16" />
      <canvas id="editor" style="display: none" />
    </div>
    """
  end
end
