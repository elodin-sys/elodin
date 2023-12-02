defmodule ParacosmDashboardWeb.SandboxComponents do
  use Phoenix.Component

  attr(:path, :string, default: nil)
  attr(:name, :string, default: nil)
  attr(:img, :string, default: nil)
  attr(:phx_click, :string, default: nil)

  def sandbox_card(assigns) do
    ~H"""
    <div
      style={"background: url(\"#{@img}\");"}
      class="w-[200px] h-[158px] bg-[url({@img})] bg-[100%_100%] h-[158px] rounded-elo-xs overflow-hidden"
    >
      <.link
        patch={@path}
        class="flex w-[198px] h-[41px] items-center justify-center pt-elo-lg pb-elo-lg relative top-[116px] left-px bg-tokens-surface-secondary rounded-elo-xs overflow-hidden"
        phx-click={@phx_click}
      >
        <div class="w-fit font-bold text-primative-colors-white-opacity-900 text-sm text-center">
          <%= @name %>
        </div>
      </.link>
    </div>
    """
  end
end
