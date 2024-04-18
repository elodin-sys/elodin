defmodule ElodinDashboardWeb.SandboxComponents do
  use Phoenix.Component
  import ElodinDashboardWeb.CoreComponents
  alias Phoenix.LiveView.JS

  attr(:path, :string, default: nil)
  attr(:href, :string, default: nil)
  attr(:name, :string, default: nil)
  attr(:img, :string, default: nil)
  attr(:phx_click, :string, default: nil)
  attr(:id, :string, default: nil)
  attr(:delete_button, :boolean, default: false)
  attr(:delete_path, :string, default: nil)

  def sandbox_card(assigns) do
    ~H"""
    <div
      id={@id}
      class="w-52 h-40 p-1 flex flex-col gap-1 rounded-elo-xs text-sm font-bold bg-center bg-cover group"
      style={"background-image: url('#{@img}');"}
    >
      <.link patch={@path} href={@href} phx-click={@phx_click} class="flex-grow"></.link>

      <div class="flex items-center justify-center bg-primary-smoke rounded-elo-xs py-2 overflow-hidden">
        <.link patch={@path} href={@href} phx-click={@phx_click} class="flex-grow text-center">
          <%= @name %>
        </.link>

        <%= if @delete_button do %>
          <button
            phx-click="delete"
            phx-value-id={@id}
            class="w-0 border-l-[1px] border-primary-onyx-9 transition-all ease-in-out duration-300 group-hover:w-12"
          >
            x
          </button>
        <% end %>
      </div>
    </div>
    """
  end
end
