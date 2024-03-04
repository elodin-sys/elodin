defmodule ElodinDashboardWeb.ModalComponents do
  use Phoenix.Component
  alias Phoenix.LiveView.JS
  import ElodinDashboardWeb.CoreComponents

  @doc """
  Renders a guide modal.

  ## Examples

      <.guide_modal id="guide-modal">
        This is a modal.
      </.guide_modal>

  JS commands may be passed to the `:on_cancel` to configure
  the closing/cancel event, for example:

      <.guide_modal id="guide" on_cancel={JS.navigate(~p"/posts")}>
        This is another modal.
      </.guide_modal>

  """
  attr(:id, :string, required: true)
  attr(:show, :boolean, default: false)
  attr(:bg_color, :string, default: "black-primary")
  attr(:step_count, :integer, default: 4)
  attr(:cur_step, :integer, default: 1)
  attr(:prev_page, JS, default: %JS{})
  attr(:next_page, JS, default: %JS{})
  attr(:on_cancel, JS, default: %JS{})
  slot(:inner_block, required: true)

  def guide_modal(assigns) do
    ~H"""
    <.modal
      id={@id}
      wrapper_class=""
      bg_color={@bg_color}
      container_padding="0"
      show={@show}
      on_cancel={@on_cancel}
    >
      <div class="h-[560px] w-[820px] relative flex flex-col items-stretch">
        <div class="basis-5/6 flex flex-col items-center">
          <%= render_slot(@inner_block) %>
        </div>

        <div class="basis-1/6 flex items-center justify-between bg-black-primary">
          <div class="px-6">
            <.button
              :if={@cur_step > 1 && @cur_step <= @step_count}
              class="px-6 py-4"
              type="secondary"
              phx-click={@prev_page}
            >
              Previous
            </.button>
          </div>
          <div class="gap-2 flex">
            <div
              :for={step <- 1..@step_count}
              class={"w-[7px] h-[7px] bg-white rounded-full#{if step != @cur_step, do: " opacity-40", else: ""}"}
            >
            </div>
          </div>
          <div class="px-6">
            <.button
              :if={@cur_step > 1 && @cur_step <= @step_count}
              class="px-6 py-4"
              phx-click={if @cur_step == @step_count, do: @on_cancel, else: @next_page}
            >
              <%= if @cur_step == @step_count do %>
                GET STARTED
              <% else %>
                CONTINUE
              <% end %>
            </.button>
          </div>
        </div>
      </div>
    </.modal>
    """
  end
end
