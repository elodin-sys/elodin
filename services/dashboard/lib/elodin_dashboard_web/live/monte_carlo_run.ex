defmodule ElodinDashboardWeb.MonteCarloRunLive do
  use ElodinDashboardWeb, :live_view
  alias Elodin.Types.Api
  alias ElodinDashboard.Atc
  import ElodinDashboardWeb.CoreComponents
  import ElodinDashboardWeb.NavbarComponents
  import ElodinDashboardWeb.IconComponents

  def mount(params, _, socket) do
    grid =
      1..360
      |> Enum.map(fn i ->
        Map.merge(
          %{sample: i},
          case i do
            20 -> %{type: "fail", progress: 100}
            31 -> %{type: "fail", progress: 100}
            67 -> %{type: "fail", progress: 100}
            x when x in 0..110 -> %{type: "pass", progress: 100}
            x when x in 100..150 -> %{type: "active", progress: i - 100}
            _ -> %{type: "pending", progress: 0}
          end
        )
      end)

    {:ok, socket |> assign(grid: grid)}
  end

  def status_color(type) do
    case(type) do
      "pass" -> "bg-green"
      "fail" -> "bg-red"
      "active" -> "bg-yellow"
      "pending" -> "bg-tan"
    end
  end

  def grid_box(%{type: "pending"} = assigns) do
    ~H"""
    <div
      style="width: 24px; height: 24px;"
      class={["rounded-elo-xs border-tan border-opacity-60 border"]}
    />
    """
  end

  def grid_box(%{type: type} = assigns) do
    color = status_color(type)

    assigns = assign(assigns, :color, color)

    ~H"""
    <div style="width: 24px; height: 24px;" class={["rounded-elo-xs", @color]} />
    """
  end

  def status_label(%{type: type} = assigns) do
    color = status_color(type)

    assigns = assigns |> assign(:color, color) |> assign(:type, String.upcase(type))

    ~H"""
    <div class="flex items-center gap-2">
      <div style="width: 8px; height: 8px;" class={["rounded-elo-xxs", @color]} />
      <span class="text-crema font-normal"><%= @type %></span>
    </div>
    """
  end

  def smolgress(%{type: type} = assigns) do
    color = status_color(type)

    assigns = assigns |> assign(:color, color)

    ~H"""
    <div class="flex items-center">
      <div style="width: 16px; height: 8px;" class="flex mr-elo-m">
        <div style={"width: #{@value / 100.0 * 16}px; height: 8px;"} class={@color} />
        <div style={"width: #{(1 - @value / 100.0) * 16}px; height: 8px;"} class="bg-[#FFFBF0]" />
      </div>
      <span class="text-crema"><%= @value %>%</span>
    </div>
    """
  end

  def render(assigns) do
    ~H"""
    <.navbar_layout current_user={@current_user}>
      <div class="flex flex-col min-h-full p-6 bg-black-primary">
        <div class="flex w-full gap-elo-xl flex-col elo-grid-md:flex-row">
          <div class="flex min-w-[22rem] bg-black-secondary rounded-elo-xs border border-white border-opacity-10 flex-col">
            <div class="flex w-full p-6 flex-col">
              <div class="text-crema pb-1">
                Lorem Ipsum Run
              </div>
              <div class="text-crema-60">
                Project
              </div>
              <.divider />
              <.horizontal_label label="STARTED" value="2024-03-08 15:00:00" />
              <.horizontal_label label="EST TIME REMAINING" value="3:00:00" class="mb-0" />
              <.divider />
              <.label_progress_bar value={40.0} label="PASS" />
              <.label_progress_bar value={10.0} label="FAIL" color="bg-red" />
              <.label_progress_bar value={20.0} label="ACTIVE" color="bg-yellow" />
              <.label_progress_bar value={20.0} label="PENDING" color="bg-tan" />
              <.button type="danger">
                <.stop />
                <span class="leading-3">STOP</span>
              </.button>
            </div>
          </div>
          <div class="flex h-full w-[360px] sm:w-[500px] md:w-[682px] bg-black-secondary rounded-elo-xs border border-white border-opacity-10 flex-col items-center self-center">
            <div class="grid grid-cols-10 sm:grid-cols-16 md:grid-cols-20 gap-2 text-white p-6">
              <%= for i <- @grid do %>
                <.grid_box type={i.type} />
              <% end %>
            </div>
          </div>
        </div>
        <div class="flex w-full max-h-[500px] mt-elo-xl bg-black-secondary rounded-elo-xs border border-white border-opacity-10 overflow-scroll text-mono">
          <table class="w-full text-crema text-medium text-left">
            <thead class="sticky top-0">
              <tr class="h-[51px] bg-black-header text-crema text-mono text-crema-60 text-sm">
                <th class="pl-elo-xl">SAMPLE</th>
                <th>STATE</th>
                <th>PROGRESS</th>
              </tr>
            </thead>
            <tbody>
              <%= for i <- @grid do %>
                <tr class="h-[51px] ">
                  <td class="pl-elo-xl"><%= i.sample %></td>
                  <td><.status_label type={i.type} /></td>
                  <td><.smolgress value={i.progress} type={i.type} /></td>
                </tr>
              <% end %>
            </tbody>
          </table>
        </div>
      </div>
    </.navbar_layout>
    """
  end
end
