defmodule ElodinDashboardWeb.MonteCarloRunLive do
  use ElodinDashboardWeb, :live_view
  alias Elodin.Types.Api
  alias ElodinDashboard.Atc
  import ElodinDashboardWeb.CoreComponents
  import ElodinDashboardWeb.NavbarComponents
  import ElodinDashboardWeb.IconComponents

  def mount(%{"project" => _project, "run" => id}, _, socket) do
    token = socket.assigns[:current_user]["token"]
    uuid = UUID.string_to_binary!(id)
    {:ok, resp} = Atc.get_monte_carlo_run(%Api.GetMonteCarloRunReq{id: uuid}, token)
    {:ok, started_at} = DateTime.from_unix(resp.started)

    samples =
      resp.batches
      |> Enum.flat_map(fn b -> batch_samples(b) end)

    grid_count = max((samples |> Enum.count()) / 400, 1) |> ceil()

    grid = calculate_grid(samples)

    stats = calculate_stats(samples)

    run =
      %{
        id: resp.id,
        name: resp.name,
        started: DateTime.to_string(started_at)
      }
      |> Map.merge(stats)

    spawn_batch_event_task(self(), token, uuid)

    {:ok,
     socket
     |> stream(:grid, grid)
     |> stream(:samples, samples)
     |> assign(:run, run)
     |> assign(:samples, samples |> Enum.to_list())}
  end

  defp spawn_batch_event_task(pid, token, uuid) do
    Task.start(fn ->
      addr = Application.get_env(:elodin_dashboard, ElodinDashboard.Atc)[:internal_addr]

      {:ok, channel} =
        GRPC.Stub.connect(addr)

      {:ok, stream} =
        channel
        |> Elodin.Types.Api.Api.Stub.monte_carlo_batch_events(
          %Api.GetSandboxReq{id: uuid},
          metadata: %{"Authorization" => "Bearer #{token}"}
        )

      Enum.each(stream, fn event ->
        {:ok, batch} = event
        send(pid, {:batch_event, batch})
      end)
    end)
  end

  def batch_samples(b) do
    for(<<bit::1 <- b.failures>>, do: bit == 1)
    |> Enum.with_index()
    |> Enum.take(b.samples)
    |> Enum.map(fn {is_failure, i} ->
      %{
        id: b.batch_number * b.samples + i,
        type:
          case {b.status, is_failure} do
            {:PENDING, _} -> "pending"
            {:RUNNING, _} -> "active"
            {:DONE, true} -> "fail"
            {:DONE, false} -> "pass"
          end,
        progress:
          case b.status do
            :PENDING -> 0
            :RUNNING -> 50
            :DONE -> 100
          end
      }
    end)
  end

  def calculate_grid(samples) do
    max_grid_count = 400
    grid_count = max((samples |> Enum.count()) / max_grid_count, 1) |> ceil()

    samples
    |> Enum.chunk_every(grid_count)
    |> Enum.with_index()
    |> Enum.map(fn {chunk, i} ->
      chunk
      |> Enum.reduce(
        %{
          id: i,
          type: "pending"
        },
        fn sample, acc ->
          if acc[:type] == "fail",
            do: acc,
            else: %{
              id: acc[:id],
              type: sample.type
            }
        end
      )
    end)
  end

  def calculate_stats(samples) do
    failure_count =
      samples
      |> Enum.map(fn s -> if s.type == "fail", do: 1, else: 0 end)
      |> Enum.sum()

    passed_count =
      samples
      |> Enum.map(fn s -> if s.type == "pass", do: 1, else: 0 end)
      |> Enum.sum()

    pending_count =
      samples
      |> Enum.map(fn s -> if s.type == "pending", do: 1, else: 0 end)
      |> Enum.sum()

    active_count =
      samples
      |> Enum.map(fn s -> if s.type == "active", do: 1, else: 0 end)
      |> Enum.sum()

    sample_count = samples |> Enum.count()

    %{
      sample_count: sample_count,
      active_count: active_count,
      pending_count: pending_count,
      passed_count: passed_count,
      failure_count: failure_count
    }
  end

  def handle_info({:batch_event, batch}, socket) do
    samples = socket.assigns[:samples]
    run = socket.assigns[:run]
    offset = batch.batch_number * 100
    new_samples = batch_samples(batch)

    samples =
      0..(length(new_samples) - 1)
      |> Enum.reduce(samples, fn index, samples ->
        samples |> List.replace_at(index + offset, Enum.at(new_samples, index))
      end)

    grid = calculate_grid(samples)

    {:noreply,
     socket
     |> assign(:samples, samples)
     |> stream(:grid, grid)
     |> stream(:samples, samples)
     |> assign(:run, run |> Map.merge(calculate_stats(samples)))}
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
      class={["rounded-elo-xxs border-tan border-opacity-60 border transition-all"]}
    />
    """
  end

  def grid_box(%{type: type} = assigns) do
    color = status_color(type)

    assigns = assign(assigns, :color, color)

    ~H"""
    <div style="width: 24px; height: 24px;" class={["rounded-elo-xxs transition-all", @color]} />
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
        <div style={"width: #{(1 - (@value / 100.0)) * 16}px; height: 8px;"} class="bg-[#FFFBF0]" />
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
                <%= @run.name %>
              </div>
              <div class="text-crema-60">
                Project
              </div>
              <.divider />
              <.horizontal_label label="STARTED" value={@run.started} class="mb-elo-lg" />
              <.horizontal_label label="EST TIME REMAINING" value="3:00:00" class="mb-elo-lg" />
              <.horizontal_label label="SAMPLE COUNT" value={@run.sample_count} class="" />
              <.divider />
              <.label_progress_bar
                number={@run.passed_count}
                value={@run.passed_count / @run.sample_count}
                label="PASS"
              />
              <.label_progress_bar
                number={@run.failure_count}
                value={@run.failure_count / @run.sample_count}
                label="FAIL"
                color="bg-red"
              />
              <.label_progress_bar
                number={@run.active_count}
                value={@run.active_count / @run.sample_count}
                label="ACTIVE"
                color="bg-yellow"
              />
              <.label_progress_bar
                number={@run.pending_count}
                value={@run.pending_count / @run.sample_count}
                label="PENDING"
                color="bg-tan"
              />
              <%= if @run.pending_count > 0 do %>
                <.button type="danger">
                  <.stop />
                  <span class="leading-3">STOP</span>
                </.button>
              <% end %>
            </div>
          </div>
          <div class="flex max-md:h-full md:h-[682px] w-[360px] sm:w-[500px] md:w-[682px] bg-black-secondary rounded-elo-xs border border-white border-opacity-10 flex-col items-center self-center">
            <div class="grid grid-cols-10 sm:grid-cols-16 md:grid-cols-20 gap-2 text-white p-6">
              <.grid_box :for={{id, sample} <- @streams.grid} type={sample.type} />
            </div>
          </div>
        </div>
        <div class="flex w-full max-h-[500px] mt-elo-xl bg-black-secondary rounded-elo-xs border border-white border-opacity-10 overflow-scroll font-mono font-medium tracking-elo-mono-small">
          <table class="w-full text-crema text-sm text-left">
            <thead class="sticky top-0">
              <tr class="h-[51px] bg-black-header text-crema text-crema-60 text-sm">
                <td class="pl-elo-xl">SAMPLE</td>
                <td>STATE</td>
                <td>PROGRESS</td>
              </tr>
            </thead>
            <tbody>
              <tr :for={{id, sample} <- @streams.samples} class="h-[51px] font-mono font-medium">
                <td class="pl-elo-xl"><%= sample.id %></td>
                <td><.status_label type={sample.type} /></td>
                <td><.smolgress value={sample.progress} type={sample.type} /></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </.navbar_layout>
    """
  end
end
