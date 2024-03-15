defmodule ElodinDashboardWeb.MonteCarloProjectLive do
  use ElodinDashboardWeb, :live_view
  alias Elodin.Types.Api
  alias ElodinDashboard.Atc
  import ElodinDashboardWeb.NavbarComponents
  import ElodinDashboardWeb.SidebarComponents

  def mount(%{"project" => project}, _, socket) do
    token = socket.assigns[:current_user]["token"]

    monte_carlo_runs =
      case Atc.list_monte_carlo_runs(%Api.ListMonteCarloRunsReq{}, token) do
        {:ok, monte_carlo_runs} ->
          Enum.map(monte_carlo_runs.monte_carlo_runs, fn run ->
            %{
              id: UUID.binary_to_string!(run.id),
              name: run.name,
              progress:
                case run.status do
                  :PENDING -> 0.1
                  :RUNNING -> 0.5
                  :DONE -> 1.0
                end
            }
          end)

        {:error, _} ->
          []
      end

    {:ok,
     socket
     |> assign(:project, project)
     |> assign(:project_runs, monte_carlo_runs)}
  end

  def render(assigns) do
    ~H"""
    <.navbar_layout current_user={@current_user}>
      <.sidebar project={@project} project_runs={@project_runs} />

      <div class="flex grow items-center justify-center text-orange-50 opacity-60 font-semibold">
        SELECT A RUN FROM THE SIDEBAR
      </div>
    </.navbar_layout>
    """
  end
end
