defmodule ElodinDashboardWeb.MonteCarloProjectLive do
  require Logger
  use ElodinDashboardWeb, :live_view
  alias Elodin.Types.Api
  alias ElodinDashboard.Atc
  import ElodinDashboardWeb.NavbarComponents
  import ElodinDashboardWeb.SidebarComponents

  def mount(%{"project" => project}, _, socket) do
    token = socket.assigns[:current_user]["token"]

    Logger.info("monte-carlo project page accessed",
      montecarlo_project: project,
      user_email: socket.assigns[:current_user]["email"]
    )

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

        {:error, err} ->
          Logger.error(
            "monte-carlo project page - list_monte_carlo_runs - error",
            montecarlo_project: project,
            user_email: socket.assigns[:current_user]["email"],
            error: inspect(err)
          )

          []
      end

    subscription = socket.assigns.current_user["subscription_status"]
    has_subscription = subscription != nil

    subscription_end =
      if(has_subscription && subscription.subscription_end != nil,
        do: DateTime.from_unix!(subscription.subscription_end),
        else: DateTime.utc_now()
      )

    {:ok,
     socket
     |> assign(:project, project)
     |> assign(:project_runs, monte_carlo_runs)
     |> assign(:monte_carlo_reset_date, subscription_end)
     |> assign(
       :monte_carlo_free_credit,
       if(has_subscription, do: subscription.monte_carlo_credit, else: 0)
     )
     |> assign(:monte_carlo_minutes_used, socket.assigns.current_user["monte_carlo_minutes_used"])}
  end

  def render(assigns) do
    ~H"""
    <.navbar_layout current_user={@current_user}>
      <.sidebar
        project={@project}
        project_runs={@project_runs}
        monte_carlo_reset_date={@monte_carlo_reset_date}
        monte_carlo_free_credit={@monte_carlo_free_credit}
        monte_carlo_minutes_used={@monte_carlo_minutes_used}
      />

      <div class="flex grow items-center justify-center text-orange-50 opacity-60 font-semibold">
        SELECT A RUN FROM THE SIDEBAR
      </div>
    </.navbar_layout>
    """
  end
end
