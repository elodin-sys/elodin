defmodule ParacosmDashboard.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      ParacosmDashboardWeb.Telemetry,
      ParacosmDashboard.Repo,
      {DNSCluster, query: Application.get_env(:paracosm_dashboard, :dns_cluster_query) || :ignore},
      {Phoenix.PubSub, name: ParacosmDashboard.PubSub},
      # Start the Finch HTTP client for sending emails
      {Finch, name: ParacosmDashboard.Finch},
      # Start a worker by calling: ParacosmDashboard.Worker.start_link(arg)
      # {ParacosmDashboard.Worker, arg},
      # Start to serve requests, typically the last entry
      ParacosmDashboardWeb.Endpoint
    ]

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: ParacosmDashboard.Supervisor]
    Supervisor.start_link(children, opts)
  end

  # Tell Phoenix to update the endpoint configuration
  # whenever the application is updated.
  @impl true
  def config_change(changed, _new, removed) do
    ParacosmDashboardWeb.Endpoint.config_change(changed, removed)
    :ok
  end
end
