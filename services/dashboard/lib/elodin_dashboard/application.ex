defmodule ElodinDashboard.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      ElodinDashboardWeb.Telemetry,
      {DNSCluster, query: Application.get_env(:elodin_dashboard, :dns_cluster_query) || :ignore},
      {Phoenix.PubSub, name: ElodinDashboard.PubSub},
      # Start the Finch HTTP client for sending emails
      {Finch, name: ElodinDashboard.Finch},
      # Start a worker by calling: ElodinDashboard.Worker.start_link(arg)
      # {ElodinDashboard.Worker, arg},
      # Start to serve requests, typically the last entry
      ElodinDashboardWeb.Endpoint,
      :poolboy.child_spec(:atc, atc_poolboy_config())
    ]

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: ElodinDashboard.Supervisor]
    Supervisor.start_link(children, opts)
  end

  def atc_poolboy_config do
    [
      name: {:local, :atc},
      worker_module: ElodinDashboard.AtcAgent,
      size: 8,
      max_overflow: 64
    ]
  end

  # Tell Phoenix to update the endpoint configuration
  # whenever the application is updated.
  @impl true
  def config_change(changed, _new, removed) do
    ElodinDashboardWeb.Endpoint.config_change(changed, removed)
    :ok
  end
end
