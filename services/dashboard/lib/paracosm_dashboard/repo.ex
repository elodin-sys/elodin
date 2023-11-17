defmodule ParacosmDashboard.Repo do
  use Ecto.Repo,
    otp_app: :paracosm_dashboard,
    adapter: Ecto.Adapters.Postgres
end
