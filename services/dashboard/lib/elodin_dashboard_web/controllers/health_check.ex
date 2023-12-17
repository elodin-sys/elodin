defmodule ElodinDashboardWeb.HealthCheckController do
  use ElodinDashboardWeb, :controller

  def callback(conn, _opts) do
    conn |> text("OK")
  end
end
