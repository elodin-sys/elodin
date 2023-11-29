defmodule ParacosmDashboardWeb.HealthCheckController do
  use ParacosmDashboardWeb, :controller

  def callback(conn, _opts) do
    conn |> text("OK")
  end
end
