defmodule ElodinDashboardWeb.OAuthController do
  use ElodinDashboardWeb, :controller

  alias ElodinDashboardWeb.UserAuth

  def callback(conn, %{"code" => code, "state" => state}) do
    conn |> UserAuth.callback(%{"code" => code, "state" => state})
  end

  def log_out_callback(conn, _) do
    conn |> UserAuth.log_out_user()
  end
end
