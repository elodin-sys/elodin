defmodule ElodinDashboardWeb.OAuthController do
  use ElodinDashboardWeb, :controller

  alias ElodinDashboardWeb.UserAuth

  def callback(conn, %{"code" => code, "state" => state}) do
    conn |> UserAuth.callback(%{"code" => code, "state" => state})
  end
end
