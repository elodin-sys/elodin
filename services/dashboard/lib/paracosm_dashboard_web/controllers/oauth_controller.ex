defmodule ParacosmDashboardWeb.OAuthController do
  use ParacosmDashboardWeb, :controller

  alias ParacosmDashboardWeb.UserAuth

  def callback(conn, %{"code" => code, "state" => state}) do
    conn |> UserAuth.callback(%{"code" => code, "state" => state})
  end
end
