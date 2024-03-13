defmodule ElodinDashboardWeb.UserSessionController do
  use ElodinDashboardWeb, :controller

  alias ElodinDashboardWeb.UserAuth

  def log_in(conn, _params) do
    conn |> UserAuth.redirect_to_login()
  end

  def delete(conn, _params) do
    conn
    |> UserAuth.redirect_to_logout()
  end
end
