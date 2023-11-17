defmodule ParacosmDashboardWeb.UserSessionController do
  use ParacosmDashboardWeb, :controller

  alias ParacosmDashboardWeb.UserAuth

  def log_in(conn, _params) do
    conn |> UserAuth.redirect_to_login()
  end

  def delete(conn, _params) do
    conn
    |> put_flash(:info, "Logged out successfully.")
    |> UserAuth.log_out_user()
  end
end
