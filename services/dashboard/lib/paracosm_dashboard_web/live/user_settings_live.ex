defmodule ParacosmDashboardWeb.UserSettingsLive do
  use ParacosmDashboardWeb, :live_view

  def render(assigns) do
    ~H"""
        Settings
    """
  end

  def mount(%{"token" => token}, _session, socket) do
  end

  def mount(_params, _session, socket) do
    {:ok, socket}
  end
end
