defmodule ParacosmDashboardWeb.UserSettingsLive do
  use ParacosmDashboardWeb, :live_view

  def render(assigns) do
    ~H"""
    """
  end

  def mount(%{"token" => token}, _session, socket) do
  end

  def mount(_params, _session, socket) do
    {:ok, socket}
  end

  def handle_params(_params, _uri, socket) do
    {:noreply, socket}
  end
end
