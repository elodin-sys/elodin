defmodule ParacosmDashboard.Release do
  @moduledoc """
  Used for executing DB release tasks when run in production without Mix
  installed.
  """
  @app :paracosm_dashboard

  defp load_app do
    Application.load(@app)
  end
end
