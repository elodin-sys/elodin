defmodule ParacosmDashboardWeb.Router do
  use ParacosmDashboardWeb, :router

  import ParacosmDashboardWeb.UserAuth

  pipeline :browser do
    plug(:accepts, ["html"])
    plug(:fetch_session)
    plug(:fetch_live_flash)
    plug(:put_root_layout, html: {ParacosmDashboardWeb.Layouts, :root})
    plug(:protect_from_forgery)
    plug(:put_secure_browser_headers)
    plug(:fetch_current_user)
  end

  pipeline :api do
    plug(:accepts, ["json"])
  end

  scope "/", ParacosmDashboardWeb do
    pipe_through(:browser)
  end

  # Other scopes may use custom stacks.
  # scope "/api", ParacosmDashboardWeb do
  #   pipe_through :api
  # end

  # Enable LiveDashboard and Swoosh mailbox preview in development
  if Application.compile_env(:paracosm_dashboard, :dev_routes) do
    # If you want to use the LiveDashboard in production, you should put
    # it behind authentication and allow only admins to access it.
    # If your application does not have an admins-only section yet,
    # you can use Plug.BasicAuth to set up some basic authentication
    # as long as you are also using SSL (which you should anyway).
    import Phoenix.LiveDashboard.Router

    scope "/dev" do
      pipe_through(:browser)

      live_dashboard("/dashboard", metrics: ParacosmDashboardWeb.Telemetry)
      forward("/mailbox", Plug.Swoosh.MailboxPreview)
    end
  end

  ## Authentication routes

  scope "/", ParacosmDashboardWeb do
    pipe_through([:browser, :redirect_if_user_is_authenticated])

    get("/oauth/callback", OAuthController, :callback)
  end

  scope "/", ParacosmDashboardWeb do
    pipe_through([:browser, :require_authenticated_user])

    live_session :require_authenticated_user,
      on_mount: [{ParacosmDashboardWeb.UserAuth, :ensure_authenticated}] do
      live("/users/settings", UserSettingsLive, :edit)
      live("/", EditorLive, :edit)
    end
  end

  scope "/", ParacosmDashboardWeb do
    pipe_through([:browser])

    get("/users/log_in", UserSessionController, :log_in)
    delete("/users/log_out", UserSessionController, :delete)
  end
end
