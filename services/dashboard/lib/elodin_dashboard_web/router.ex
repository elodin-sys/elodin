defmodule ElodinDashboardWeb.Router do
  use ElodinDashboardWeb, :router

  import ElodinDashboardWeb.UserAuth

  pipeline :browser do
    plug(:accepts, ["html"])
    plug(:fetch_session)
    plug(:fetch_live_flash)
    plug(:put_root_layout, html: {ElodinDashboardWeb.Layouts, :root})
    plug(:protect_from_forgery)
    plug(:put_secure_browser_headers)
    plug(:fetch_current_user)
  end

  pipeline :api do
    plug(:accepts, ["json"])
  end

  scope "/", ElodinDashboardWeb do
    pipe_through(:browser)
  end

  # Other scopes may use custom stacks.
  # scope "/api", ElodinDashboardWeb do
  #   pipe_through :api
  # end

  # Enable LiveDashboard and Swoosh mailbox preview in development
  if Application.compile_env(:elodin_dashboard, :dev_routes) do
    # If you want to use the LiveDashboard in production, you should put
    # it behind authentication and allow only admins to access it.
    # If your application does not have an admins-only section yet,
    # you can use Plug.BasicAuth to set up some basic authentication
    # as long as you are also using SSL (which you should anyway).
    import Phoenix.LiveDashboard.Router

    scope "/dev" do
      pipe_through(:browser)

      live_dashboard("/dashboard", metrics: ElodinDashboardWeb.Telemetry)
      forward("/mailbox", Plug.Swoosh.MailboxPreview)
    end
  end

  ## Health route

  scope "/", ElodinDashboardWeb do
    get("/healthz", HealthCheckController, :callback)
  end

  ## Authentication routes

  scope "/", ElodinDashboardWeb do
    pipe_through([:browser, :redirect_if_user_is_authenticated])

    get("/oauth/callback", OAuthController, :callback)
  end

  scope "/", ElodinDashboardWeb do
    pipe_through([:browser, :require_authenticated_user])

    live_session :require_authenticated_user,
      on_mount: [{ElodinDashboardWeb.UserAuth, :ensure_authenticated}] do
      live("/onboard", OnboardingLive, :view)
      live("/onboard/:page_num", OnboardingLive, :view)
    end

    live_session :require_onboard,
      on_mount: [
        # {ElodinDashboardWeb.UserAuth, :ensure_authenticated},
        {ElodinDashboardWeb.UserAuth, :ensure_onboarded}
      ] do
      live("/", SandboxPickerLive, :list)
      live("/sandbox/new", SandboxPickerLive, :new)
      live("/sandbox/new/:template", SandboxPickerLive, :new)
      live("/sandbox/delete/:id", SandboxPickerLive, :delete)

      live("/monte_carlo/:project", MonteCarloProjectLive, :view)
      live("/monte_carlo/:project/:run", MonteCarloRunLive, :view)
    end
  end

  scope "/", ElodinDashboardWeb do
    pipe_through([:browser])

    get("/users/log_in", UserSessionController, :log_in)
    get("/users/log_out", UserSessionController, :delete)
    get("/oauth/log_out_callback", OAuthController, :log_out_callback)

    live_session :unauthed,
      on_mount: [{ElodinDashboardWeb.UserAuth, :mount_current_user}] do
      live("/sandbox/hn/:template", SandboxHNTemplateLive, :new)
      live("/sandbox/:id", EditorLive, :edit)
      live("/sandbox/:id/share", EditorLive, :share)
    end
  end
end
