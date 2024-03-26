defmodule ElodinDashboardWeb.UserAuth do
  use ElodinDashboardWeb, :verified_routes
  import Logger
  import Plug.Conn
  import Phoenix.Controller

  alias Assent.Strategy.Auth0
  alias Assent.Strategy.OIDC

  # Make the remember me cookie valid for 60 days.
  # If you want bump or reduce this value, also change
  # the token expiry itself in UserToken.
  @max_age 60 * 60 * 24 * 60
  @id_token_cookie "_elodin_dashboard_web_user_token_id"
  @id_token_options [sign: true]
  @remember_me_cookie "_elodin_dashboard_web_user_remember_me"
  @remember_me_options [sign: true, max_age: @max_age, same_site: "Lax"]

  def assent_config do
    Keyword.merge(
      Application.get_env(:elodin_dashboard, ElodinDashboardWeb.UserAuth),
      http_adapter: {Assent.HTTPAdapter.Finch, [supervisor: ElodinDashboard.Finch]}
    )
  end

  def get_user_by_token(token) do
    case ElodinDashboard.Atc.current_user(struct(Elodin.Types.Api.CurrentUserReq), token) do
      {:ok, user} ->
        {:ok,
         %{
           "token" => token,
           "name" => user.name,
           "email" => user.email,
           "avatar" => user.avatar,
           "id" => user.id
         }}

      {:error, err} ->
        {:error, err}
    end
  end

  def redirect_to_login(conn) do
    {:ok, %{url: url, session_params: session_params}} =
      assent_config() |> Auth0.authorize_url()

    conn
    |> put_session(:session_params, session_params[:state])
    |> redirect(external: url)
  end

  def callback(conn, params) do
    state = get_session(conn, :session_params)

    {:ok, %{token: token}} =
      assent_config()
      |> Assent.Config.put(:session_params, %{state: state})
      |> Auth0.callback(params)

    log_in_user(conn, token, params)
  end

  @doc """
  Logs the user in.

  It renews the session ID and clears the whole session
  to avoid fixation attacks. See the renew_session
  function to customize this behaviour.

  It also sets a `:live_socket_id` key in the session,
  so LiveView sessions are identified and automatically
  disconnected on log out. The line can be safely removed
  if you are not using LiveView.
  """
  def log_in_user(conn, token, params \\ %{}) do
    user_return_to = get_session(conn, :user_return_to)
    access_token = token["access_token"]

    query_params =
      case get_user_by_token(access_token) do
        {:error, %GRPC.RPCError{status: 5}} ->
          ElodinDashboard.Atc.create_user(struct(Elodin.Types.Api.CreateUserReq), access_token)
          "?onboarding=1"

        {:error, _} ->
          ""

        {:ok, _} ->
          ""
      end

    conn
    |> renew_session()
    |> put_user_id_token(token["id_token"])
    |> put_token_in_session(access_token)
    |> maybe_write_remember_me_cookie(access_token, params)
    |> redirect(to: "#{user_return_to || signed_in_path(conn)}#{query_params}")
  end

  defp maybe_write_remember_me_cookie(conn, token, %{"remember_me" => "true"}) do
    put_resp_cookie(conn, @remember_me_cookie, token, @remember_me_options)
  end

  defp maybe_write_remember_me_cookie(conn, _token, _params) do
    conn
  end

  defp put_user_id_token(conn, token) do
    state = get_session(conn, :session_params)
    config = assent_config() |> Assent.Config.put(:session_params, %{state: state})

    case OIDC.validate_id_token(config, token) do
      {:ok, jwt} ->
        max_age = jwt.claims["exp"] - :os.system_time(:second)

        put_resp_cookie(conn, @id_token_cookie, token, [{:max_age, max_age} | @id_token_options])

      {:error, error} ->
        info("Failed to validate id_token: #{inspect(error)}")
        conn
    end
  end

  # This function renews the session ID and erases the whole
  # session to avoid fixation attacks. If there is any data
  # in the session you may want to preserve after log in/log out,
  # you must explicitly fetch the session data before clearing
  # and then immediately set it after clearing, for example:
  #
  #     defp renew_session(conn) do
  #       preferred_locale = get_session(conn, :preferred_locale)
  #
  #       conn
  #       |> configure_session(renew: true)
  #       |> clear_session()
  #       |> put_session(:preferred_locale, preferred_locale)
  #     end
  #
  defp renew_session(conn) do
    conn
    |> configure_session(renew: true)
    |> clear_session()
  end

  defp drop_session(conn) do
    conn
    |> configure_session(drop: true)
    |> clear_session()
    |> delete_resp_cookie(@remember_me_cookie)
    |> delete_resp_cookie(@id_token_cookie)
  end

  defp logout_url(config, params, alt \\ false) do
    with {:ok, base_url} <- Assent.Config.__base_url__(config) do
      logout_url =
        case alt do
          true -> Assent.Config.get(config, :alt_logout_url, "/v2/logout")
          false -> Assent.Config.get(config, :logout_url, "/oidc/logout")
        end

      Assent.Strategy.to_url(base_url, logout_url, params)
    end
  end

  defp fetch_id_token(conn, config) do
    conn = fetch_cookies(conn, signed: [@id_token_cookie])

    case conn.cookies[@id_token_cookie] do
      token when not is_nil(token) ->
        case OIDC.validate_id_token(config, token) do
          {:ok, _} ->
            token

          {:error, error} ->
            info("Failed to validate id_token: #{inspect(error)}")
            nil
        end

      _ ->
        info("User's id_token is missing")
        nil
    end
  end

  def redirect_to_logout(conn) do
    state = get_session(conn, :session_params)
    config = assent_config() |> Assent.Config.put(:session_params, %{state: state})

    {:ok, client_id} = Assent.Config.fetch(config, :client_id)

    {:ok, post_logout_redirect_uri} =
      Assent.Config.fetch(config, :post_logout_redirect_uri)

    logout_url =
      case fetch_id_token(conn, config) do
        id_token when not is_nil(id_token) ->
          logout_url(config, [
            {:id_token_hint, id_token},
            {:client_id, client_id},
            {:post_logout_redirect_uri, post_logout_redirect_uri}
          ])

        _ ->
          logout_url(
            config,
            [
              {:client_id, client_id},
              {:returnTo, post_logout_redirect_uri}
            ],
            true
          )
      end

    redirect(conn, external: logout_url)
  end

  defp disconnect_socket(conn) do
    if live_socket_id = get_session(conn, :live_socket_id) do
      ElodinDashboardWeb.Endpoint.broadcast(live_socket_id, "disconnect", %{})
    end

    conn
  end

  @doc """
  Logs the user out.

  It clears all session data for safety. See renew_session.
  """
  def log_out_user(conn) do
    conn
    |> disconnect_socket()
    |> drop_session()
    |> redirect(to: ~p"/")
  end

  @doc """
  Authenticates the user by looking into the session
  and remember me token.
  """
  def fetch_current_user(conn, _opts) do
    with {user_token, conn} when not is_nil(user_token) <- ensure_user_token(conn),
         {:ok, user} <- get_user_by_token(user_token) do
      assign(conn, :current_user, user)
    else
      _ -> assign(conn, :current_user, nil)
    end
  end

  defp ensure_user_token(conn) do
    if token = get_session(conn, :user_token) do
      {token, conn}
    else
      conn = fetch_cookies(conn, signed: [@remember_me_cookie])

      if token = conn.cookies[@remember_me_cookie] do
        {token, put_token_in_session(conn, token)}
      else
        {nil, conn}
      end
    end
  end

  @doc """
  Handles mounting and authenticating the current_user in LiveViews.

  ## `on_mount` arguments

    * `:mount_current_user` - Assigns current_user
      to socket assigns based on user_token, or nil if
      there's no user_token or no matching user.

    * `:ensure_authenticated` - Authenticates the user from the session,
      and assigns the current_user to socket assigns based
      on user_token.
      Redirects to login page if there's no logged user.

    * `:redirect_if_user_is_authenticated` - Authenticates the user from the session.
      Redirects to signed_in_path if there's a logged user.

  ## Examples

  Use the `on_mount` lifecycle macro in LiveViews to mount or authenticate
  the current_user:

      defmodule ElodinDashboardWeb.PageLive do
        use ElodinDashboardWeb, :live_view

        on_mount {ElodinDashboardWeb.UserAuth, :mount_current_user}
        ...
      end

  Or use the `live_session` of your router to invoke the on_mount callback:

      live_session :authenticated, on_mount: [{ElodinDashboardWeb.UserAuth, :ensure_authenticated}] do
        live "/profile", ProfileLive, :index
      end
  """
  def on_mount(:mount_current_user, _params, session, socket) do
    {:cont, mount_current_user(socket, session)}
  end

  def on_mount(:ensure_authenticated, _params, session, socket) do
    socket = mount_current_user(socket, session)

    if socket.assigns.current_user do
      {:cont, socket}
    else
      socket =
        socket
        |> redirect_to_login()

      {:halt, socket}
    end
  end

  def on_mount(:redirect_if_user_is_authenticated, _params, session, socket) do
    socket = mount_current_user(socket, session)

    if socket.assigns.current_user do
      {:halt, Phoenix.LiveView.redirect(socket, to: signed_in_path(socket))}
    else
      {:cont, socket}
    end
  end

  defp mount_current_user(socket, session) do
    Phoenix.Component.assign_new(socket, :current_user, fn ->
      if user_token = session["user_token"] do
        {:ok, user} = get_user_by_token(user_token)
        user
      end
    end)
  end

  @doc """
  Used for routes that require the user to not be authenticated.
  """
  def redirect_if_user_is_authenticated(conn, _opts) do
    if conn.assigns[:current_user] do
      conn
      |> redirect(to: signed_in_path(conn))
      |> halt()
    else
      conn
    end
  end

  @doc """
  Used for routes that require the user to be authenticated.

  If you want to enforce the user email is confirmed before
  they use the application at all, here would be a good place.
  """
  def require_authenticated_user(conn, _opts) do
    if conn.assigns[:current_user] do
      conn
    else
      conn
      |> maybe_store_return_to()
      |> redirect_to_login()
      |> halt()
    end
  end

  defp put_token_in_session(conn, token) do
    conn
    |> put_session(:user_token, token)
    |> put_session(:live_socket_id, "users_sessions:#{Base.url_encode64(token)}")
  end

  defp maybe_store_return_to(%{method: "GET"} = conn) do
    put_session(conn, :user_return_to, current_path(conn))
  end

  defp maybe_store_return_to(conn), do: conn

  defp signed_in_path(_conn), do: ~p"/"
end
