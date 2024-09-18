defmodule ElodinDashboardWeb.EditorLive do
  require Logger
  use ElodinDashboardWeb, :live_view
  alias Elodin.Types.Api
  alias ElodinDashboard.Atc
  alias ElodinDashboardWeb.EditorComponents
  import ElodinDashboardWeb.CoreComponents
  import ElodinDashboardWeb.NavbarComponents
  import ElodinDashboardWeb.IconComponents

  def mount(%{"id" => id}, _, socket) do
    token = socket.assigns[:current_user]["token"]

    Logger.info("editor page accessed",
      user: socket.assigns[:current_user]["email"],
      sandbox_id: id
    )

    atc_config = Application.get_env(:elodin_dashboard, Atc)
    scheme = if atc_config[:tls], do: "wss", else: "ws"

    uri =
      %URI{scheme: scheme, authority: atc_config[:addr], path: "/sim/ws"}
      |> URI.append_path("/#{id}")
      |> URI.append_query(URI.encode_query(token: token))

    uuid = UUID.string_to_binary!(id)

    if Map.has_key?(socket.assigns, :sandbox) do
      {:ok, socket}
    else
      with {:ok, sandbox} <-
             Atc.get_sandbox(%Api.GetSandboxReq{id: uuid}, token),
           {:ok, _} <- Atc.boot_sandbox(%Api.BootSandboxReq{id: uuid}, token) do
        id_string = UUID.binary_to_string!(sandbox.id)

        spawn_sandbox_task(self(), token, uuid)

        {:ok,
         socket
         |> assign(:is_demo_sandbox, sandbox.user_id == nil)
         |> assign(:url, uri)
         |> assign(:sandbox, %{
           id_string: id_string,
           id: sandbox.id,
           name: sandbox.name,
           code: sandbox.code,
           draft_code: sandbox.draft_code,
           status: sandbox.status,
           public: sandbox.public,
           readonly:
             sandbox.user_id != socket.assigns[:current_user]["id"] && sandbox.user_id != ""
         })
         |> assign(:errors, [])
         |> assign(:share_link, "#{ElodinDashboardWeb.Endpoint.url()}/sandbox/#{id_string}")
         |> assign(:show_console, true)}
      else
        err -> err
      end
    end
  end

  defp spawn_sandbox_task(pid, token, uuid) do
    Task.start(fn ->
      addr = Application.get_env(:elodin_dashboard, ElodinDashboard.Atc)[:internal_addr]

      {:ok, channel} =
        GRPC.Stub.connect(addr)

      {:ok, stream} =
        channel
        |> Elodin.Types.Api.Api.Stub.sandbox_events(
          %Api.GetSandboxReq{id: uuid},
          metadata: %{"Authorization" => "Bearer #{token}"}
        )

      Enum.each(stream, fn event ->
        {:ok, sandbox} = event
        send(pid, {:update_sandbox, sandbox})
      end)
    end)
  end

  def handle_params(_, _, socket) do
    {:noreply, socket}
  end

  def handle_info({:update_sandbox, sandbox}, socket) do
    {:noreply,
     socket
     |> assign(:sandbox, %{
       id_string: UUID.binary_to_string!(sandbox.id),
       id: sandbox.id,
       name: sandbox.name,
       code: sandbox.code,
       draft_code: sandbox.draft_code,
       status: sandbox.status,
       public: sandbox.public,
       readonly: sandbox.user_id != socket.assigns[:current_user]["id"] && sandbox.user_id != ""
     })}
  end

  def handle_event("set_editor_value", %{"value" => value}, socket) do
    token = socket.assigns[:current_user]["token"]
    sandbox = socket.assigns[:sandbox]

    {:ok, _} =
      Atc.update_sandbox(
        %Api.UpdateSandboxReq{id: sandbox[:id], draft_code: value, name: sandbox[:name]},
        token
      )

    {:noreply, socket |> assign(:sandbox, Map.put(sandbox, :draft_code, value))}
  end

  def handle_event("update_code", resp, socket) do
    token = socket.assigns[:current_user]["token"]
    sandbox = socket.assigns[:sandbox]
    value = sandbox[:draft_code]

    {:ok, update_resp} =
      Atc.update_sandbox(
        %Api.UpdateSandboxReq{id: sandbox[:id], code: value, name: sandbox[:name]},
        token
      )

    {:noreply,
     socket
     |> assign(:sandbox, Map.put(sandbox, :code, value))
     |> assign(:errors, update_resp.errors)}
  end

  def handle_event("set_public", %{"public" => public_str}, socket) do
    token = socket.assigns[:current_user]["token"]
    sandbox = socket.assigns[:sandbox]
    public = public_str == "true"

    {:ok, _} =
      Atc.update_sandbox(
        %Api.UpdateSandboxReq{id: sandbox[:id], public: public, name: sandbox[:name]},
        token
      )

    {:noreply, socket |> assign(:sandbox, Map.put(sandbox, :public, public))}
  end

  def handle_event("toggle_console", %{"show_console" => show_console}, socket) do
    {:noreply, socket |> assign(:show_console, show_console)}
  end

  def render(assigns) do
    ~H"""
    <.navbar_layout current_user={@current_user} is_demo_sandbox={@is_demo_sandbox}>
      <:navbar_center>
        <span class="font-semibold"><%= @sandbox.name %></span>
      </:navbar_center>
      <:navbar_right>
        <.link patch={~p"/sandbox/#{@sandbox.id_string}/share"} phx-click={show_modal("share")}>
          <.button class="mr-1.5">
            Share
          </.button>
        </.link>
        <.link href="https://docs.elodin.systems">
          <.button type="secondary" class="mr-1.5">
            Docs
          </.button>
        </.link>
      </:navbar_right>

      <div class="flex flex-col overflow-auto h-full bg-black-primary w-full">
        <div class="flex w-full md:h-full flex-col-reverse md:flex-row">
          <div class="flex flex-col md:w-1/2 md:h-full">
            <div
              class={[
                "bg-black-primary transition-all",
                "max-md:h-[calc(50vh-128px)]",
                if(@show_console, do: "md:h-[calc(100%-20rem)]", else: "md:h-[calc(100%-4rem)]")
              ]}
              style={}
            >
              <LiveMonacoEditor.code_editor
                class="code-editor"
                value={@sandbox.draft_code}
                change="set_editor_value"
                opts={
                  Map.merge(
                    LiveMonacoEditor.default_opts(),
                    %{
                      "language" => "python",
                      "minimap" => %{"enabled" => false},
                      "automaticLayout" => true,
                      "readOnly" => @sandbox.readonly,
                      "theme" => "vs-dark"
                    }
                  )
                }
              />
            </div>

            <div class="max-md:fixed max-md:w-[100vh] max-md:bottom-0">
              <div class="flex flex-row justify-between h-16 p-4 shadow-lg bg-black-primary flex items-center">
                <.button
                  type="secondary"
                  class="flex gap-2 py-2"
                  phx-click={JS.push("toggle_console", value: %{show_console: !@show_console})}
                >
                  <.arrow_chevron_up class={[
                    "transition-all",
                    if(@show_console, do: "-scale-100", else: "")
                  ]} />
                  <span class="leading-4">CONSOLE</span>
                </.button>
                <.button class="flex gap-2 py-2.5" phx-click={JS.push("update_code")}>
                  <.lightning />
                  <span class="leading-3">UPDATE SIM</span>
                </.button>
              </div>

              <EditorComponents.console
                hide={!@show_console}
                logs={([@sandbox.status] ++ @errors) |> Enum.join("\n")}
              />
            </div>
          </div>
          <div class="flex max-md:h-[50vh] md:w-1/2 md:h-full justify-center items-center">
            <%= if @sandbox.status == :RUNNING do %>
              <EditorComponents.editor_wasm url={@url} />
            <% else %>
              <.spinner class="animate-spin w-16 h-16" />
            <% end %>
          </div>
        </div>
      </div>
    </.navbar_layout>

    <.modal
      id="share"
      show={@live_action == :share}
      on_cancel={JS.patch(~p"/sandbox/#{@sandbox.id_string}")}
    >
      <h2 class="font-semibold absolute top-elo-xl left-elo-xl ">Share</h2>
      <.form
        class="pt-6 flex justify-left items-center flex-row gap-elo-xl mt-elo-xl"
        phx-change="set_public"
      >
        <.input name="public" type="checkbox" id="public" value={@sandbox.public} />
        <.label for="public" class="mr-elo-m">Public Sandbox?</.label>
      </.form>
      <div class="flex items-center flex-row gap-elo-xl mt-elo-xl">
        <.input name="share-link" id="share-link" value={@share_link} class="w-full max-w-[300px]" />
        <.button class="h-[36px] mt-2" phx-click={JS.dispatch("phx:copy", to: "\#share-link")}>
          Copy Link
        </.button>
      </div>
    </.modal>
    """
  end
end
