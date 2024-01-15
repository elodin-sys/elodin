defmodule ElodinDashboardWeb.EditorLive do
  use ElodinDashboardWeb, :live_view
  alias Elodin.Types.Api
  alias ElodinDashboard.Atc
  alias ElodinDashboardWeb.EditorComponents
  import ElodinDashboardWeb.CoreComponents
  import ElodinDashboardWeb.NavbarComponents
  import ElodinDashboardWeb.IconComponents

  def mount(%{"id" => id}, _, socket) do
    token = socket.assigns[:current_user]["token"]
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
         |> assign(:url, uri)
         |> assign(:sandbox, %{
           id_string: id_string,
           id: sandbox.id,
           name: sandbox.name,
           code: sandbox.code,
           draft_code: sandbox.draft_code,
           status: sandbox.status,
           public: sandbox.public,
           readonly: sandbox.user_id != socket.assigns[:current_user]["id"]
         })
         |> assign(:share_link, "https://elodin.dev/sandbox/#{id_string}")}
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
       readonly: sandbox.user_id != socket.assigns[:current_user]["id"]
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

  def handle_event("update_code", _, socket) do
    token = socket.assigns[:current_user]["token"]
    sandbox = socket.assigns[:sandbox]
    value = sandbox[:draft_code]

    {:ok, _} =
      Atc.update_sandbox(
        %Api.UpdateSandboxReq{id: sandbox[:id], code: value, name: sandbox[:name]},
        token
      )

    {:noreply, socket |> assign(:sandbox, Map.put(sandbox, :code, value))}
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

  def render(assigns) do
    ~H"""
    <.navbar_layout current_user={@current_user}>
      <:navbar_left>
        <.link patch={~p"/"}>
          <.arrow_left class="mr-elo-m" />
        </.link>
        <span class="font-semibold">
          Sandbox - <%= @sandbox.name %>
        </span>
      </:navbar_left>
      <:navbar_right>
        <.link patch={~p"/sandbox/#{@sandbox.id_string}/share"} phx-click={show_modal("share")}>
          <.button type="outline" class="mr-1.5">
            Share
          </.button>
        </.link>
      </:navbar_right>

      <div class="flex flex-col h-full">
        <div class="flex w-full h-full">
          <div class="w-1/2 h-full">
            <div style="height: calc(100% - 256px - 40px);" class="pt-3 bg-code">
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
                      "readOnly" => @sandbox.readonly
                    }
                  )
                }
              />
            </div>
            <EditorComponents.console logs={@sandbox.status} update_click={JS.push("update_code")} />
          </div>
          <%= if @sandbox.status == :RUNNING do %>
            <EditorComponents.editor_wasm url={@url} />
          <% else %>
            <div class="w-1/2 h-full flex justify-center items-center">
              <.spinner class="animate-spin w-16 h-16" />
            </div>
          <% end %>
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
        class="flex justify-left items-center flex-row gap-elo-xl mt-elo-xl"
        phx-change="set_public"
      >
        <.input name="public" type="checkbox" id="public" value={@sandbox.public} />
        <.label for="public" class="mr-elo-m">Public Sandbox?</.label>
      </.form>
      <div class="flex justify-center items-center flex-row gap-elo-xl mt-elo-xl">
        <.input name="share-link" id="share-link" value={@share_link} />
        <.button class="h-[36px] mt-2" phx-click={JS.dispatch("phx:copy", to: "\#share-link")}>
          Copy Link
        </.button>
      </div>
    </.modal>
    """
  end
end
