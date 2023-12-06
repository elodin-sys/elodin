defmodule ParacosmDashboardWeb.EditorLive do
  use ParacosmDashboardWeb, :live_view
  alias ParacosmDashboardWeb.EditorComponents
  alias Paracosm.Types.Api
  alias ParacosmDashboard.Atc
  import ParacosmDashboardWeb.CoreComponents
  import ParacosmDashboardWeb.NavbarComponents
  import ParacosmDashboardWeb.IconComponents

  def mount(%{"id" => id}, _, socket) do
    token = socket.assigns[:current_user]["token"]
    atc_config = Application.get_env(:paracosm_dashboard, Atc)
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
             Atc.get_sandbox(Api.GetSandboxReq.new(id: uuid), token),
           {:ok, _} <- Atc.boot_sandbox(Api.BootSandboxReq.new(id: uuid), token) do
        pid = self()

        Task.start(fn ->
          addr = Application.get_env(:paracosm_dashboard, ParacosmDashboard.Atc)[:internal_addr]

          {:ok, channel} =
            GRPC.Stub.connect(addr)

          {:ok, stream} =
            channel
            |> Paracosm.Types.Api.Api.Stub.sandbox_events(
              Api.GetSandboxReq.new(id: uuid),
              metadata: %{"Authorization" => "Bearer #{token}"}
            )

          Enum.each(stream, fn event ->
            {:ok, sandbox} = event
            send(pid, {:update_sandbox, sandbox})
          end)
        end)

        id_string = UUID.binary_to_string!(sandbox.id)

        {:ok,
         socket
         |> assign(:url, uri)
         |> assign(:sandbox, %{
           id_string: id_string,
           id: sandbox.id,
           name: sandbox.name,
           code: sandbox.code,
           status: sandbox.status
         })
         |> assign(
           :share_form,
           to_form(%{"link" => "https://elodin.dev/sandbox/#{id_string}"})
         )}
      else
        err -> err
      end
    end
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
       status: sandbox.status
     })}
  end

  def handle_event("set_editor_value", %{"value" => value}, socket) do
    token = socket.assigns[:current_user]["token"]
    sandbox = socket.assigns[:sandbox]

    {:ok, _} =
      Atc.update_sandbox(
        Api.UpdateSandboxReq.new(id: sandbox[:id], code: value, name: sandbox[:name]),
        token
      )

    {:noreply, socket}
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
                value={@sandbox.code}
                change="set_editor_value"
                opts={
                  Map.merge(
                    LiveMonacoEditor.default_opts(),
                    %{
                      "language" => "python",
                      "minimap" => %{"enabled" => false},
                      "automaticLayout" => true
                    }
                  )
                }
              />
            </div>
            <EditorComponents.console logs={@sandbox.status} />
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


    <.modal id="share" show={@live_action == :share} on_cancel={JS.patch(~p"/sandbox/#{@sandbox.id_string}")}>
      <h2 class="font-semibold absolute top-elo-xl left-elo-xl ">Share</h2>
      <.form
        for={@share_form}
        phx-submit="save"
        class="flex justify-center items-center flex-row gap-elo-xl mt-elo-xl"
      >
        <.input name="link" field={@share_form[:link]}/>
        <.button class="h-[36px] mt-2 ">Copy Link</.button>
      </.form>
    </.modal>
    """
  end
end
