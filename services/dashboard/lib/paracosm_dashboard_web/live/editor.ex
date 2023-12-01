defmodule ParacosmDashboardWeb.EditorLive do
  use ParacosmDashboardWeb, :live_view
  alias ParacosmDashboardWeb.EditorComponents
  alias Paracosm.Types.Api
  alias ParacosmDashboard.Atc
  import ParacosmDashboardWeb.CoreComponents

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

        {:ok,
         socket
         |> assign(:url, uri)
         |> assign(:sandbox, %{
           id: sandbox.id,
           name: sandbox.name,
           code: sandbox.code,
           status: sandbox.status
         })}
      else
        err -> err
      end
    end
  end

  def handle_info({:update_sandbox, sandbox}, socket) do
    {:noreply,
     socket
     |> assign(:sandbox, %{
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
          Loading
        <% end %>
      </div>
    </div>
    """
  end
end
