defmodule ParacosmDashboardWeb.SandboxPickerLive do
  use ParacosmDashboardWeb, :live_view
  alias Paracosm.Types.Api
  alias ParacosmDashboard.Atc
  alias ParacosmDashboard.NameGen
  import ParacosmDashboardWeb.CoreComponents
  import ParacosmDashboardWeb.SandboxComponents

  def mount(_, _, socket) do
    token = socket.assigns[:current_user]["token"]

    case Atc.list_sandboxes(Api.ListSandboxesReq.new(), token) do
      {:ok, sandboxes} ->
        sandboxes =
          Enum.map(sandboxes.sandboxes, fn s ->
            %{id: UUID.binary_to_string!(s.id), name: s.name}
          end)

        {:ok,
         socket
         |> assign(:sandboxes, sandboxes)
         |> assign(:new_form, to_form(%{"name" => NameGen.generate()}))}

      {:err, err} ->
        {:ok,
         socket
         |> put_flash(:error, "Error creating sandbox: #{err}")
         |> assign(:sandboxes, [])
         |> assign(:new_form, to_form(%{"name" => NameGen.generate()}))}
    end
  end

  def handle_params(_, _, socket) do
    {:noreply, socket}
  end

  def handle_event("save", %{"name" => name}, socket) do
    token = socket.assigns[:current_user]["token"]

    case Atc.create_sandbox(Api.CreateSandboxReq.new(name: name), token) do
      {:ok, sandbox} ->
        id = UUID.binary_to_string!(sandbox.id)

        {:noreply,
         socket
         |> put_flash(:info, "Successfully created sandbox #{name}")
         |> redirect(to: ~p"/sandbox/#{id}")}

      err ->
        {:noreply, socket |> put_flash(:error, "Error creating sandbox: #{err}")}
    end
  end

  def render(assigns) do
    ~H"""
    <div class="flex flex-col min-h-full items-start p-elo-lg bg-tokens-surface-secondary">
      <div class="flex flex-col flex-wrap items-start w-full bg-primative-colors-white-opacity-50 rounded-elo-xs ">
        <div class="p-elo-lg border-b border-primative-colors-white-opacity-100 w-fit mt-[-1.00px] font-bold text-primative-colors-white-opacity-900 text-[14px] w-full">
          Elodin sandboxes
        </div>
        <div class="inline-flex items-start p-elo-lg gap-elo-lg">
          <.sandbox_card name="Double Pendulum" img="/images/double-pend-bg.svg" />
          <.sandbox_card name="3 Body Problem" img="/images/3-body-bg.svg" />
        </div>
        <div class="flex flex-wrap flex-col items-start self-stretch w-full flex-[0_0_auto] bg-primative-colors-white-opacity-50 rounded-elo-xs ">
          <div class="p-elo-lg border-b border-primative-colors-white-opacity-100 w-fit mt-[-1.00px] font-bold text-primative-colors-white-opacity-900 text-[14px] w-full">
            Your sandboxes
          </div>
          <div class="inline-flex items-start p-elo-lg flex-wrap gap-elo-lg">
            <.sandbox_card
              :for={sandbox <- @sandboxes}
              name={sandbox.name}
              img="/images/blue-circle-8.svg"
              path={~p"/sandbox/#{sandbox.id}"}
            />
            <.sandbox_card
              name="Create New"
              img="/images/blue-circle-8.svg"
              path={~p"/sandbox/new"}
              phx_click={show_modal("new")}
            />
          </div>
        </div>
      </div>
    </div>

    <.modal id="new" show={@live_action == :new} on_cancel={JS.navigate(~p"/")}>
      <.form
        for={@new_form}
        phx-submit="save"
        class="flex justify-center align-center flex-col mx-elo-lg gap-elo-xl"
      >
        <.input name="name" label="Name" field={@new_form[:name]} />
        <.button class="">New Sandbox</.button>
      </.form>
    </.modal>
    """
  end
end
