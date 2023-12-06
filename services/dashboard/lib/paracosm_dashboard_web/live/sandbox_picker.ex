defmodule ParacosmDashboardWeb.SandboxPickerLive do
  use ParacosmDashboardWeb, :live_view
  alias Paracosm.Types.Api
  alias ParacosmDashboard.Atc
  alias ParacosmDashboard.NameGen
  import ParacosmDashboardWeb.CoreComponents
  import ParacosmDashboardWeb.SandboxComponents
  import ParacosmDashboardWeb.NavbarComponents

  def mount(_, _, socket) do
    token = socket.assigns[:current_user]["token"]

    case Atc.list_sandboxes(struct(Api.ListSandboxesReq), token) do
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

    case Atc.create_sandbox(%Api.CreateSandboxReq{name: name}, token) do
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
    <.navbar_layout current_user={@current_user}>
      <:navbar_right>
        <.link patch={~p"/sandbox/new"} phx-click={show_modal("new")}>
          <.button type="link" class="mr-1.5">
            Create New
          </.button>
        </.link>
      </:navbar_right>
      <div class="flex flex-col min-h-full items-start p-elo-lg bg-surface-secondary">
        <div class="flex flex-col flex-wrap items-start w-full bg-primative-colors-white-opacity-50 rounded-elo-sm bg-primative-colors-white-opacity-50">
          <div class="p-elo-xl w-fit font-bold text-primative-colors-white-opacity-900 text-[14px] w-full">
            Elodin sandboxes
          </div>
          <div class="inline-flex items-start px-elo-xl pb-elo-xl gap-elo-lg">
            <.sandbox_card name="Double Pendulum" img="/images/double-pend-bg.svg" />
            <.sandbox_card name="3 Body Problem" img="/images/3-body-bg.svg" />
          </div>
        </div>
        <div class="flex flex-wrap flex-col items-start self-stretch w-full flex-[0_0_auto] rounded-elo-xs ">
          <div class="py-elo-xl w-fit font-bold text-primative-colors-white-opacity-900 text-[14px] w-full">
            Your sandboxes
          </div>
          <div class="inline-flex items-start flex-wrap gap-elo-lg">
            <.sandbox_card
              :for={sandbox <- @sandboxes}
              name={sandbox.name}
              img="/images/blue-circle-8.svg"
              path={~p"/sandbox/#{sandbox.id}"}
            />
          </div>
        </div>
      </div>
    </.navbar_layout>

    <.modal id="new" show={@live_action == :new} on_cancel={JS.navigate(~p"/")}>
      <h2 class="font-semibold absolute top-elo-xl left-elo-xl ">New Sandbox</h2>
      <.form
        for={@new_form}
        phx-submit="save"
        class="flex justify-center align-center flex-col gap-elo-xl mt-elo-xl"
      >
        <.input name="name" label="Name" field={@new_form[:name]} />
        <.button class="">New Sandbox</.button>
      </.form>
    </.modal>
    """
  end
end
