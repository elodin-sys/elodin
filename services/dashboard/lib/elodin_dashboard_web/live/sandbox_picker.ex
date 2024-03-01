defmodule ElodinDashboardWeb.SandboxPickerLive do
  use ElodinDashboardWeb, :live_view
  alias Elodin.Types.Api
  alias ElodinDashboard.Atc
  alias ElodinDashboard.NameGen
  import ElodinDashboardWeb.CoreComponents
  import ElodinDashboardWeb.SandboxComponents
  import ElodinDashboardWeb.NavbarComponents
  import ElodinDashboardWeb.ModalComponents

  def mount(params, _, socket) do
    token = socket.assigns[:current_user]["token"]

    new_form = to_form(%{"name" => NameGen.generate()})

    case Atc.list_sandboxes(struct(Api.ListSandboxesReq), token) do
      {:ok, sandboxes} ->
        sandboxes =
          Enum.map(sandboxes.sandboxes, fn s ->
            %{id: UUID.binary_to_string!(s.id), name: s.name}
          end)

        {:ok,
         socket
         |> assign(:sandboxes, sandboxes)
         |> assign(:new_form, new_form)
         |> assign(:template, params["template"])
         |> assign(:is_onboarding, params["onboarding"] == "1")
         |> assign(:onboarding_step, 1)}

      {:error, err} ->
        {:ok,
         socket
         |> put_flash(:error, "Error creating sandbox: #{err}")
         |> assign(:sandboxes, [])
         |> assign(:new_form, new_form)
         |> assign(:template, params["template"])
         |> assign(:is_onboarding, false)
         |> assign(:onboarding_step, 1)}
    end
  end

  def handle_params(_, _, socket) do
    {:noreply, socket}
  end

  def handle_event("save", %{"name" => name}, socket) do
    token = socket.assigns[:current_user]["token"]
    template = socket.assigns[:template]

    case Atc.create_sandbox(%Api.CreateSandboxReq{name: name, template: template}, token) do
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

  def handle_event("change_page", %{"step" => step}, socket) do
    {:noreply, socket |> assign(:onboarding_step, step)}
  end

  def handle_event("select_template", %{"template" => template}, socket) do
    {:noreply, socket |> assign(:template, template)}
  end

  def show_template_new(js \\ %JS{}, template) do
    js |> JS.push("select_template", value: %{"template" => template}) |> show_modal("new")
  end

  def render(assigns) do
    ~H"""
    <.navbar_layout current_user={@current_user}>
      <:navbar_right>
        <.link patch={~p"/sandbox/new"} phx-click={show_modal("new")}>
          <.button type="link" type="outline" class="mr-1.5">
            Create New
          </.button>
        </.link>
      </:navbar_right>
      <div class="flex flex-col min-h-full items-start p-elo-lg bg-black-secondary">
        <div class="flex flex-col flex-wrap items-start w-full rounded-elo-xs bg-black-primary">
          <div class="p-elo-xl w-fit font-bold text-primative-colors-white-opacity-900 text-[14px] w-full">
            Elodin templates
          </div>
          <div class="inline-flex items-start px-elo-xl pb-elo-xl gap-elo-lg">
            <.sandbox_card
              name="3 Body Problem"
              img="/images/3-body-bg.svg"
              path={~p"/sandbox/new/three-body"}
              phx_click={show_template_new("three-body")}
            />
            <.sandbox_card
              name="Cube Sat"
              img="/images/cube-sat-bg.svg"
              path={~p"/sandbox/new/cube-sat"}
              phx_click={show_template_new("cube-sat")}
            />
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
              href={~p"/sandbox/#{sandbox.id}"}
            />
          </div>
        </div>
      </div>
    </.navbar_layout>

    <.guide_modal
      id="onboarding"
      bg_color={if @onboarding_step == 1, do: "hyper-blue", else: "surface-secondary"}
      show={@is_onboarding}
      step_count={4}
      cur_step={@onboarding_step}
      prev_page={JS.push("change_page", value: %{step: @onboarding_step - 1})}
      next_page={JS.push("change_page", value: %{step: @onboarding_step + 1})}
      on_cancel={JS.navigate(~p"/")}
    >
      <%= if @onboarding_step == 1 do %>
        <div class="mt-[64px]">
          <img src="/images/onboarding/elodin-logo.svg" />
        </div>

        <div class="mt-[14px]">
          <img src="/images/onboarding/alpha-logo.svg" />
        </div>

        <div class="mb-[18px] mt-[34px] h-[100px] w-[380px] flex flex-col text-center text-white">
          <div class="border-b border-l border-white-opacity-200">
            Welcome and thanks for trying our alpha demo.
          </div>
          <div class="border-b border-l border-r border-white-opacity-200">
            Check out the following screens to see what you
          </div>
          <div class="border-b border-l border-r border-white-opacity-200">
            can do, and what to look forward to!
          </div>
          <div class="h-full border-r border-white-opacity-200"></div>
        </div>

        <.button class="px-6 py-4" type="invert" phx-click={JS.push("change_page", value: %{step: 2})}>
          Continue
        </.button>
      <% end %>

      <%= if @onboarding_step == 2 do %>
        <div class="mt-[50px] text-2xl font-semibold text-white">Features in this version</div>

        <div class="mt-[24px] text-base font-normal text-white text-opacity-80">
          These are the things currently available to use in the Alpha.
        </div>

        <div class="mt-[78px] flex flex-row gap-[34px]">
          <div class="flex h-[158px] w-[200px] flex-col rounded-md border border-black">
            <div class="basis-3/4 overflow-hidden">
              <img src="/images/onboarding/1-1-python-code-editor.png" />
            </div>
            <div class="flex basis-1/4 items-center justify-center text-sm font-medium text-white">
              Python Code Editor
            </div>
          </div>

          <div class="flex h-[158px] w-[200px] flex-col rounded-md border border-black">
            <div class="basis-3/4 overflow-hidden">
              <img src="/images/onboarding/1-2-3d-sim-output.svg" />
            </div>
            <div class="flex basis-1/4 items-center justify-center text-sm font-medium text-white">
              3D Sim Output
            </div>
          </div>

          <%!-- <div class="flex h-[158px] w-[200px] flex-col rounded-md border border-black">
            <div class="basis-3/4 overflow-hidden">
              <img src="/images/onboarding/1-3-sim-timeline.svg" />
            </div>
            <div class="flex basis-1/4 items-center justify-center text-sm font-medium text-white">
              Sim Timeline
            </div>
          </div> --%>
        </div>
      <% end %>

      <%= if @onboarding_step == 3 do %>
        <div class="mt-[50px] text-2xl font-semibold text-white">Features coming soon</div>

        <div class="mt-[24px] text-base font-normal text-white text-opacity-80">
          Some of the larger features that will be available at a later date.
        </div>

        <div class="mt-[78px] flex flex-row gap-[34px]">
          <div class="flex h-[158px] w-[200px] flex-col rounded-md border border-black">
            <div class="basis-3/5 overflow-hidden">
              <img src="/images/onboarding/2-1-monte-carlo-runs.svg" />
            </div>
            <div class="flex basis-2/5 items-center justify-center text-sm font-medium text-white">
              Monte Carlo Runs
            </div>
          </div>

          <div class="flex h-[158px] w-[200px] flex-col rounded-md border border-black">
            <div class="basis-3/5 overflow-hidden">
              <img src="/images/onboarding/2-2-gpu-scaling.svg" />
            </div>
            <div class="flex basis-2/5 items-center justify-center text-sm font-medium text-white">
              GPU Scaling
            </div>
          </div>

          <div class="flex h-[158px] w-[200px] flex-col rounded-md border border-black">
            <div class="basis-3/5 overflow-hidden">
              <img src="/images/onboarding/2-3-dockerized-flight-software.svg" />
            </div>
            <div class="flex basis-2/5 items-center justify-center text-center text-sm font-medium text-white">
              Dockerized <br /> Flight Software
            </div>
          </div>
        </div>
      <% end %>

      <%= if @onboarding_step == 4 do %>
        <div class="mt-[50px] text-2xl font-semibold text-white">
          Resources
        </div>

        <div class="mt-[24px] text-base font-normal text-white text-opacity-80">
          Feel free to dig into our documentation and join our Discord for support!
        </div>

        <div class="mt-[78px] w-[266px] flex flex-col gap-3">
          <.link href="https://docs.elodin.systems">
            <.button type="outline" class="w-full py-4">API Documentation</.button>
          </.link>

          <.link href="https://discord.gg/agvGJaZXy5">
            <.button type="outline" class="w-full py-4">Discord</.button>
          </.link>

          <.link href="https://www.youtube.com/watch?v=UWPzF0JFgOA">
            <.button type="outline" class="w-full py-4">Video Demo</.button>
          </.link>
        </div>
      <% end %>
    </.guide_modal>

    <.modal id="new" show={@live_action == :new} on_cancel={JS.navigate(~p"/")}>
      <h2 class="font-semibold absolute top-elo-xl left-elo-xl ">New Sandbox</h2>
      <.form
        for={@new_form}
        phx-submit="save"
        class="flex justify-center align-center flex-col gap-elo-xl mt-elo-xl"
      >
        <.input name="name" label="Name" field={@new_form[:name]} autocomplete="false" data-1p-ignore />
        <.button class="">New Sandbox</.button>
      </.form>
    </.modal>
    """
  end
end
