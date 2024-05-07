defmodule ElodinDashboardWeb.OnboardingLive do
  use ElodinDashboardWeb, :live_view
  alias Elodin.Types.Api
  alias ElodinDashboard.Atc
  import ElodinDashboardWeb.CoreComponents
  import ElodinDashboardWeb.IconComponents

  def mount(params, _, socket) do
    current_user = socket.assigns[:current_user]

    {max_page, min_page} =
      IO.inspect(
        cond do
          current_user["billing_account_id"] == nil ->
            {0, 0}

          current_user["onboarding_data"] == nil ->
            {1, 1}

          true ->
            {1000, 2}
        end
      )

    param_page =
      case Integer.parse(params["page_num"] || "2") do
        {param_page, _} -> param_page
        :error -> 2
      end

    page = max(min(param_page, max_page), min_page)

    if param_page != page do
      {:ok,
       socket
       |> push_redirect(to: "/onboard/#{page}")}
    else
      {:ok,
       socket
       |> assign(page: page)
       |> assign(selected_tier: 0)
       |> assign(loading: false)
       |> assign(selected_industries: MapSet.new())}
    end
  end

  def handle_event("select_tier", %{"tier" => tier}, socket) do
    {:noreply, assign(socket, :selected_tier, tier)}
  end

  def handle_event("select_industry", %{"id" => id}, socket) do
    selected_industries =
      if MapSet.member?(socket.assigns[:selected_industries], id) do
        socket.assigns[:selected_industries] |> MapSet.delete(id)
      else
        selected_industries = socket.assigns[:selected_industries] |> MapSet.put(id)
      end

    {:noreply, assign(socket, :selected_industries, selected_industries)}
  end

  def handle_event("start_trial", %{"tier" => tier}, socket) do
    send(self(), {:start_trial, tier})
    {:noreply, assign(socket, :loading, true)}
  end

  def handle_event("next_page", _, socket) do
    page = socket.assigns[:page] + 1
    {:noreply, push_redirect(socket, to: "/onboard/#{page}")}
  end

  def handle_event("poll_results", %{"selected_industries" => selected_industries}, socket) do
    send(self(), {:poll_results, selected_industries})
    {:noreply, assign(socket, :loading, true)}
  end

  def handle_event("select_example", %{"example" => example}, socket) do
    {:noreply, socket |> push_redirect(to: "/example#{example}")}
  end

  def handle_info({:start_trial, tier}, socket) do
    Atc.create_billing_account(
      %Api.CreateBillingAccountReq{trial_license_type: tier, name: "Default Account"},
      socket.assigns[:current_user]["token"]
    )

    page = socket.assigns[:page] + 1
    {:noreply, socket |> push_redirect(to: "/onboard/#{page}") |> assign(:loading, false)}
  end

  def handle_info({:poll_results, selected_industries}, socket) do
    Atc.update_user(
      %Api.UpdateUserReq{
        onboarding_data: %Api.OnboardingData{
          usecases: Enum.to_list(selected_industries)
        }
      },
      socket.assigns[:current_user]["token"]
    )

    {:noreply, socket |> assign(:page, 2) |> assign(:loading, false)}
  end

  def pricing_tier(assigns) do
    ~H"""
    <div
      class={[
        if(@selected, do: "bg-opacity-100", else: "bg-opacity-20 hover:bg-opacity-50"),
        "w-40 h-24 flex flex-col items-center justify-center rounded-elo-sm  transition-all",
        "border border-solid  active:bg-opacity-100",
        @class
      ]}
      phx-click={JS.push("select_tier", value: %{tier: @tier})}
    >
      <h4 class="font-medium"><%= @label %></h4>
      <div class="text-xs text-mono"><%= @price %></div>
    </div>
    """
  end

  def poll_item(assigns) do
    ~H"""
    <div
      class={[
        if(@selected,
          do: "bg-opacity-60 border-opacity-70",
          else: "bg-opacity-10 hover:bg-opacity-20"
        ),
        "w-32 h-16 flex flex-col items-center justify-center rounded-elo-sm  transition-all",
        "border border-solid active:bg-opacity-60 active:border-opacity-70",
        @class
      ]}
      phx-click={JS.push("select_industry", value: %{id: @id})}
    >
      <h4 class="font-medium"><%= @label %></h4>
    </div>
    """
  end

  def code(assigns) do
    ~H"""
    <div class="rounded-elo-sm border border-green border-opacity-30 border-solid mt-6 p-6 bg-green bg-opacity-10 text-green flex items-center">
      <pre class="inline"><span class="select-none">$ </span><span class="select-all" id="download-code"><%= @code %></span></pre>
      <svg
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
        stroke-width="1.5"
        stroke="currentColor"
        class="w-5 h-5 inline ml-2 hover:text-white transition-all cursor-pointer"
        phx-click={JS.dispatch("phx:copy-inner", to: "\#download-code")}
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          d="M15.75 17.25v3.375c0 .621-.504 1.125-1.125 1.125h-9.75a1.125 1.125 0 0 1-1.125-1.125V7.875c0-.621.504-1.125 1.125-1.125H6.75a9.06 9.06 0 0 1 1.5.124m7.5 10.376h3.375c.621 0 1.125-.504 1.125-1.125V11.25c0-4.46-3.243-8.161-7.5-8.876a9.06 9.06 0 0 0-1.5-.124H9.375c-.621 0-1.125.504-1.125 1.125v3.5m7.5 10.375H9.375a1.125 1.125 0 0 1-1.125-1.125v-9.25m12 6.625v-1.875a3.375 3.375 0 0 0-3.375-3.375h-1.5a1.125 1.125 0 0 1-1.125-1.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H9.75"
        />
      </svg>
    </div>
    """
  end

  def template_button(assigns) do
    ~H"""
    <div class={[
      "bg-opacity-10 hover:bg-opacity-20",
      "w-32 h-16 flex flex-col items-center justify-center rounded-elo-sm  transition-all",
      "border border-solid active:bg-opacity-60 active:border-opacity-70",
      @class
    ]}>
      <h4 class="font-medium flex items-center">
        <%= render_slot(@inner_block) %>
      </h4>
    </div>
    """
  end

  def page(%{page: 0} = assigns) do
    ~H"""
    <div class="text-xl font-medium text-center">Welcome to Elodin!</div>
    <div class="text-md text-center">
      Please select which Elodin tier you want to try free for 15 days
    </div>
    <div class="flex gap-6 mt-8">
      <.pricing_tier
        label="Non Commercial"
        price="$50/month"
        class="bg-blue border-blue"
        selected={@selected_tier == 1}
        tier={1}
      />
      <.pricing_tier
        label="Commercial"
        price="$500/month"
        class="bg-yellow border-yellow"
        selected={@selected_tier == 2}
        tier={2}
      />
    </div>
    <.button
      class={[
        "mt-8 px-6 py-4 transition-all",
        if(@selected_tier == 0, do: "opacity-20", else: "")
      ]}
      disabled={@selected_tier == 0}
      phx-click={if @selected_tier != 0, do: JS.push("start_trial", value: %{tier: @selected_tier})}
    >
      <%= if @loading do %>
        <.spinner class="animate-spin w-6 h-6" />
      <% else %>
        Start Trial
      <% end %>
    </.button>
    """
  end

  def page(%{page: 1} = assigns) do
    ~H"""
    <div class="text-xl font-medium text-center">Use Cases</div>
    <div class="text-md text-center">
      Please select what you are interested in using Elodin for
    </div>
    <div class="flex gap-6 mt-8">
      <.poll_item
        label="Space"
        class="bg-blue border-blue"
        selected={@selected_industries |> Enum.member?("space")}
        id="space"
      />
      <.poll_item
        label="Drones"
        class="bg-slate border-slate"
        selected={@selected_industries |> Enum.member?("drones")}
        id="drones"
      />
      <.poll_item
        label="Defense"
        class="bg-orange border-orange"
        selected={@selected_industries |> Enum.member?("defense")}
        id="defense"
      />
    </div>
    <.button
      class={[
        "mt-8 px-6 py-4 transition-all"
      ]}
      phx-click={
        JS.push("poll_results", value: %{selected_industries: Enum.to_list(@selected_industries)})
      }
    >
      <%= if @loading do %>
        <.spinner class="animate-spin w-8 h-4" />
      <% else %>
        Continue
      <% end %>
    </.button>
    """
  end

  def page(%{page: 2} = assigns) do
    ~H"""
    <div class="text-xl font-medium text-center">Congrats on signing up for Elodin!</div>
    <div class="text-md text-center text-crema text-opacity-70 mt-2">
      To get started using Elodin copy and paste the following command into your terminal
    </div>
    <.code code="curl sh.elodin.systems | sh" />
    <div class="text-sm text-opacity-70 text-white mt-4 w-full pl-10">
      Use Windows?
      <a class="text-green text-opacity-70" href="https://docs.elodin.systems/quickstart">
        Visit our docs to install
      </a>
    </div>
    <.button
      class={[
        "mt-8 px-6 py-4 transition-all"
      ]}
      phx-click={JS.push("next_page")}
    >
      Next Steps
    </.button>
    """
  end

  def page(%{page: 3} = assigns) do
    ~H"""
    <div class="text-xl font-medium text-center">Login to Elodin</div>
    <div class="text-md text-center text-crema text-opacity-70 mt-2">
      Before using the CLI you need to login
    </div>
    <.code code="elodin login"></.code>
    <.button
      class={[
        "mt-8 px-6 py-4 transition-all"
      ]}
      phx-click={JS.push("next_page")}
    >
      Continue
    </.button>
    """
  end

  def page(%{page: 4} = assigns) do
    ~H"""
    <div class="text-xl font-medium text-center">Get Started with a Template</div>
    <div class="text-md text-center text-crema text-opacity-70 mt-2">
      Download one of our templates to get started
    </div>
    <div class="flex gap-6 mt-8">
      <.template_button class="bg-blue border-blue">🛰️ Cube Sat</.template_button>
      <.template_button class="bg-red border-red">🚀 Rocket</.template_button>
    </div>
    <.code code="elodin run example.py"></.code>
    <.button
      class={[
        "mt-8 px-6 py-4 transition-all"
      ]}
      phx-click={JS.push("next_page")}
    >
      Continue
    </.button>
    """
  end

  def page(%{page: 5} = assigns) do
    ~H"""
    <div class="text-xl font-medium text-center">Run a Monte Carlo Sim!</div>
    <div class="text-md text-center text-crema text-opacity-70 mt-2">
      Now you can try and run that sample example in a monte-carlo simulation
    </div>
    <.code code="elodin monte-carlo example.py"></.code>
    <.button
      class={[
        "mt-8 px-6 py-4 transition-all"
      ]}
      phx-click={JS.push("next_page")}
    >
      Continue
    </.button>
    """
  end

  def page(%{page: 6} = assigns) do
    ~H"""
    <div class="text-xl font-medium text-center">Congrats! You are all done</div>
    <div class="text-md text-center text-crema text-opacity-70 mt-2">
      Check out our docs, or head straight to the dashboard
    </div>

    <div
      phx-hook="FireworksHook"
      id="fireworks"
      style="width: 100vw; height: 100vh; position: absolute; top: 0; left: 0; pointer-events: none; z-index: 10;"
    >
    </div>

    <div class="flex gap-6 mt-8">
      <.link href="https://docs.elodin.systems">
        <.template_button class="bg-blue border-blue text-blue">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            stroke-width="1.5"
            stroke="currentColor"
            class="w-4 h-4 mr-1"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z"
            />
          </svg>
          Docs
        </.template_button>
      </.link>
      <.link patch={~p"/"}>
        <.template_button class="bg-yellow border-yellow text-yellow">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="currentColor"
            class="w-4 h-4 mr-1"
          >
            <path d="M11.47 3.841a.75.75 0 0 1 1.06 0l8.69 8.69a.75.75 0 1 0 1.06-1.061l-8.689-8.69a2.25 2.25 0 0 0-3.182 0l-8.69 8.69a.75.75 0 1 0 1.061 1.06l8.69-8.689Z" />
            <path d="m12 5.432 8.159 8.159c.03.03.06.058.091.086v6.198c0 1.035-.84 1.875-1.875 1.875H15a.75.75 0 0 1-.75-.75v-4.5a.75.75 0 0 0-.75-.75h-3a.75.75 0 0 0-.75.75V21a.75.75 0 0 1-.75.75H5.625a1.875 1.875 0 0 1-1.875-1.875v-6.198a2.29 2.29 0 0 0 .091-.086L12 5.432Z" />
          </svg>
          Dashboard
        </.template_button>
      </.link>
    </div>
    """
  end

  def render(assigns) do
    ~H"""
    <div class="flex items-center w-full h-full justify-center bg-black-primary z-20">
      <div class="max-w-[600px] flex flex-col items-center p-16 rounded-elo-sm bg-black-secondary border border-white border-opacity-10 border-solid">
        <.page
          page={@page}
          selected_tier={@selected_tier}
          loading={@loading}
          selected_industries={@selected_industries}
        />
      </div>
    </div>
    """
  end
end
