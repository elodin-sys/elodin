defmodule ElodinDashboardWeb.OnboardingLive do
  use ElodinDashboardWeb, :live_view
  alias Elodin.Types.Api
  alias ElodinDashboard.Atc
  import ElodinDashboardWeb.CoreComponents
  import ElodinDashboardWeb.IconComponents
  import ElodinDashboardWeb.NavbarComponents

  def mount(params, _, socket) do
    current_user = socket.assigns[:current_user]

    {max_page, min_page} =
      IO.inspect(
        cond do
          current_user["billing_account_id"] == nil ->
            {0, 0}

          current_user["onboarding_data"] == nil ->
            {2, 1}

          true ->
            {1000, 3}
        end
      )

    param_page =
      case Integer.parse(params["page_num"] || "3") do
        {param_page, _} -> param_page
        :error -> 3
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
       |> assign(loading: false)
       |> assign(selected_industries: MapSet.new())}
    end
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
    {:noreply, assign(socket, :loading, tier)}
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

    page = 1
    current_user = socket.assigns[:current_user]
    current_user = Map.put(current_user, "billing_account_id", "tmp")

    {:noreply,
     socket
     |> push_redirect(to: "/onboard/#{page}")
     |> assign(:loading, false)
     |> assign(:current_user, current_user)}
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

    {:noreply, socket |> assign(:page, 3) |> assign(:loading, false)}
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

  attr(:class, :string, default: "")
  attr(:title_class, :string, default: "")
  slot :icon

  def code(assigns) do
    ~H"""
    <div class={[
      "border-t border-b border-onyx-9 border-opacity-30 border-solid mx-[-40px] mt-6 py-6 px-10 flex items-center",
      "h-[90px]",
      @class
    ]}>
      <div>
        <%= render_slot(@icon) %>
      </div>
      <div class="flex flex-col w-full">
        <div class={["flex w-full font-mono text-green text-sm font-medium", @title_class]}>
          <%= @title %>
        </div>
        <div class="flex w-full items-center justify-between">
          <pre class="inline"><span class="select-none">$ </span><span class="select-all" id="download-code"><%= @code %></span></pre>
          <svg
            width="19"
            height="18"
            viewBox="0 0 19 18"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            class="hover:opacity-75 transition-all cursor-pointer"
            phx-click={JS.dispatch("phx:copy-inner", to: "\#download-code")}
          >
            <g clip-path="url(#clip0_2462_13322)">
              <rect x="2.5" y="6" width="10" height="10" stroke="#FFFBF0" stroke-width="2" />
              <path d="M4.5 2H16.5V14" stroke="#FFFBF0" stroke-width="2" />
            </g>
            <defs>
              <clipPath id="clip0_2462_13322">
                <rect width="18" height="18" fill="white" transform="translate(0.5)" />
              </clipPath>
            </defs>
          </svg>
        </div>
      </div>
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

  def checkbox(assigns) do
    ~H"""
    <div class={["w-4 h-4 border border-solid border-grey rounded-elo-xxs inline-block mr-4", @class]}>
      <svg width="16" height="16" viewBox="1 1 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M11 6L7 10L5 8" stroke="currentColor" stroke-width="2" stroke-linecap="square" />
      </svg>
    </div>
    """
  end

  def link_arrow(assigns) do
    ~H"""
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">
      <g clip-path="url(#clip0_2563_17756)">
        <path
          fill-rule="evenodd"
          clip-rule="evenodd"
          d="M4.5 4.5L13.5533 4.5L13.5533 13.5L12.0533 13.5L12.0533 6L4.5 6V4.5Z"
          fill="#FFFBF0"
        />
        <rect
          x="11.6519"
          y="5.25"
          width="1.5"
          height="9.49847"
          transform="rotate(45 11.6519 5.25)"
          fill="#FFFBF0"
        />
      </g>
      <defs>
        <clipPath id="clip0_2563_17756">
          <rect width="18" height="18" fill="white" />
        </clipPath>
      </defs>
    </svg>
    """
  end

  attr(:button_text, :string, default: "START FREE 15 DAY TRIAL")
  attr(:per_month, :boolean, default: true)
  attr(:tier, :integer, default: nil)

  def pricebox(assigns) do
    ~H"""
    <div class="w-[418px] bg-onyx p-10 rounded-elo-xs border border-solid border-crema border-opacity-10 flex-col">
      <div class="text-4xl font-space">
        <%= @title %>
      </div>
      <div class={[
        "font-mono tracking-elo-mono-small font-medium text-sm w-full py-2",
        @highlight_class
      ]}>
        <%= @users %>
      </div>
      <div class="text-crema-60 text-lg w-full py-2">
        <%= @subtitle %>
      </div>
      <div class="text-3xl w-full py-2 font-medium font-mono">
        <%= @price %><span :if={@per_month} class="text-xl">/month</span>
      </div>
      <hr class="border-onyx-9 my-8" />
      <div class="font-mono text-crema tracking-elo-mono-small font-medium text-sm w-full py-2">
        INCLUDES
      </div>
      <ul class="font-medium space-y-2">
        <li :for={feature <- @features} class="flex items-center">
          <.checkbox class={@highlight_class} /> <%= feature %>
        </li>
      </ul>
      <%= if @tier do %>
        <.button
          class={["w-full mt-8 py-6 flex justify-center", @button_class]}
          phx-click={JS.push("start_trial", value: %{tier: @tier})}
        >
          <%= if @loading == @tier do %>
            <.spinner class={["animate-spin w-8 h-4 my-[-4px] m-0", @highlight_class]} />
          <% else %>
            <%= @button_text %>
          <% end %>
        </.button>
      <% else %>
        <.button class={["w-full mt-8 py-6 flex justify-center", @button_class]}>
          <%= @button_text %>
        </.button>
      <% end %>
    </div>
    """
  end

  def questions(assigns) do
    ~H"""
    <div class="w-full flex-col flex">
      <div :for={question <- @questions} class="w-full mt-20 flex items-center">
        <div class={[
          "rounded-full w-12 h-12 flex items-center justify-center font-bold font-mono text-xl",
          if(question.index <= @selected,
            do: "text-bone bg-primary-smoke",
            else: "text-primary-smoke border-primary-smoke border"
          )
        ]}>
          <%= question.number %>
        </div>
        <div style="position:relative;">
          <%= if question.index < @selected do %>
            <div style="width: 2px; height: 85px; background: #0D0D0D; position: absolute; left: -25.5px; top: 23px;">
            </div>
          <% end %>
          <%= if question.index == @selected && @selected != Enum.count(@questions) do %>
            <svg
              width="16"
              height="44"
              viewBox="0 0 16 44"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              style="position: absolute; left: -31.5px; top: 23px;"
            >
              <path
                d="M7.29289 43.7071C7.68342 44.0976 8.31658 44.0976 8.70711 43.7071L15.0711 37.3431C15.4616 36.9526 15.4616 36.3195 15.0711 35.9289C14.6805 35.5384 14.0474 35.5384 13.6568 35.9289L8 41.5858L2.34314 35.9289C1.95261 35.5384 1.31945 35.5384 0.928926 35.9289C0.538402 36.3195 0.538402 36.9526 0.928927 37.3432L7.29289 43.7071ZM6.99997 6.41874e-07L7 43L9 43L8.99997 -6.41876e-07L6.99997 6.41874e-07Z"
                fill="#0D0D0D"
              />
            </svg>
          <% end %>
        </div>
        <div class="flex flex-col text-primary-smoke ml-8">
          <div class="font-mono text-sm font-medium">
            <%= question.title %>
          </div>
          <div>
            <%= question.prompt %>
          </div>
        </div>
      </div>
    </div>
    """
  end

  def poll_selector(assigns) do
    ~H"""
    <.onboard_box
      heading={"#{@number} / Questions"}
      title={@prompt}
      action="This helps us greatly. Please check all that apply."
    >
      <div class="w-full font-mono mt-8">
        <div
          :for={{item, i} <- @items |> Enum.with_index()}
          class={[
            "h-[72px] flex items-center mx-[-40px] px-10",
            "bg-opacity-0 hover:bg-opacity-20 hover:border-b-primary-smoke",
            "cursor-pointer text-sm font-medium",
            item.class,
            if(@selected_industries |> Enum.member?(item.id), do: item.selected_class || "", else: ""),
            if(@selected_industries |> Enum.member?(item.id),
              do: "!bg-opacity-100",
              else: "border-b border-onyx-9"
            ),
            if(i == 0 && !(@selected_industries |> Enum.member?(item.id)), do: "border-t", else: "")
          ]}
          phx-click={JS.push("select_industry", value: %{id: item.id})}
        >
          <%= item.name |> String.upcase() %>
        </div>
      </div>
      <:button>
        <div class="flex justify-between items-center">
          <div>
            <.button
              class="!py-4 !px-8"
              type="crema"
              phx-click={
                if(@number == "02",
                  do:
                    JS.push("poll_results",
                      value: %{selected_industries: Enum.to_list(@selected_industries)}
                    ),
                  else: JS.push("next_page")
                )
              }
            >
              <%= if @loading do %>
                <.spinner class={["animate-spin w-8 h-4 my-[-4px] m-0"]} />
              <% else %>
                Next
              <% end %>
            </.button>
          </div>
          <div class="text-mono font-medium text-sm">
            <%= @number %> / 02
          </div>
        </div>
      </:button>
    </.onboard_box>
    """
  end

  slot :inner_block, required: true
  slot :button, required: false

  def onboard_box(assigns) do
    ~H"""
    <div class="flex flex-col bg-primary-smoke w-[556px] h-[662px] p-10 border border-crema rounded-elo-xs border-opacity-5 justify-between">
      <div class="flex flex-col">
        <div class="text-onyx-9 text-sm font-mono font-medium">
          <%= @heading %>
        </div>
        <div class="font-space text-xl font-medium mt-2">
          <%= @title %>
        </div>
        <div class="text-md text-crema-60 mt-2">
          <%= @action %>
        </div>
        <%= render_slot(@inner_block) %>
      </div>
      <div class="flex">
        <div class="w-full font-mono mt-8">
          <%= render_slot(@button) %>
        </div>
      </div>
    </div>
    """
  end

  def default_onboard_questions() do
    [
      %{
        index: 1,
        number: "01",
        title: "QUESTIONS",
        prompt: "Tell us a bit about yourself."
      },
      %{
        index: 2,
        number: "02",
        title: "INSTALL CLI",
        prompt: "Get the native app running."
      },
      %{
        index: 3,
        number: "03",
        title: "DOWNLOAD TEMPLATE(S)",
        prompt: "Download one of our templates via the CLI."
      },
      %{
        index: 4,
        number: "04",
        title: "RUN MONTE CARLO SIM",
        prompt: "See your template in action."
      }
    ]
  end

  def page(%{page: 0} = assigns) do
    ~H"""
    <div class="w-full flex min-h-full px-20 p-6 bg-primary-smoke align-stretch flex-col">
      <div class="w-full flex h-16 justify-between">
        <.link href="/">
          <img src="/images/o-logo.svg" class="w-5" />
        </.link>
        <img
          src={@current_user["avatar"]}
          class="ml-elo-lg w-8 h-8 inline-block rounded-full"
          phx-click={toggle("#user_dropdown")}
        />
        <.user_dropdown current_user={@current_user} />
      </div>
      <div class="w-full flex flex-col">
        <div class="font-mono text-peach tracking-elo-mono-small font-medium text-sm w-full text-center">
          PRICING
        </div>
        <div class="text-5xl w-full text-center mt-4 font-space">
          Select your Plan
        </div>
        <div class="text-xl w-full text-center mt-4 text-crema-60">
          Try for free (no credit card required)
        </div>
        <div class="w-full mt-16 flex justify-center gap-8 flex-wrap">
          <.pricebox
            title="Non-Commercial"
            subtitle="Perfect for students and hackers getting started on a new project."
            users="INDIVIDUALS / STUDENTS"
            price="$50"
            highlight_class="border-green text-green"
            button_class="bg-green border-green"
            features={["60 mins of Compute Credits", "Non-Commercial Usage", "Community Support"]}
            tier={1}
            loading={@loading}
          />
          <.pricebox
            title="Commercial"
            subtitle="For the startups and innovators of the next-generation of aerospace companies."
            users="BUSINESSES"
            price="$500"
            highlight_class="text-yellow"
            button_class="bg-yellow border-yellow text-yellow"
            features={["60,000 mins of Compute Credits", "Commercial Usage", "Advanced Support"]}
            tier={2}
            loading={@loading}
          />
          <.pricebox
            title="Enterprise"
            subtitle="Large organizations that need customized solutions, and dedicated support."
            users="LARGE ORG / CORPORATIONS"
            price="Custom"
            per_month={false}
            highlight_class="text-orange"
            button_class="bg-orange border-orange text-orange"
            button_text="CONTACT SALES"
            features={["SSO", "Self Hosted", "ITAR Compliance"]}
            loading={@loading}
          />
        </div>
      </div>
    </div>
    """
  end

  def page(%{page: 1} = assigns) do
    ~H"""
    <div class="w-full flex min-h-full bg-onyx max-lg:flex-col items-stretch">
      <div class="lg:w-1/2 max-lg:w-full bg-bone p-6 px-20 flex flex-col">
        <div class="w-full flex align-stretch">
          <.link href="/" style="color: #000;">
            <.ologo class="w-5" />
          </.link>
        </div>
        <div class="w-full text-sm font-mono text-black mt-16 ">
          GETTING STARTED
        </div>
        <div class="w-full text-2xl text-space text-black font-medium mt-2">
          Introduction to Elodin
        </div>
        <.questions questions={default_onboard_questions()} selected={1} />
      </div>
      <div class="lg:w-1/2 max-lg:w-full flex flex-col px-20 p-6 bg-onyx">
        <div class="w-full flex flex-col items-end">
          <img
            src={@current_user["avatar"]}
            class="ml-elo-lg w-8 h-8 inline-block rounded-full"
            phx-click={toggle("#user_dropdown")}
          /> <.user_dropdown current_user={@current_user} />
        </div>
        <div class="flex w-full justify-center items-center mt-12">
          <.poll_selector
            number="01"
            prompt="What are you working on?"
            selected_industries={@selected_industries}
            loading={@loading}
            items={[
              %{
                id: "drones",
                name: "Drones",
                selected_class: "text-black",
                class: "bg-yellow"
              },
              %{
                id: "rockets",
                name: "Rockets",
                selected_class: "text-black",
                class: "bg-red"
              },
              %{
                id: "space",
                name: "Space",
                selected_class: "text-black",
                class: "bg-slate"
              },
              %{
                id: "robots",
                name: "Robots",
                selected_class: "text-black",
                class: "bg-orange"
              },
              %{
                id: "other",
                name: "Other",
                selected_class: "text-black",
                class: "bg-green"
              }
            ]}
          />
        </div>
      </div>
    </div>
    """
  end

  def page(%{page: 2} = assigns) do
    ~H"""
    <div class="w-full flex min-h-full bg-onyx max-lg:flex-col items-stretch">
      <div class="lg:w-1/2 max-lg:w-full bg-bone p-6 px-20 flex flex-col">
        <div class="w-full flex align-stretch">
          <.link href="/" style="color: #000;">
            <.ologo class="w-5" />
          </.link>
        </div>
        <div class="w-full text-sm font-mono text-black mt-16 ">
          GETTING STARTED
        </div>
        <div class="w-full text-2xl text-space text-black font-medium mt-2">
          Introduction to Elodin
        </div>
        <.questions questions={default_onboard_questions()} selected={1} />
      </div>
      <div class="lg:w-1/2 max-lg:w-full flex flex-col px-20 p-6 bg-onyx">
        <div class="w-full flex flex-col items-end">
          <img
            src={@current_user["avatar"]}
            class="ml-elo-lg w-8 h-8 inline-block rounded-full"
            phx-click={toggle("#user_dropdown")}
          /> <.user_dropdown current_user={@current_user} />
        </div>
        <div class="flex w-full justify-center items-center mt-12">
          <.poll_selector
            number="02"
            prompt="What excites you about Elodin?"
            selected_industries={@selected_industries}
            loading={@loading}
            items={[
              %{
                id: "python-lib",
                name: "PHYSICS SIMULATION LIBRARY",
                selected_class: "text-black",
                class: "bg-yellow"
              },
              %{
                id: "3d-viewer",
                name: "LIVE 3D VIEWER",
                selected_class: "text-black",
                class: "bg-red"
              },
              %{
                id: "monte-carlo",
                name: "MONTE CARLO CLOUD RUNNER",
                selected_class: "text-black",
                class: "bg-slate"
              },
              %{
                id: "fsw",
                name: "FLIGHT SOFTWARE MODULES",
                selected_class: "text-black",
                class: "bg-orange"
              },
              %{
                id: "other-component",
                name: "Other",
                selected_class: "text-black",
                class: "bg-green"
              }
            ]}
          />
        </div>
      </div>
    </div>
    """
  end

  def page(%{page: 3} = assigns) do
    ~H"""
    <div class="w-full flex min-h-full bg-onyx max-lg:flex-col items-stretch">
      <div class="lg:w-1/2 max-lg:w-full bg-bone p-6 px-20 flex flex-col">
        <div class="w-full flex align-stretch">
          <.link href="/" style="color: #000;">
            <.ologo class="w-5" />
          </.link>
        </div>
        <div class="w-full text-sm font-mono text-black mt-16 ">
          GETTING STARTED
        </div>
        <div class="w-full text-2xl text-space text-black font-medium mt-2">
          Introduction to Elodin
        </div>
        <.questions questions={default_onboard_questions()} selected={2} />
      </div>
      <div class="lg:w-1/2 max-lg:w-full flex flex-col px-20 p-6 bg-onyx">
        <div class="w-full flex flex-col items-end">
          <img
            src={@current_user["avatar"]}
            class="ml-elo-lg w-8 h-8 inline-block rounded-full"
            phx-click={toggle("#user_dropdown")}
          /> <.user_dropdown current_user={@current_user} />
        </div>
        <div class="flex w-full justify-center items-center mt-12">
          <.onboard_box
            heading="02 / INSTALL"
            title="Download & Install the CLI"
            action="Copy and paste the following command in your terminal."
          >
            <.code title="INSTALL THE CLI" code="curl sh.elodin.systems | sh" />
            <div class="text-crema-60 font-medium text-sm mt-3 flex items-center">
              <span class="display-on-hover flex items-center">
                Use Windows?
                <div class="h-[18px] inline-block bg-onyx rounded-elo-xxs ml-2 ">
                  <svg
                    class="transition-all block w-[18px]"
                    width="18"
                    height="18"
                    viewBox="0 0 18 18"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <g clip-path="url(#clip0_1586_8894)">
                      <path
                        fill-rule="evenodd"
                        clip-rule="evenodd"
                        d="M4 10C3.44772 10 3 9.55228 3 9C3 8.44772 3.44772 8 4 8C4.55228 8 5 8.44772 5 9C5 9.55228 4.55228 10 4 10ZM9 10C8.44772 10 8 9.55228 8 9C8 8.44772 8.44772 8 9 8C9.55228 8 10 8.44772 10 9C10 9.55228 9.55228 10 9 10ZM13 9C13 9.55228 13.4477 10 14 10C14.5523 10 15 9.55228 15 9C15 8.44772 14.5523 8 14 8C13.4477 8 13 8.44772 13 9Z"
                        fill="#FFFBF0"
                      />
                    </g>
                    <defs>
                      <clipPath id="clip0_1586_8894">
                        <rect width="18" height="18" fill="white" />
                      </clipPath>
                    </defs>
                  </svg>
                  <a
                    href="https://docs.elodin.systems/quickstart"
                    class="px-1 text-xs text-green transition-all h-[18px] flex items-center"
                  >
                    Visit our docs for install instructions
                  </a>
                </div>
              </span>
            </div>
            <:button>
              <.button class="mt-10 !py-4 !px-8" type="crema" phx-click={JS.push("next_page")}>
                Next
              </.button>
            </:button>
          </.onboard_box>
        </div>
      </div>
    </div>
    """
  end

  def page(%{page: 4} = assigns) do
    ~H"""
    <div class="w-full flex min-h-full bg-onyx max-lg:flex-col items-stretch">
      <div class="lg:w-1/2 max-lg:w-full bg-bone p-6 px-20 flex flex-col">
        <div class="w-full flex align-stretch">
          <.link href="/" style="color: #000;">
            <.ologo class="w-5" />
          </.link>
        </div>
        <div class="w-full text-sm font-mono text-black mt-16 ">
          GETTING STARTED
        </div>
        <div class="w-full text-2xl text-space text-black font-medium mt-2">
          Introduction to Elodin
        </div>
        <.questions questions={default_onboard_questions()} selected={3} />
      </div>
      <div class="lg:w-1/2 max-lg:w-full flex flex-col px-20 p-6 bg-onyx">
        <div class="w-full flex flex-col items-end">
          <img
            src={@current_user["avatar"]}
            class="ml-elo-lg w-8 h-8 inline-block rounded-full"
            phx-click={toggle("#user_dropdown")}
          /> <.user_dropdown current_user={@current_user} />
        </div>
        <div class="flex w-full justify-center items-center mt-12">
          <.onboard_box
            heading="03 / TEMPLATES"
            title="Try one of our templates"
            action="Download one of our templates via the CLI."
          >
            <.code title="CUBE SAT" code="elodin create --template cubesat">
              <:icon>
                <svg
                  width="57"
                  height="56"
                  viewBox="0 0 57 56"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                  class="mr-4"
                >
                  <path
                    d="M0.5 8C0.5 3.58172 4.08172 0 8.5 0H48.5C52.9183 0 56.5 3.58172 56.5 8V48C56.5 52.4183 52.9183 56 48.5 56H8.5C4.08172 56 0.5 52.4183 0.5 48V8Z"
                    fill="#88DE9F"
                    fill-opacity="0.05"
                  />
                  <path
                    d="M46.05 34.3408L37.56 25.8508L38.27 25.1708C38.4562 24.9834 38.5608 24.73 38.5608 24.4658C38.5608 24.2016 38.4562 23.9482 38.27 23.7608L32.61 18.0708C32.4226 17.8845 32.1692 17.78 31.905 17.78C31.6408 17.78 31.3874 17.8845 31.2 18.0708L30.49 18.7808L22 10.2908C21.8126 10.1045 21.5592 10 21.295 10C21.0308 10 20.7774 10.1045 20.59 10.2908L14.93 15.9508C14.8363 16.0438 14.7619 16.1544 14.7111 16.2762C14.6603 16.3981 14.6342 16.5288 14.6342 16.6608C14.6342 16.7928 14.6603 16.9235 14.7111 17.0454C14.7619 17.1672 14.8363 17.2778 14.93 17.3708L23.42 25.8508L22.71 26.5608C22.5238 26.7482 22.4192 27.0016 22.4192 27.2658C22.4192 27.53 22.5238 27.7834 22.71 27.9708L24.83 30.0908L21.3 33.6308C21.1137 33.8182 21.0092 34.0716 21.0092 34.3358C21.0092 34.6 21.1137 34.8534 21.3 35.0408C21.4874 35.227 21.7408 35.3316 22.005 35.3316C22.2692 35.3316 22.5226 35.227 22.71 35.0408L26.25 31.5108L28.37 33.6308C28.5574 33.817 28.8108 33.9216 29.075 33.9216C29.3392 33.9216 29.5926 33.817 29.78 33.6308L30.49 32.9208L38.97 41.4108C39.0634 41.5035 39.1743 41.5768 39.2961 41.6266C39.4179 41.6763 39.5484 41.7016 39.68 41.7008C39.8116 41.7016 39.9421 41.6763 40.0639 41.6266C40.1857 41.5768 40.2966 41.5035 40.39 41.4108L46.05 35.7508C46.2363 35.5634 46.3408 35.31 46.3408 35.0458C46.3408 34.7816 46.2363 34.5282 46.05 34.3408ZM17.05 16.6608L21.3 12.4208L29.08 20.1708L24.83 24.4208L17.05 16.6608ZM26.96 29.3908L24.84 27.2708L25.55 26.5708L31.17 20.9008L31.87 20.1908L36.17 24.4408L35.46 25.1408L29.8 30.8008L29.1 31.5108L26.96 29.3908ZM39.68 39.2908L31.9 31.5108L36.17 27.2608L43.94 35.0408L39.68 39.2908ZM23 39.3508C23 39.616 22.8946 39.8704 22.7071 40.0579C22.5196 40.2454 22.2652 40.3508 22 40.3508C20.4087 40.3508 18.8826 39.7187 17.7574 38.5934C16.6321 37.4682 16 35.9421 16 34.3508C16 34.0856 16.1054 33.8312 16.2929 33.6437C16.4804 33.4561 16.7348 33.3508 17 33.3508C17.2652 33.3508 17.5196 33.4561 17.7071 33.6437C17.8946 33.8312 18 34.0856 18 34.3508C18 35.4117 18.4214 36.4291 19.1716 37.1792C19.9217 37.9294 20.9391 38.3508 22 38.3508C22.2652 38.3508 22.5196 38.4561 22.7071 38.6437C22.8946 38.8312 23 39.0856 23 39.3508ZM23 45.3508C23 45.616 22.8946 45.8704 22.7071 46.0579C22.5196 46.2454 22.2652 46.3508 22 46.3508C18.8174 46.3508 15.7652 45.0865 13.5147 42.8361C11.2643 40.5856 10 37.5334 10 34.3508C10 34.0856 10.1054 33.8312 10.2929 33.6437C10.4804 33.4561 10.7348 33.3508 11 33.3508C11.2652 33.3508 11.5196 33.4561 11.7071 33.6437C11.8946 33.8312 12 34.0856 12 34.3508C12 37.003 13.0536 39.5465 14.9289 41.4219C16.8043 43.2972 19.3478 44.3508 22 44.3508C22.2635 44.3508 22.5163 44.4548 22.7036 44.6401C22.8908 44.8255 22.9974 45.0773 23 45.3408V45.3508Z"
                    fill="#88DE9F"
                  />
                </svg>
              </:icon>
            </.code>
            <.code
              title="ROCKET"
              code="elodin create --template rocket"
              class="border-t-transparent !mt-0"
              title_class="!text-yellow"
            >
              <:icon>
                <svg
                  width="57"
                  height="56"
                  viewBox="0 0 57 56"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                  class="mr-4"
                >
                  <path
                    d="M0.5 8C0.5 3.58172 4.08172 0 8.5 0H48.5C52.9183 0 56.5 3.58172 56.5 8V48C56.5 52.4183 52.9183 56 48.5 56H8.5C4.08172 56 0.5 52.4183 0.5 48V8Z"
                    fill="#FEC504"
                    fill-opacity="0.05"
                  />
                  <path
                    fill-rule="evenodd"
                    clip-rule="evenodd"
                    d="M19.2799 30.1043L10.2824 27.1536C10.0303 27.0709 9.83869 26.8632 9.77617 26.603C9.71364 26.3448 9.79028 26.0705 9.97987 25.883L14.0016 21.8612C14.1468 21.716 14.3424 21.6353 14.5482 21.6374L27.6742 21.708L29.3442 20.0399C29.084 19.5054 29.1767 18.8439 29.6205 18.4022L31.7987 16.2219C32.2425 15.7782 32.904 15.6854 33.4385 15.9456L37.1153 12.2688C37.1254 12.2587 37.1354 12.2486 37.1455 12.2385H37.1475C40.3463 9.3765 44.0193 7.884 48.2485 8.00704C48.6539 8.01914 48.9807 8.34588 48.9927 8.75128C49.1158 12.9182 47.6697 16.5447 44.8904 19.7093C44.8541 19.7799 44.8077 19.8445 44.7492 19.9029L41.0724 23.5798C41.3326 24.1143 41.2398 24.7738 40.7961 25.2175L38.6158 27.3978C38.1721 27.8415 37.5126 27.9343 36.9781 27.6741L35.3101 29.3421L35.3807 42.4703C35.3827 42.676 35.302 42.8717 35.1568 43.0169L31.1351 47.0386C30.9455 47.2262 30.6733 47.3048 30.4151 47.2423C30.1549 47.1798 29.9472 46.9882 29.8645 46.7361L26.9138 37.7385L24.8747 39.7776C24.9413 39.9733 24.9655 40.183 24.9413 40.3948C24.8949 40.8264 24.6529 41.2136 24.2858 41.4476L21.6175 43.1398C21.0507 43.4988 20.3126 43.4161 19.8366 42.9421L14.0764 37.182C13.6025 36.706 13.5198 35.9658 13.8788 35.4011L15.5709 32.7327C15.8049 32.3656 16.1921 32.1236 16.6238 32.0772C16.8355 32.053 17.0453 32.0772 17.2409 32.1438L19.2799 30.1043ZM26.1496 23.2325L14.8591 23.172L11.9467 26.0844L20.4945 28.8879L26.1496 23.2325ZM30.3729 21.1793L18.4309 33.1214L23.8946 38.5873L35.8388 26.6451L30.3729 21.1793ZM39.6405 24.2046L32.8131 17.3772L30.776 19.4143L37.6034 26.2417L39.6405 24.2046ZM38.7853 12.8537L44.1463 18.2168C46.2358 15.6614 47.3955 12.7871 47.4641 9.53781C44.2129 9.60639 41.3391 10.7662 38.7853 12.8537ZM40.0439 22.4398L43.1236 19.36L37.6578 13.8942L34.578 16.974L40.0439 22.4398ZM33.7834 30.8684L28.1302 36.5217C28.6142 38.0001 30.9337 45.0715 30.9337 45.0715L33.8461 42.1571L33.7834 30.8684ZM16.7989 33.6597L15.2177 36.1526L20.8652 41.8002L23.3581 40.219L16.7989 33.6597ZM13.6445 38.8411C13.943 38.5426 14.4291 38.5426 14.7276 38.8411C15.0281 39.1416 15.0281 39.6257 14.7296 39.9262L11.1537 43.5002C10.8552 43.7987 10.3691 43.7987 10.0706 43.5002C9.77005 43.2017 9.77005 42.7156 10.0706 42.4151L13.6445 38.8411ZM17.0752 42.2719C17.3737 41.9733 17.8598 41.9733 18.1583 42.2719C18.4588 42.5704 18.4588 43.0564 18.1583 43.357L14.5844 46.9309C14.2859 47.2294 13.7998 47.2294 13.5013 46.9309C13.2008 46.6304 13.2008 46.1464 13.5013 45.8458L17.0752 42.2719ZM15.3589 40.5575C15.6594 40.257 16.1454 40.257 16.444 40.5575C16.7425 40.856 16.7425 41.342 16.444 41.6406L10.3085 47.7761C10.01 48.0746 9.52393 48.0746 9.22539 47.7761C8.92487 47.4756 8.92487 46.9916 9.22539 46.691L15.3589 40.5575Z"
                    fill="#FEC504"
                  />
                </svg>
              </:icon>
            </.code>
            <:button>
              <.button class="mt-10 !py-4 !px-8" type="crema" phx-click={JS.push("next_page")}>
                Next
              </.button>
            </:button>
          </.onboard_box>
        </div>
      </div>
    </div>
    """
  end

  def page(%{page: 5} = assigns) do
    ~H"""
    <div class="w-full flex h-full bg-onyx max-lg:flex-col">
      <div class="lg:w-1/2 max-lg:w-full h-full bg-bone p-6 px-20 flex flex-col">
        <div class="w-full flex align-stretch">
          <.link href="/" style="color: #000;">
            <.ologo class="w-5" />
          </.link>
        </div>
        <div class="w-full text-sm font-mono text-black mt-16 ">
          GETTING STARTED
        </div>
        <div class="w-full text-2xl text-space text-black font-medium mt-2">
          Introduction to Elodin
        </div>
        <.questions questions={default_onboard_questions()} selected={4} />
      </div>
      <div class="lg:w-1/2 max-lg:w-full flex flex-col px-20 p-6 bg-onyx">
        <div class="w-full flex flex-col items-end">
          <img
            src={@current_user["avatar"]}
            class="ml-elo-lg w-8 h-8 inline-block rounded-full"
            phx-click={toggle("#user_dropdown")}
          /> <.user_dropdown current_user={@current_user} />
        </div>
        <div class="flex w-full justify-center items-center mt-12">
          <.onboard_box
            heading="04 / MONTE CARLO"
            title="Run a Monte Carlo Sim"
            action="See the template simulation in action."
          >
            <.code title="MONTE CARLO" code="elodin monte-carlo example.py" />
            <:button>
              <.button class="mt-10 !py-4 !px-8" type="crema" phx-click={JS.push("next_page")}>
                Next
              </.button>
            </:button>
          </.onboard_box>
        </div>
      </div>
    </div>
    """
  end

  def page(%{page: 6} = assigns) do
    ~H"""
    <div class="w-full flex h-full bg-onyx max-lg:flex-col">
      <div class="lg:w-1/2 max-lg:w-full h-full bg-bone flex flex-col items-center justify-center">
        <div phx-hook="FireworksHook" id="fireworks" style="width: 300px; height: 375.652px;"></div>
      </div>
      <div class="lg:w-1/2 max-lg:w-full flex flex-col px-20 p-6 bg-onyx">
        <div class="w-full flex flex-col items-end">
          <img
            src={@current_user["avatar"]}
            class="ml-elo-lg w-8 h-8 inline-block rounded-full"
            phx-click={toggle("#user_dropdown")}
          /> <.user_dropdown current_user={@current_user} />
        </div>
        <div class="flex w-full justify-center items-center mt-12">
          <.onboard_box
            heading="DONE"
            title="You're all set"
            action="Check out our docs, explore the dashboard, join our discord for support, and to share work!"
          >
            <div class="w-full mt-8 font-mono font-medium text-sm">
              <.link
                class={[
                  "h-20 flex items-center mx-[-40px] px-10 justify-between",
                  "border-t border-onyx-9"
                ]}
                href="/"
              >
                DASHBOARD <.link_arrow />
              </.link>
              <.link
                class={[
                  "h-20 flex items-center mx-[-40px] px-10 justify-between",
                  "border-b border-t border-onyx-9"
                ]}
                href="https://docs.elodin.systems"
              >
                DOCUMENTATION <.link_arrow />
              </.link>
              <.link
                class={[
                  "h-20 flex items-center mx-[-40px] px-10 justify-between",
                  "border-b border-onyx-9"
                ]}
                href="https://elodin.app/discord"
              >
                DISCORD <.link_arrow />
              </.link>
            </div>
          </.onboard_box>
        </div>
      </div>
    </div>
    """
  end

  def render(assigns) do
    ~H"""
    <.page
      page={@page}
      loading={@loading}
      selected_industries={@selected_industries}
      current_user={@current_user}
    />
    """
  end
end
