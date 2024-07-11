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

  def handle_event("set_page", %{"page" => page}, socket) do
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

  attr(:title, :string)
  attr(:download_title, :string)
  attr(:download_href, :string)
  attr(:sha_href, :string)
  attr(:class, :string, default: "")
  attr(:title_class, :string, default: "")
  slot :title_note
  slot :icon

  def download_option(assigns) do
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
          <%= @title %><%= render_slot(@title_note) %>
        </div>
        <div class="flex w-full items-center justify-between">
          <.link href={@download_href} class="underline underline-offset-4 text-crema">
            <%= @download_title %>
          </.link>
          <.link href={@sha_href} class="underline underline-offset-4 text-crema">
            sha256
          </.link>
        </div>
      </div>
    </div>
    """
  end

  attr(:title, :string)
  attr(:code, :string)
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
          <pre class="inline overflow-scroll"><span class="select-none">$ </span><span class="select-all" id="download-code"><%= @code %></span></pre>
          <span class="ml-4">
            <span
              class="hover:opacity-75 transition-all cursor-pointer"
              phx-click={JS.dispatch("phx:copy-inner", to: "\#download-code")}
            >
              <.icon_copy />
            </span>
          </span>
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

  def onboarding_steps(assigns) do
    ~H"""
    <div class="w-full flex-col flex">
      <div :for={onboarding_step <- @onboarding_steps} class="w-full mt-20 flex items-center">
        <div
          class={[
            "rounded-full w-12 h-12 flex items-center justify-center font-bold font-mono text-xl",
            if(onboarding_step.index <= @selected,
              do: "text-bone bg-primary-smoke",
              else: "text-primary-smoke border-primary-smoke border"
            ),
            if(onboarding_step.index <= @selected && onboarding_step.index != 1,
              do: "cursor-pointer",
              else: ""
            )
          ]}
          phx-click={
            if(onboarding_step.index <= @selected && onboarding_step.index != 1,
              do: JS.push("set_page", value: %{page: onboarding_step.index + 1}),
              else: nil
            )
          }
        >
          <%= onboarding_step.number %>
        </div>
        <div style="position:relative;">
          <%= if onboarding_step.index < @selected do %>
            <div style="width: 2px; height: 85px; background: #0D0D0D; position: absolute; left: -25.5px; top: 23px;">
            </div>
          <% end %>
          <%= if onboarding_step.index == @selected && @selected != Enum.count(@onboarding_steps) do %>
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
            <%= onboarding_step.title %>
          </div>
          <div>
            <%= onboarding_step.prompt %>
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

  def default_onboard_steps() do
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
        <.onboarding_steps onboarding_steps={default_onboard_steps()} selected={1} />
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
        <.onboarding_steps onboarding_steps={default_onboard_steps()} selected={1} />
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
        <.onboarding_steps onboarding_steps={default_onboard_steps()} selected={2} />
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
            action="Select and download the CLI for your OS."
          >
            <.download_option
              title="Apple Silicon macOS"
              download_title="elodin-aarch64-apple-darwin.tar.gz"
              download_href="https://storage.googleapis.com/elodin-releases/latest/elodin-aarch64-apple-darwin.tar.gz"
              sha_href="https://storage.googleapis.com/elodin-releases/latest/elodin-aarch64-apple-darwin.tar.gz.sha256"
            />
            <.download_option
              title="x64 Linux"
              class="border-t-transparent !mt-0"
              download_title="elodin-x86_64-unknown-linux-gnu.tar.gz"
              download_href="https://storage.googleapis.com/elodin-releases/latest/elodin-x86_64-unknown-linux-gnu.tar.gz"
              sha_href="https://storage.googleapis.com/elodin-releases/latest/elodin-x86_64-unknown-linux-gnu.tar.gz.sha256"
            />
            <.download_option
              title="x64 Windows"
              class="border-t-transparent !mt-0 display-on-hover"
              download_title="elodin-x86_64-pc-windows-msvc.zip"
              download_href="https://storage.googleapis.com/elodin-releases/latest/elodin-x86_64-pc-windows-msvc.zip"
              sha_href="https://storage.googleapis.com/elodin-releases/latest/elodin-x86_64-pc-windows-msvc.zip.sha256"
            >
              <:title_note>
                <div class="h-[18px] inline-block bg-onyx rounded-elo-xxs ml-2">
                  <span class="display-on-hover-hide transition-all block w-[18px]">*</span>
                  <a
                    href="https://docs.elodin.systems/quickstart#windows"
                    class="display-on-hover-show px-1 text-xs text-green transition-all h-[18px] flex items-center"
                  >
                    Extra instructions
                  </a>
                </div>
              </:title_note>
            </.download_option>
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
        <.onboarding_steps onboarding_steps={default_onboard_steps()} selected={3} />
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
            <.code title="CUBE SAT" code="elodin create --template cube-sat">
              <:icon>
                <.icon_cubsat />
              </:icon>
            </.code>
            <.code
              title="ROCKET"
              code="elodin create --template rocket"
              class="border-t-transparent !mt-0"
              title_class="!text-yellow"
            >
              <:icon>
                <.icon_rocket />
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
        <.onboarding_steps onboarding_steps={default_onboard_steps()} selected={4} />
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
            <.code title="MONTE CARLO" code="elodin monte-carlo run --name rocket rocket.py" />
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
