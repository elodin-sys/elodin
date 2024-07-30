defmodule ElodinDashboardWeb.NavbarComponents do
  use Phoenix.Component
  import ElodinDashboardWeb.CoreComponents
  import ElodinDashboardWeb.IconComponents

  slot(:navbar_center, required: false)
  slot(:navbar_right, required: false)
  attr(:current_user, :map, required: false)

  def navbar(assigns) do
    ~H"""
    <ul class="w-full z-10 flex items-center gap-4 px-4 sm:px-6 lg:px-4 justify-start h-16 bg-black-secondary text-white shrink-0 fixed border-b-sep-black border-b border-b-solid">
      <li class="mr-auto flex ">
        <.link href="/">
          <img src="/images/o-logo.svg" class="w-5" />
        </.link>
      </li>
      <li style="position: absolute; left: calc(50%); transform: translate(-50%);">
        <%= render_slot(@navbar_center) %>
      </li>
      <li class="text-[0.8125rem] ml-auto flex items-center">
        <%= render_slot(@navbar_right) %>
        <%= if @current_user do %>
          <img
            src={@current_user["avatar"]}
            class="ml-elo-lg w-8 h-8 inline-block rounded-full"
            phx-click={toggle("#user_dropdown")}
          />
          <.user_dropdown current_user={@current_user} />
        <% else %>
          Log In
        <% end %>
      </li>
    </ul>
    """
  end

  def user_dropdown(assigns) do
    subscription = assigns.current_user["subscription_status"]
    has_subscription = subscription != nil

    sub_start =
      if(has_subscription && subscription.trial_start != nil,
        do: DateTime.from_unix!(subscription.trial_start - 1),
        else: DateTime.utc_now()
      )

    sub_end =
      if(has_subscription && subscription.subscription_end != nil,
        do: DateTime.from_unix!(subscription.subscription_end),
        else: DateTime.utc_now()
      )

    assigns =
      assigns
      |> assign(:subscription, subscription)
      |> assign(:has_subscription, has_subscription)
      |> assign(:trial_length, DateTime.diff(sub_end, sub_start, :day))
      |> assign(:trial_left, DateTime.diff(sub_end, DateTime.utc_now(), :day))

    ~H"""
    <div
      class={[
        "hidden absolute right-2 top-[68px] z-10 w-80 origin-top-right rounded-elo-md",
        "bg-primary-smoke shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none",
        "border border-primary-onyx-9"
      ]}
      role="menu"
      aria-orientation="vertical"
      aria-labelledby="menu-button"
      tabindex="-1"
      id="user_dropdown"
    >
      <div class="flex flex-col gap-4 p-4">
        <div class="py-1 flex items-center mt-elo-md" role="none">
          <img src={@current_user["avatar"]} class="ml-elo-md w-8 h-8 inline-block rounded-full" />
          <div class="flex flex-col ml-elo-lg">
            <div><%= @current_user["name"] %></div>
            <div class="text-white-opacity-300"><%= @current_user["email"] %></div>
          </div>
        </div>
        <hr class="border-primary-onyx-9" />
        <%= if @has_subscription do %>
          <.trial_progress_bar
            :if={@subscription.subscription_end == @subscription.trial_end}
            trial_left={@trial_left}
            trial_length={@trial_length}
            label={if(@trial_left == 1, do: "DAY", else: "DAYS") <> " LEFT IN FREE TRIAL"}
          />
          <.link href={@subscription.portal_url} class="p-1 gap-2 flex items-center">
            <.icon_link /> Billing Dashboard
          </.link>
        <% end %>
        <.link href="/users/log_out" class="p-1 gap-2 flex items-center">
          <.icon_exit /> Log Out
        </.link>
      </div>
    </div>
    """
  end

  attr(:current_user, :map, required: true)

  def subscription_status_modal(assigns) do
    subscription = assigns.current_user["subscription_status"]
    has_subscription = subscription != nil

    sub_start =
      if(has_subscription && subscription.trial_start != nil,
        do: DateTime.from_unix!(subscription.trial_start - 1),
        else: DateTime.utc_now()
      )

    sub_end =
      if(has_subscription && subscription.subscription_end != nil,
        do: DateTime.from_unix!(subscription.subscription_end - 1),
        else: DateTime.utc_now()
      )

    assigns =
      assigns
      |> assign(:subscription_ended, DateTime.compare(DateTime.utc_now(), sub_end) == :gt)
      |> assign(
        :is_trial,
        has_subscription && subscription.subscription_end == subscription.trial_end
      )
      |> assign(:trial_length, DateTime.diff(sub_end, sub_start, :day))
      |> assign(
        :dashboard_url,
        if(has_subscription, do: subscription.portal_url, else: "")
      )

    ~H"""
    <.modal id="subscription_status" show={@subscription_ended} can_close={false}>
      <.icon_warning />

      <h2 class="font-semibold mt-6 mb-1.5">
        <%= if @is_trial do %>
          <%= @trial_length %> day trial has ended
        <% else %>
          You need to subscribe to continue
        <% end %>
      </h2>

      <p class="text-primary-creame-60">
        <%= if @is_trial do %>
          Not all good things have to come to an end, but your Elodin trial has expired.
        <% else %>
          Not all good things have to come to an end, but your Elodin subscription has ended.
        <% end %>
      </p>

      <div class="flex items-center flex-row mt-8">
        <.link href={@dashboard_url}>
          <.button class="py-4 px-6">BILLING DASHBOARD</.button>
        </.link>
      </div>
    </.modal>
    """
  end

  slot(:inner_block, required: true)
  slot(:navbar_center, required: false)
  slot(:navbar_right, required: false)
  attr(:current_user, :map, required: false)

  def navbar_layout(assigns) do
    ~H"""
    <.navbar current_user={@current_user}>
      <:navbar_center>
        <%= render_slot(@navbar_center) %>
      </:navbar_center>
      <:navbar_right>
        <%= render_slot(@navbar_right) %>
      </:navbar_right>
    </.navbar>
    <div class="h-full flex pt-[64px]">
      <%= render_slot(@inner_block) %>
      <.subscription_status_modal current_user={@current_user} />
    </div>
    """
  end
end
