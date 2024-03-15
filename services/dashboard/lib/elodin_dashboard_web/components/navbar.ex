defmodule ElodinDashboardWeb.NavbarComponents do
  use Phoenix.Component
  import ElodinDashboardWeb.CoreComponents

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
    ~H"""
    <div
      class="hidden absolute right-2 top-[68px] z-10 mt-2 w-56 origin-top-right rounded-elo-md bg-black-primrary shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none"
      role="menu"
      aria-orientation="vertical"
      aria-labelledby="menu-button"
      tabindex="-1"
      id="user_dropdown"
    >
      <div class="py-1 flex items-center mt-elo-md" role="none">
        <img src={@current_user["avatar"]} class="ml-elo-md w-8 h-8 inline-block rounded-full" />
        <div class="flex flex-col ml-elo-lg">
          <div><%= @current_user["name"] %></div>
          <div class="text-white-opacity-300"><%= @current_user["email"] %></div>
        </div>
      </div>
      <.link href="/users/log_out">
        <.button class="m-elo-md">Log Out</.button>
      </.link>
    </div>
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
    </div>
    """
  end
end
