defmodule ParacosmDashboardWeb.NavbarComponents do
  use Phoenix.Component
  import ParacosmDashboardWeb.CoreComponents

  slot(:navbar_left, required: false)

  def navbar(assigns) do
    ~H"""
    <ul class="w-full z-10 flex items-center gap-4 px-4 sm:px-6 lg:px-8 justify-start h-16 bg-surface-secondary text-white shrink-0 fixed">
      <li class="mr-auto flex ">
        <%= render_slot(@navbar_left) %>
      </li>
      <li style="position: absolute; left: calc(50% - 25px/2);">
        <img src="/images/o-logo.svg" class="w-5" />
      </li>
      <li class="text-[0.8125rem] ml-auto">
        <%= render_slot(@navbar_right) %>
        <%= if @current_user do %>
          <%= @current_user["email"] %>
        <% else %>
          Log In
        <% end %>
      </li>
    </ul>
    """
  end

  slot(:inner_block, required: true)
  slot(:navbar_left, required: false)
  slot(:navbar_right, required: false)

  def navbar_layout(assigns) do
    ~H"""
    <.navbar current_user={@current_user} navbar_left={@navbar_left} navbar_right={@navbar_right} />
    <div class="h-full pt-[56px]">
      <%= render_slot(@inner_block) %>
    </div>
    """
  end
end
