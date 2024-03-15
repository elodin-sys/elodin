defmodule ElodinDashboardWeb.SidebarComponents do
  use Phoenix.Component
  import ElodinDashboardWeb.CoreComponents
  import ElodinDashboardWeb.IconComponents

  attr(:name, :string, required: true)
  attr(:run_id, :string, required: true)
  attr(:project, :string, required: true)
  attr(:selected, :boolean, default: false)
  attr(:progress, :float, default: 0.0)

  def project_run_list_item(assigns) do
    ~H"""
    <a
      href={"/monte_carlo/#{@project}/#{@run_id}"}
      class={[
        "p-2.5 flex justify-between items-center",
        @selected && "rounded bg-orange-50 text-black-secondary"
      ]}
    >
      <div><%= @name %></div>
      <%= if @progress < 1.0 do %>
        <div class={[
          "h-1 w-5",
          if(@selected, do: "bg-black-secondary", else: "bg-orange-50")
        ]}>
          <div class="h-full bg-yellow-400" style={"width: #{@progress * 100}%"} />
        </div>
      <% end %>
    </a>
    """
  end

  attr(:project, :string, required: true)
  attr(:project_run, :string, default: nil)
  attr(:projects, :list, default: [%{value: "project", label: "Project"}])
  attr(:project_runs, :list, default: [])

  def sidebar(assigns) do
    ~H"""
    <div class="w-80 h-full bg-black-secondary border-r border-neutral-800 flex flex-col font-semibold text-xs text-orange-50">
      <div class="m-4">
        <select
          name="projects"
          id="project-select"
          disabled="true"
          class="h-12 w-full bg-black-secondary border border-orange-50/20"
        >
          <%= for p <- @projects do %>
            <option selected={p.value == @project} value={p.value}>
              <%= p.label %>
            </option>
          <% end %>
        </select>
      </div>
      <div class="flex flex-col grow">
        <div class="mx-4 my-5 opacity-60">
          MONTE CARLO RUNS
        </div>

        <div class="mx-4 flex flex-col gap-1">
          <%= for run <- @project_runs do %>
            <.project_run_list_item
              name={run.name}
              run_id={run.id}
              project={@project}
              selected={run.id == @project_run}
              progress={run.progress}
            />
          <% end %>
        </div>
      </div>

      <div class="m-4">
        <.divider />

        <a href="https://elodin.app/discord" class="p-2.5 flex justify-between items-center">
          Discord <.icon_link />
        </a>
        <a href="https://docs.elodin.systems" class="p-2.5 flex justify-between items-center">
          Documentation <.icon_link />
        </a>
      </div>
    </div>
    """
  end
end
