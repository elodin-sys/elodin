defmodule ElodinDashboardWeb.CoreComponents do
  @moduledoc """
  Provides core UI components.

  At first glance, this module may seem daunting, but its goal is to provide
  core building blocks for your application, such as modals, tables, and
  forms. The components consist mostly of markup and are well-documented
  with doc strings and declarative assigns. You may customize and style
  them in any way you want, based on your application growth and needs.

  The default components use Tailwind CSS, a utility-first CSS framework.
  See the [Tailwind CSS documentation](https://tailwindcss.com) to learn
  how to customize them or feel free to swap in another framework altogether.

  Icons are provided by [heroicons](https://heroicons.com). See `icon/1` for usage.
  """
  use Phoenix.Component

  alias Phoenix.HTML
  alias Phoenix.LiveView.JS
  import ElodinDashboardWeb.Gettext

  attr(:trial_left, :integer, default: 5)
  attr(:trial_length, :integer, default: 15)
  attr(:label, :string, default: "DAYS LEFT IN THE TRIAL")

  def trial_progress_bar(assigns) do
    trial_progress_value = assigns.trial_left / assigns.trial_length

    trial_progress_color =
      cond do
        trial_progress_value > 0.66 -> "mint"
        trial_progress_value > 0.33 -> "yolk"
        trial_progress_value >= 0.0 -> "reddish"
      end

    assigns =
      assigns
      |> assign(:value, trial_progress_value)
      |> assign(:color, trial_progress_color)

    ~H"""
    <div class="flex flex-col gap-3 p-4 bg-primary-onyx rounded-lg">
      <div class="w-full font-mono text-sm text-primary-creame-60 tracking-elo-mono-medium leading-tight text-center">
        <span class={"text-#{@color}"}>
          <%= @trial_left %>
        </span>
        <%= @label %>
      </div>

      <div class="w-full h-1.5 flex">
        <%= if @value > 0.0 do %>
          <div
            class={["h-full rounded bg-#{@color}", if(@value < 1.0, do: "mr-2", else: "")]}
            style={"width: #{@value * 100.0}%;"}
          />
        <% end %>
        <%= if @value < 1.0 do %>
          <div
            class="border border-primary-creame-10 h-full rounded"
            style={"width: #{(1.0 - @value) * 100}%;"}
          />
        <% end %>
      </div>
    </div>
    """
  end

  attr(:minutes_used, :integer, default: 0)
  attr(:free_minutes_total, :integer, default: 60)
  attr(:refresh_date, DateTime, default: nil)

  def monte_carlo_usage_bar(assigns) do
    refresh_date =
      if(assigns.refresh_date != nil, do: assigns.refresh_date, else: Date.utc_today())

    past_free_allowance = assigns.minutes_used > assigns.free_minutes_total

    assigns =
      assigns
      |> assign(
        :value,
        if(
          past_free_allowance,
          do: assigns.free_minutes_total / assigns.minutes_used,
          else: assigns.minutes_used / assigns.free_minutes_total
        )
      )
      |> assign(:past_free_allowance, past_free_allowance)
      |> assign(
        :refresh_date,
        Calendar.Strftime.strftime!(refresh_date, "%b %d/%y") |> String.upcase()
      )

    ~H"""
    <div class="flex flex-col gap-4 p-4 bg-primary-onyx rounded-lg tracking-elo-mono-medium">
      <div class="w-full font-mono text-sm">
        MONTE CARLO COMPUTE
      </div>

      <hr class="border-primary-onyx-9" />

      <div class="flex flex-col gap-3">
        <div class="w-full font-mono text-sm text-primary-creame-60">
          USAGE MINUTES
        </div>

        <div class="w-full h-1.5 flex">
          <%= if @value > 0.0 do %>
            <div
              class={["h-full rounded bg-mint", if(@value < 1.0, do: "mr-1", else: "")]}
              style={"width: #{@value * 100.0}%;"}
            />
          <% end %>

          <%= if @value < 1.0 do %>
            <div
              class={[
                "h-full rounded",
                if(@past_free_allowance, do: "bg-yolk", else: "border border-primary-creame-10")
              ]}
              style={"width: #{(1.0 - @value) * 100}%;"}
            />
          <% end %>
        </div>

        <div class="w-full font-mono text-sm">
          <%= @minutes_used %>
          <span class="text-primary-creame-60">
            OF <%= @free_minutes_total %> MIN
          </span>
        </div>
      </div>

      <hr class="border-primary-onyx-9" />

      <div class="w-full font-mono text-sm flex justify-between">
        <span class="text-primary-creame-60">
          MINUTES REFRESH
        </span>
        <span class="text-primary-creame">
          <%= @refresh_date %>
        </span>
      </div>
    </div>
    """
  end

  attr(:number, :float, default: 50.0)
  attr(:value, :float, default: 50.0)
  attr(:label, :string, default: "LABEL")
  attr(:color, :string, default: "bg-mint")

  def label_progress_bar(assigns) do
    ~H"""
    <.horizontal_label value={@number} label={@label} class="mb-elo-md" />
    <.progress_bar value={@value} color={@color} />
    """
  end

  attr(:value, :float, default: 1.0)
  attr(:color, :string, default: "bg-mint")

  def progress_bar(assigns) do
    ~H"""
    <div class="w-full h-6 flex mb-elo-xl">
      <%= if @value > 0.0 do %>
        <div
          class={["h-full rounded-elo-xs", @color, if(@value < 1.0, do: "mr-2", else: "")]}
          style={"width: #{@value * 100.0}%;"}
        />
      <% end %>
      <%= if @value < 1.0 do %>
        <div
          class="border border-primary-creame-60 h-full rounded-elo-xs"
          style={"width: #{(1.0 - @value) * 100}%;"}
        />
      <% end %>
    </div>
    """
  end

  def divider(assigns) do
    ~H"""
    <hr class="border-t border-white border-opacity-20 my-elo-xl" />
    """
  end

  attr(:label, :string, default: "LABEL")
  attr(:value, :string, default: "FOO")
  attr(:class, :string, default: nil)

  def horizontal_label(assigns) do
    ~H"""
    <div class={["w-full font-mono tracking-elo-mono-medium content-stretch", @class]}>
      <span class="float-left text-primary-creame-60 text-sm font-medium leading-tight">
        <%= @label %>
      </span>
      <span class="text-primary-creame float-right text-sm leading-tight">
        <%= @value %>
      </span>
    </div>
    """
  end

  @doc """
  Renders a modal.

  ## Examples

      <.modal id="confirm-modal">
        This is a modal.
      </.modal>

  JS commands may be passed to the `:on_cancel` to configure
  the closing/cancel event, for example:

      <.modal id="confirm" on_cancel={JS.navigate(~p"/posts")}>
        This is another modal.
      </.modal>

  """
  attr(:id, :string, required: true)
  attr(:show, :boolean, default: false)
  attr(:wrapper_class, :string, default: "w-full max-w-[500px]")
  attr(:bg_color, :string, default: "black-primary")
  attr(:container_padding, :string, default: "elo-xl")
  attr(:on_cancel, JS, default: %JS{})
  attr(:can_close, :boolean, default: true)
  slot(:inner_block, required: true)

  def modal(assigns) do
    ~H"""
    <div
      id={@id}
      phx-mounted={@show && show_modal(@id)}
      phx-remove={hide_modal(@id)}
      data-cancel={if(@can_close, do: JS.exec(@on_cancel, "phx-remove"), else: nil)}
      class="relative z-50 hidden"
    >
      <div
        id={"#{@id}-bg"}
        class="backdrop-blur-sm bg-black-opacity-600 fixed inset-0 transition-opacity"
        aria-hidden="true"
      />
      <div
        class="fixed inset-0 overflow-y-auto"
        aria-labelledby={"#{@id}-title"}
        aria-describedby={"#{@id}-description"}
        role="dialog"
        aria-modal="true"
        tabindex="0"
      >
        <div class="flex min-h-full items-center justify-center">
          <div class={["p-4 sm:p-6 lg:py-8", @wrapper_class]}>
            <.focus_wrap
              id={"#{@id}-container"}
              phx-window-keydown={JS.exec("data-cancel", to: "##{@id}")}
              phx-key="escape"
              phx-click-away={JS.exec("data-cancel", to: "##{@id}")}
              class={"relative hidden overflow-hidden rounded-elo-xs bg-#{@bg_color} text-white p-#{@container_padding} shadow-lg transition"}
            >
              <div :if={@can_close} class="absolute top-elo-xl right-5">
                <button
                  phx-click={JS.exec("data-cancel", to: "##{@id}")}
                  type="button"
                  class="-m-3 flex-none p-3 hover:opacity-40"
                  aria-label={gettext("close")}
                >
                  <ElodinDashboardWeb.IconComponents.x />
                </button>
              </div>
              <div id={"#{@id}-content"}>
                <%= render_slot(@inner_block) %>
              </div>
            </.focus_wrap>
          </div>
        </div>
      </div>
    </div>
    """
  end

  @doc """
  Renders frame wrapper with default style.

  ## Examples

      <.frame>Welcome Back!</.frame>
  """
  slot(:inner_block, required: true)

  def frame(assigns) do
    ~H"""
    <div class="p-6 bg-black-secondary border border-white border-opacity-10 rounded-elo-xs">
      <%= render_slot(@inner_block) %>
    </div>
    """
  end

  @doc """
  Renders flash notices.

  ## Examples

      <.flash kind={:info} flash={@flash} />
      <.flash kind={:info} phx-mounted={show("#flash")}>Welcome Back!</.flash>
  """
  attr(:id, :string, doc: "the optional id of flash container")
  attr(:flash, :map, default: %{}, doc: "the map of flash messages to display")
  attr(:title, :string, default: nil)
  attr(:kind, :atom, values: [:info, :error], doc: "used for styling and flash lookup")
  attr(:rest, :global, doc: "the arbitrary HTML attributes to add to the flash container")

  slot(:inner_block, doc: "the optional inner block that renders the flash message")

  def flash(assigns) do
    assigns = assign_new(assigns, :id, fn -> "flash-#{assigns.kind}" end)

    ~H"""
    <div
      :if={msg = render_slot(@inner_block) || Phoenix.Flash.get(@flash, @kind)}
      id={@id}
      phx-click={JS.push("lv:clear-flash", value: %{key: @kind}) |> hide("##{@id}")}
      role="alert"
      class={[
        "fixed top-2 right-2 mr-2 w-80 sm:w-96 z-50 rounded-elo-xs p-3 ring-1 ring-opacity-40",
        @kind == :info && "bg-mint-40 text-mint ring-mint fill-mint",
        @kind == :error &&
          "bg-reddish-40 text-reddish shadow-md ring-reddish fill-reddish"
      ]}
      {@rest}
    >
      <p :if={@title} class="flex items-center gap-1.5 text-sm font-semibold leading-6">
        <.icon :if={@kind == :info} name="hero-information-circle-mini" class="h-4 w-4" />
        <.icon :if={@kind == :error} name="hero-exclamation-circle-mini" class="h-4 w-4" />
        <%= @title %>
      </p>
      <p class="mt-2 text-sm leading-5"><%= msg %></p>
      <button type="button" class="group absolute top-1 right-1 p-2" aria-label={gettext("close")}>
        <.icon name="hero-x-mark-solid" class="h-5 w-5 opacity-40 group-hover:opacity-70" />
      </button>
    </div>
    """
  end

  @doc """
  Shows the flash group with standard titles and content.

  ## Examples

      <.flash_group flash={@flash} />
  """
  attr(:flash, :map, required: true, doc: "the map of flash messages")
  attr(:id, :string, default: "flash-group", doc: "the optional id of flash container")

  def flash_group(assigns) do
    ~H"""
    <div id={@id}>
      <.flash kind={:info} title="Success!" flash={@flash} />
      <.flash kind={:error} title="Error!" flash={@flash} />
      <.flash
        id="client-error"
        kind={:error}
        title="We can't find the internet"
        phx-disconnected={show(".phx-client-error #client-error")}
        phx-connected={hide("#client-error")}
        hidden
      >
        Attempting to reconnect <.icon name="hero-arrow-path" class="ml-1 h-3 w-3 animate-spin" />
      </.flash>

      <.flash
        id="server-error"
        kind={:error}
        title="Something went wrong!"
        phx-disconnected={show(".phx-server-error #server-error")}
        phx-connected={hide("#server-error")}
        hidden
      >
        Hang in there while we get back on track
        <.icon name="hero-arrow-path" class="ml-1 h-3 w-3 animate-spin" />
      </.flash>
    </div>
    """
  end

  @doc """
  Renders a simple form.

  ## Examples

      <.simple_form for={@form} phx-change="validate" phx-submit="save">
        <.input field={@form[:email]} label="Email"/>
        <.input field={@form[:username]} label="Username" />
        <:actions>
          <.button>Save</.button>
        </:actions>
      </.simple_form>
  """
  attr(:for, :any, required: true, doc: "the datastructure for the form")
  attr(:as, :any, default: nil, doc: "the server side parameter to collect all input under")

  attr(:rest, :global,
    include: ~w(autocomplete name rel action enctype method novalidate target multipart),
    doc: "the arbitrary HTML attributes to apply to the form tag"
  )

  slot(:inner_block, required: true)
  slot(:actions, doc: "the slot for form actions, such as a submit button")

  def simple_form(assigns) do
    ~H"""
    <.form :let={f} for={@for} as={@as} {@rest}>
      <div class="mt-10 space-y-8 bg-white">
        <%= render_slot(@inner_block, f) %>
        <div :for={action <- @actions} class="mt-2 flex items-center justify-between gap-6">
          <%= render_slot(action, f) %>
        </div>
      </div>
    </.form>
    """
  end

  @doc """
  Renders a button.

  ## Examples

      <.button>Send!</.button>
      <.button phx-click="go" class="ml-2">Send!</.button>
  """
  attr(:type, :string, default: nil)
  attr(:class, :any, default: nil)
  attr(:disabled, :boolean, default: false)
  attr(:rest, :global, include: ~w(disabled form name value))

  slot(:inner_block, required: true)

  def button(%{type: "outline", class: class} = assigns) do
    button(Map.merge(assigns, %{type: "", class: [class, "bg-transparent"]}))
  end

  def button(%{type: "secondary", class: class} = assigns) do
    ~H"""
    <button
      class={[
        "phx-submit-loading:opacity-75 rounded-elo-xxs border-solid border border-white border-opacity-5 p-[12px] ",
        "transition-all",
        "bg-opacity-0 bg-white",
        "hover:bg-opacity-5 hover:border-opacity-20",
        "active:brightness-75",
        "text-[12px] leading-[8px] font-semibold text-primary-creame",
        @class
      ]}
      {@rest}
    >
      <%= render_slot(@inner_block) %>
    </button>
    """
  end

  def button(%{type: "creame", class: class} = assigns) do
    ~H"""
    <button
      class={[
        "phx-submit-loading:opacity-75 rounded-elo-xxs border-solid border border-primary-creame border-opacity-40 p-[12px] ",
        "transition-all",
        "bg-opacity-5 bg-primary-creame",
        "hover:bg-opacity-10 hover:border-opacity-45",
        "active:brightness-75",
        "text-[12px] leading-[8px] font-semibold text-primary-creame",
        @class
      ]}
      {@rest}
    >
      <%= render_slot(@inner_block) %>
    </button>
    """
  end

  def button(%{type: "invert"} = assigns) do
    ~H"""
    <button
      type={@type}
      disabled={@disabled}
      class={[
        "rounded-elo-sm bg-white hover:bg-white/80 p-[12px] ",
        "text-[12px] leading-[8px] font-semibold text-hyper-blue active:text-hyper-blue-dim",
        @class
      ]}
      {@rest}
    >
      <%= render_slot(@inner_block) %>
    </button>
    """
  end

  def button(%{type: "danger"} = assigns) do
    ~H"""
    <button
      class={[
        "phx-submit-loading:opacity-75 rounded-elo-xxs border-solid border border-reddish border-opacity-40 p-[12px] ",
        "flex gap-2 justify-center items-center h-[38px]",
        "transition-all",
        "bg-reddish bg-opacity-5",
        "hover:bg-opacity-15 hover:border-opacity-30",
        "active:brightness-75",
        "text-[12px] leading-[8px] font-semibold text-reddish",
        @class
      ]}
      {@rest}
    >
      <%= render_slot(@inner_block) %>
    </button>
    """
  end

  def button(assigns) do
    ~H"""
    <button
      class={[
        "phx-submit-loading:opacity-75 rounded-elo-xxs border-solid border border-mint border-opacity-40 p-[12px] ",
        "transition-all",
        "bg-mint bg-opacity-5",
        "enabled:hover:bg-opacity-15 enabled:hover:border-opacity-30",
        "enabled:active:brightness-75",
        "text-[12px] leading-[8px] font-semibold text-mint",
        @class
      ]}
      disabled={@disabled}
      {@rest}
    >
      <%= render_slot(@inner_block) %>
    </button>
    """
  end

  @doc """
  Renders an input with label and error messages.

  A `Phoenix.HTML.FormField` may be passed as argument,
  which is used to retrieve the input name, id, and values.
  Otherwise all attributes may be passed explicitly.

  ## Types

  This function accepts all HTML input types, considering that:

    * You may also set `type="s elect"` to render a `<select>` tag

    * `type="checkbox"` is used exclusively to render boolean values

    * For live file uploads, see `Phoenix.Component.live_file_input/1`

  See https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input
  for more information.

  ## Examples

      <.input field={@form[:email]} type="email" />
      <.input name="my-input" errors={["oh no!"]} />
  """
  attr(:id, :any, default: nil)
  attr(:name, :any)
  attr(:label, :string, default: nil)
  attr(:value, :any)

  attr(:type, :string,
    default: "text",
    values: ~w(checkbox color date datetime-local email file hidden month number password
               range radio search select tel text textarea time url week)
  )

  attr(:field, Phoenix.HTML.FormField,
    doc: "a form field struct retrieved from the form, for example: @form[:email]"
  )

  attr(:errors, :list, default: [])
  attr(:checked, :boolean, doc: "the checked flag for checkbox inputs")
  attr(:prompt, :string, default: nil, doc: "the prompt for select inputs")
  attr(:options, :list, doc: "the options to pass to Phoenix.HTML.Form.options_for_select/2")
  attr(:multiple, :boolean, default: false, doc: "the multiple flag for select inputs")

  attr(:class, :string, default: nil)

  attr(:rest, :global,
    include: ~w(accept autocomplete capture cols disabled form list max maxlength min minlength
                multiple pattern placeholder readonly required rows size step data-1p-ignore)
  )

  slot(:inner_block)

  def input(%{field: %Phoenix.HTML.FormField{} = field} = assigns) do
    assigns
    |> assign(field: nil, id: assigns.id || field.id)
    |> assign(:errors, Enum.map(field.errors, &translate_error(&1)))
    |> assign_new(:name, fn -> if assigns.multiple, do: field.name <> "[]", else: field.name end)
    |> assign_new(:value, fn -> field.value end)
    |> input()
  end

  def input(%{type: "checkbox"} = assigns) do
    assigns =
      assign_new(assigns, :checked, fn ->
        HTML.Form.normalize_value("checkbox", assigns[:value])
      end)

    ~H"""
    <div phx-feedback-for={@name}>
      <label class="flex items-center gap-4 text-sm leading-6 text-white">
        <input type="hidden" name={@name} value="false" />
        <input
          type="checkbox"
          id={@id}
          name={@name}
          value="true"
          checked={@checked}
          class="rounded border-zinc-300 text-zinc-900 focus:ring-0"
          {@rest}
        />
        <%= @label %>
      </label>
      <.error :for={msg <- @errors}><%= msg %></.error>
    </div>
    """
  end

  def input(%{type: "select"} = assigns) do
    ~H"""
    <div phx-feedback-for={@name}>
      <.label for={@id}><%= @label %></.label>
      <select
        id={@id}
        name={@name}
        class="mt-2 block w-full rounded-md border border-gray-300 bg-white shadow-sm focus:border-zinc-400 focus:ring-0 sm:text-sm"
        multiple={@multiple}
        {@rest}
      >
        <option :if={@prompt} value=""><%= @prompt %></option>
        <%= Phoenix.HTML.Form.options_for_select(@options, @value) %>
      </select>
      <.error :for={msg <- @errors}><%= msg %></.error>
    </div>
    """
  end

  def input(%{type: "textarea"} = assigns) do
    ~H"""
    <div phx-feedback-for={@name}>
      <.label for={@id}><%= @label %></.label>
      <textarea
        id={@id}
        name={@name}
        class={[
          "mt-2 block w-full rounded-lg text-zinc-900 focus:ring-0 sm:text-sm sm:leading-6",
          "min-h-[6rem] phx-no-feedback:border-zinc-300 phx-no-feedback:focus:border-zinc-400",
          "transition-all",
          @errors == [] && "border-zinc-300 focus:border-zinc-400",
          @errors != [] && "border-rose-400 focus:border-rose-400"
        ]}
        {@rest}
      ><%= Phoenix.HTML.Form.normalize_value("textarea", @value) %></textarea>
      <.error :for={msg <- @errors}><%= msg %></.error>
    </div>
    """
  end

  # All other inputs text, datetime-local, url, password, etc. are handled here...
  def input(assigns) do
    ~H"""
    <div phx-feedback-for={@name}>
      <.label for={@id}><%= @label %></.label>
      <input
        type={@type}
        name={@name}
        id={@id}
        value={Phoenix.HTML.Form.normalize_value(@type, @value)}
        class={[
          "font-mono mt-2 block w-full rounded-elo-xs text-900 focus:ring-0 sm:text-sm sm:leading-6",
          "bg-opacity-0 bg-white",
          "transition-colors",
          "phx-no-feedback:border-white phx-no-feedback:border-opacity-5",
          "phx-no-feedback:focus:bg-opacity-5 phx-no-feedback:focus:border-opacity-20",
          @errors == [] && "border-zinc-300 focus:border-zinc-400",
          @errors != [] && "border-rose-400 focus:border-rose-400",
          @class
        ]}
        {@rest}
      />
      <.error :for={msg <- @errors}><%= msg %></.error>
    </div>
    """
  end

  @doc """
  Renders a label.
  """
  attr(:for, :string, default: nil)
  slot(:inner_block, required: true)

  def label(assigns) do
    ~H"""
    <label for={@for} class="block text-sm font-semibold leading-6 text-white">
      <%= render_slot(@inner_block) %>
    </label>
    """
  end

  @doc """
  Generates a generic error message.
  """
  slot(:inner_block, required: true)

  def error(assigns) do
    ~H"""
    <p class="mt-3 flex gap-3 text-sm leading-6 text-rose-600 phx-no-feedback:hidden">
      <.icon name="hero-exclamation-circle-mini" class="mt-0.5 h-5 w-5 flex-none" />
      <%= render_slot(@inner_block) %>
    </p>
    """
  end

  @doc """
  Renders a header with title.
  """
  attr(:class, :string, default: nil)

  slot(:inner_block, required: true)
  slot(:subtitle)
  slot(:actions)

  def header(assigns) do
    ~H"""
    <header class={[@actions != [] && "flex items-center justify-between gap-6", @class]}>
      <div>
        <h1 class="text-lg font-semibold leading-8 text-zinc-800">
          <%= render_slot(@inner_block) %>
        </h1>
        <p :if={@subtitle != []} class="mt-2 text-sm leading-6 text-zinc-600">
          <%= render_slot(@subtitle) %>
        </p>
      </div>
      <div class="flex-none"><%= render_slot(@actions) %></div>
    </header>
    """
  end

  @doc ~S"""
  Renders a table with generic styling.

  ## Examples

      <.table id="users" rows={@users}>
        <:col :let={user} label="id"><%= user.id %></:col>
        <:col :let={user} label="username"><%= user.username %></:col>
      </.table>
  """
  attr(:id, :string, required: true)
  attr(:rows, :list, required: true)
  attr(:row_id, :any, default: nil, doc: "the function for generating the row id")
  attr(:row_click, :any, default: nil, doc: "the function for handling phx-click on each row")

  attr(:row_item, :any,
    default: &Function.identity/1,
    doc: "the function for mapping each row before calling the :col and :action slots"
  )

  slot :col, required: true do
    attr(:label, :string)
  end

  slot(:action, doc: "the slot for showing user actions in the last table column")

  def table(assigns) do
    assigns =
      with %{rows: %Phoenix.LiveView.LiveStream{}} <- assigns do
        assign(assigns, row_id: assigns.row_id || fn {id, _item} -> id end)
      end

    ~H"""
    <div class="overflow-y-auto px-4 sm:overflow-visible sm:px-0">
      <table class="w-[40rem] mt-11 sm:w-full">
        <thead class="text-sm text-left leading-6 text-zinc-500">
          <tr>
            <th :for={col <- @col} class="p-0 pb-4 pr-6 font-normal"><%= col[:label] %></th>
            <th :if={@action != []} class="relative p-0 pb-4">
              <span class="sr-only"><%= gettext("Actions") %></span>
            </th>
          </tr>
        </thead>
        <tbody
          id={@id}
          phx-update={match?(%Phoenix.LiveView.LiveStream{}, @rows) && "stream"}
          class="relative divide-y divide-zinc-100 border-t border-zinc-200 text-sm leading-6 text-zinc-700"
        >
          <tr :for={row <- @rows} id={@row_id && @row_id.(row)} class="group hover:bg-zinc-50">
            <td
              :for={{col, i} <- Enum.with_index(@col)}
              phx-click={@row_click && @row_click.(row)}
              class={["relative p-0", @row_click && "hover:cursor-pointer"]}
            >
              <div class="block py-4 pr-6">
                <span class="absolute -inset-y-px right-0 -left-4 group-hover:bg-zinc-50 sm:rounded-l-xl" />
                <span class={["relative", i == 0 && "font-semibold text-zinc-900"]}>
                  <%= render_slot(col, @row_item.(row)) %>
                </span>
              </div>
            </td>
            <td :if={@action != []} class="relative w-14 p-0">
              <div class="relative whitespace-nowrap py-4 text-right text-sm font-medium">
                <span class="absolute -inset-y-px -right-4 left-0 group-hover:bg-zinc-50 sm:rounded-r-xl" />
                <span
                  :for={action <- @action}
                  class="relative ml-4 font-semibold leading-6 text-zinc-900 hover:text-zinc-700"
                >
                  <%= render_slot(action, @row_item.(row)) %>
                </span>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    """
  end

  @doc """
  Renders a data list.

  ## Examples

      <.list>
        <:item title="Title"><%= @post.title %></:item>
        <:item title="Views"><%= @post.views %></:item>
      </.list>
  """
  slot :item, required: true do
    attr(:title, :string, required: true)
  end

  def list(assigns) do
    ~H"""
    <div class="mt-14">
      <dl class="-my-4 divide-y divide-zinc-100">
        <div :for={item <- @item} class="flex gap-4 py-4 text-sm leading-6 sm:gap-8">
          <dt class="w-1/4 flex-none text-zinc-500"><%= item.title %></dt>
          <dd class="text-zinc-700"><%= render_slot(item) %></dd>
        </div>
      </dl>
    </div>
    """
  end

  @doc """
  Renders a back navigation link.

  ## Examples

      <.back navigate={~p"/posts"}>Back to posts</.back>
  """
  attr(:navigate, :any, required: true)
  slot(:inner_block, required: true)

  def back(assigns) do
    ~H"""
    <div class="mt-16">
      <.link
        navigate={@navigate}
        class="text-sm font-semibold leading-6 text-zinc-900 hover:text-zinc-700"
      >
        <.icon name="hero-arrow-left-solid" class="h-3 w-3" />
        <%= render_slot(@inner_block) %>
      </.link>
    </div>
    """
  end

  @doc """
  Renders a [Heroicon](https://heroicons.com).

  Heroicons come in three styles – outline, solid, and mini.
  By default, the outline style is used, but solid and mini may
  be applied by using the `-solid` and `-mini` suffix.

  You can customize the size and colors of the icons by setting
  width, height, and background color classes.

  Icons are extracted from your `assets/vendor/heroicons` directory and bundled
  within your compiled app.css by the plugin in your `assets/tailwind.config.js`.

  ## Examples

      <.icon name="hero-x-mark-solid" />
      <.icon name="hero-arrow-path" class="ml-1 w-3 h-3 animate-spin" />
  """
  attr(:name, :string, required: true)
  attr(:class, :string, default: nil)

  def icon(%{name: "hero-" <> _} = assigns) do
    ~H"""
    <span class={[@name, @class]} />
    """
  end

  ## JS Commands

  def show(js \\ %JS{}, selector) do
    JS.show(js,
      to: selector,
      transition:
        {"transition-all transform ease-out duration-300",
         "opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95",
         "opacity-100 translate-y-0 sm:scale-100"}
    )
  end

  def hide(js \\ %JS{}, selector) do
    JS.hide(js,
      to: selector,
      time: 200,
      transition:
        {"transition-all transform ease-in duration-200",
         "opacity-100 translate-y-0 sm:scale-100",
         "opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"}
    )
  end

  def toggle(js \\ %JS{}, selector) do
    JS.toggle(js,
      to: selector,
      in:
        {"transition-all transform ease-out duration-300",
         "opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95",
         "opacity-100 translate-y-0 sm:scale-100"},
      out:
        {"transition-all transform ease-in duration-200",
         "opacity-100 translate-y-0 sm:scale-100",
         "opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"}
    )
  end

  def show_modal(js \\ %JS{}, id) when is_binary(id) do
    js
    |> JS.show(to: "##{id}")
    |> JS.show(
      to: "##{id}-bg",
      transition: {"transition-all transform ease-out duration-300", "opacity-0", "opacity-100"}
    )
    |> show("##{id}-container")
    |> JS.add_class("overflow-hidden", to: "body")
    |> JS.focus_first(to: "##{id}-content")
  end

  def hide_modal(js \\ %JS{}, id) do
    js
    |> JS.hide(
      to: "##{id}-bg",
      transition: {"transition-all transform ease-in duration-200", "opacity-100", "opacity-0"}
    )
    |> hide("##{id}-container")
    |> JS.hide(to: "##{id}", transition: {"block", "block", "hidden"})
    |> JS.remove_class("overflow-hidden", to: "body")
    |> JS.pop_focus()
  end

  @doc """
  Translates an error message using gettext.
  """
  def translate_error({msg, opts}) do
    # When using gettext, we typically pass the strings we want
    # to translate as a static argument:
    #
    #     # Translate the number of files with plural rules
    #     dngettext("errors", "1 file", "%{count} files", count)
    #
    # However the error messages in our forms and APIs are generated
    # dynamically, so we need to translate them by calling Gettext
    # with our gettext backend as first argument. Translations are
    # available in the errors.po file (as we use the "errors" domain).
    if count = opts[:count] do
      Gettext.dngettext(ElodinDashboardWeb.Gettext, "errors", msg, msg, count, opts)
    else
      Gettext.dgettext(ElodinDashboardWeb.Gettext, "errors", msg, opts)
    end
  end

  @doc """
  Translates the errors for a field from a keyword list of errors.
  """
  def translate_errors(errors, field) when is_list(errors) do
    for {^field, {msg, opts}} <- errors, do: translate_error({msg, opts})
  end
end
