defmodule ElodinDashboardWeb.IconComponents do
  use Phoenix.Component

  attr(:class, :string, default: "")

  def arrow_left(assigns) do
    ~H"""
    <img src="/images/arrow-left.svg" class={["w-[24px] height-[24px]", @class]} />
    """
  end

  attr(:class, :string, default: "")

  attr(:rest, :global)

  def spinner(assigns) do
    ~H"""
    <img src="/images/spinner.svg" class={["w-[24px] height-[24px]", @class]} {@rest} />
    """
  end

  attr(:class, :string, default: "")

  attr(:rest, :global)

  def x(assigns) do
    ~H"""
    <img src="/images/x.svg" class={["w-[24px] height-[24px]", @class]} {@rest} />
    """
  end

  attr(:class, :string, default: "")

  def lightning(assigns) do
    ~H"""
    <svg
      width="12"
      height="12"
      viewBox="0 0 12 12"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      class={@class}
    >
      <path d="M6.55556 0L1 7.2H6L5.44444 12L11 4.8H6L6.55556 0Z" fill="currentColor" />
    </svg>
    """
  end

  def arrow_chevron_up(assigns) do
    ~H"""
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      class={@class}
    >
      <g clip-path="url(#clip0_165_557)">
        <path
          d="M4.94 10.2733L8 7.21998L11.06 10.2733L12 9.33331L8 5.33331L4 9.33331L4.94 10.2733Z"
          fill="currentColor"
        />
      </g>
      <defs>
        <clipPath id="clip0_165_557">
          <rect width="16" height="16" fill="currentColor" />
        </clipPath>
      </defs>
    </svg>
    """
  end
end
