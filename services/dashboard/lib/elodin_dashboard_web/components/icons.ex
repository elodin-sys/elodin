defmodule ElodinDashboardWeb.IconComponents do
  use Phoenix.Component

  attr(:class, :string, default: "")

  attr(:rest, :global)

  def ologo(assigns) do
    ~H"""
    <div class={["w-[24px] height-[24px]", @class]}>
      <svg
        class="w-full h-full"
        width="21"
        height="26"
        viewBox="0 0 21 26"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          d="M10.5172 0.5H15.5172C18.2787 0.5 20.5172 2.73858 20.5172 5.5V20.5001C20.5172 23.2615 18.2787 25.5001 15.5172 25.5001H5.51724C2.75577 25.5001 0.517242 23.2615 0.517242 20.5V5.5C0.517242 2.73858 2.75577 0.500002 5.51724 0.500002V5.5V20.5001H15.5172V5.5L5.51724 5.5C5.51724 2.73858 7.75577 0.5 10.5172 0.5Z"
          fill="currentColor"
        />
      </svg>
    </div>
    """
  end

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
    <div class={["w-[24px] height-[24px]", @class]}>
      <svg class="w-full h-full" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path
          fill-rule="evenodd"
          clip-rule="evenodd"
          d="M19.5 11C19.5 15.6944 15.6944 19.5 11 19.5C6.30558 19.5 2.5 15.6944 2.5 11C2.5 6.30558 6.30558 2.5 11 2.5V0C4.92487 0 0 4.92487 0 11C0 17.0751 4.92487 22 11 22C17.0751 22 22 17.0751 22 11H19.5Z"
          fill="currentColor"
        />
      </svg>
    </div>
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

  attr(:class, :string, default: "")

  def stop(assigns) do
    ~H"""
    <svg width="19" height="18" fill="none" xmlns="http://www.w3.org/2000/svg" class={@class}>
      <g clip-path="url(#clip0_894_699)">
        <path
          fill-rule="evenodd"
          clip-rule="evenodd"
          d="M9.5 15C12.8137 15 15.5 12.3137 15.5 9C15.5 5.68629 12.8137 3 9.5 3C6.18629 3 3.5 5.68629 3.5 9C3.5 12.3137 6.18629 15 9.5 15ZM7.5 6C6.94772 6 6.5 6.44772 6.5 7V11C6.5 11.5523 6.94772 12 7.5 12H11.5C12.0523 12 12.5 11.5523 12.5 11V7C12.5 6.44772 12.0523 6 11.5 6H7.5Z"
          fill="#E94B14"
        />
      </g>
      <defs>
        <clipPath id="clip0_894_699">
          <rect width="18" height="18" fill="white" transform="translate(0.5)" />
        </clipPath>
      </defs>
    </svg>
    """
  end

  def icon_link(assigns) do
    ~H"""
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">
      <g id="icon_18_external_link" clip-path="url(#clip0_927_406)">
        <path
          id="Vector (Stroke)"
          fill-rule="evenodd"
          clip-rule="evenodd"
          d="M4.5 4.5L13.5533 4.5L13.5533 13.5L12.0533 13.5L12.0533 6L4.5 6V4.5Z"
          fill="#FFFBF0"
        />
        <rect
          id="Rectangle 659"
          x="11.6519"
          y="5.25"
          width="1.5"
          height="9.49847"
          transform="rotate(45 11.6519 5.25)"
          fill="#FFFBF0"
        />
      </g>
      <defs>
        <clipPath id="clip0_927_406">
          <rect width="18" height="18" fill="white" />
        </clipPath>
      </defs>
    </svg>
    """
  end

  def icon_copy(assigns) do
    ~H"""
    <svg width="19" height="18" viewBox="0 0 19 18" fill="none" xmlns="http://www.w3.org/2000/svg">
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
    """
  end
end
