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
end
