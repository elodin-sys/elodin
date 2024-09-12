defmodule ElodinDashboardWeb.SandboxHNTemplateLive do
  use ElodinDashboardWeb, :live_view
  alias Elodin.Types.Api
  alias ElodinDashboard.Atc
  alias ElodinDashboard.NameGen
  import ElodinDashboardWeb.CoreComponents
  import ElodinDashboardWeb.SandboxComponents
  import ElodinDashboardWeb.NavbarComponents
  import ElodinDashboardWeb.ModalComponents

  def mount(%{"template" => template}, _, socket) do
    case Atc.create_sandbox(%Api.CreateSandboxReq{name: template, template: template}, "") do
      {:ok, sandbox} ->
        id = UUID.binary_to_string!(sandbox.id)

        {:ok,
         socket
         |> put_flash(:info, "Successfully created sandbox")
         |> redirect(to: ~p"/sandbox/#{id}")}

      err ->
        {:ok, socket |> put_flash(:error, "Error creating sandbox: #{err}")}
    end
  end

  def render(assigns) do
    ~H"""
    """
  end
end
