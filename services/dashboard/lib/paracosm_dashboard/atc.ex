defmodule ParacosmDashboard.AtcAgent do
  def start_link(_params) do
    Agent.start_link(fn ->
      # GRPC.Stub.connect(Application.get_env(:paracosm_dashboard, ParacosmDashboard.Atc)[:addr])
      {:ok, channel} =
        GRPC.Stub.connect("localhost:50051")

      channel
    end)
  end

  def create_user(pid, request, token) do
    Agent.get(pid, fn channel ->
      channel
      |> Paracosm.Types.Api.Api.Stub.create_user(request = request,
        metadata: %{"Authorization" => "Bearer #{token}"}
      )
    end)
  end

  def create_sandbox(pid, request, token) do
    Agent.get(pid, fn channel ->
      channel
      |> Paracosm.Types.Api.Api.Stub.create_sandbox(request = request,
        metadata: %{"Authorization" => "Bearer #{token}"}
      )
    end)
  end

  def boot_sandbox(pid, request, token) do
    Agent.get(pid, fn channel ->
      channel
      |> Paracosm.Types.Api.Api.Stub.boot_sandbox(request = request,
        metadata: %{"Authorization" => "Bearer #{token}"}
      )
    end)
  end
end

defmodule ParacosmDashboard.Atc do
  def create_user(request, token) do
    :poolboy.transaction(:atc, fn pid ->
      ParacosmDashboard.AtcAgent.create_user(pid, request, token)
    end)
  end

  def create_sandbox(request, token) do
    :poolboy.transaction(:atc, fn pid ->
      ParacosmDashboard.AtcAgent.create_sandbox(pid, request, token)
    end)
  end

  def boot_sandbox(request, token) do
    :poolboy.transaction(:atc, fn pid ->
      ParacosmDashboard.AtcAgent.boot_sandbox(pid, request, token)
    end)
  end
end
