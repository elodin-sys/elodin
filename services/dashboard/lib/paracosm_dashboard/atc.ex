defmodule ParacosmDashboard.AtcAgent do
  def start_link(_params) do
    Agent.start_link(fn ->
      addr = Application.get_env(:paracosm_dashboard, ParacosmDashboard.Atc)[:internal_addr]

      {:ok, channel} =
        GRPC.Stub.connect(addr, adapter_opts: [retry_timeout: 10])

      channel
    end)
  end

  def current_user(pid, request, token) do
    Agent.get(pid, fn channel ->
      channel
      |> Paracosm.Types.Api.Api.Stub.current_user(request = request,
        metadata: %{"Authorization" => "Bearer #{token}"}
      )
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

  def get_sandbox(pid, request, token) do
    Agent.get(pid, fn channel ->
      channel
      |> Paracosm.Types.Api.Api.Stub.get_sandbox(request = request,
        metadata: %{"Authorization" => "Bearer #{token}"}
      )
    end)
  end

  def sandbox_events(pid, request, token) do
    Agent.get(pid, fn channel ->
      channel
      |> Paracosm.Types.Api.Api.Stub.sandbox_events(request = request,
        metadata: %{"Authorization" => "Bearer #{token}"}
      )
    end)
  end

  def with_channel(pid, closure) do
    Agent.get(pid, closure)
  end
end

defmodule ParacosmDashboard.Atc do
  def current_user(request, token) do
    :poolboy.transaction(:atc, fn pid ->
      ParacosmDashboard.AtcAgent.current_user(pid, request, token)
    end)
  end

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

  def get_sandbox(request, token) do
    :poolboy.transaction(:atc, fn pid ->
      ParacosmDashboard.AtcAgent.get_sandbox(pid, request, token)
    end)
  end

  # def sandbox_events(request, token) do
  #   :poolboy.transaction(:atc, fn pid ->
  #     ParacosmDashboard.AtcAgent.sandbox_events(pid, request, token)
  #   end)
  # end
  def with_channel(closure) do
    :poolboy.transaction(:atc, fn pid ->
      ParacosmDashboard.AtcAgent.with_channel(pid, closure)
    end)
  end
end
