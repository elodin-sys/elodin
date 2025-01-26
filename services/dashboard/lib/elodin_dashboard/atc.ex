defmodule ElodinDashboard.AtcAgent do
  def start_link(_params) do
    Agent.start_link(fn ->
      addr = Application.get_env(:elodin_dashboard, ElodinDashboard.Atc)[:internal_addr]

      case GRPC.Stub.connect(addr, adapter_opts: [retry_timeout: 4]) do
        {:ok, channel} -> channel
        {:error, _} -> nil
      end
    end)
  end

  def with_channel(pid, closure) do
    Agent.get_and_update(
      pid,
      fn channel ->
        channel |> call_closure(closure)
      end,
      20_000
    )
  end

  defp call_closure(channel, closure) when channel == nil do
    addr = Application.get_env(:elodin_dashboard, ElodinDashboard.Atc)[:internal_addr]

    case GRPC.Stub.connect(addr, adapter_opts: [retry_timeout: 4]) do
      {:ok, channel} ->
        call_closure(channel, closure)

      {:error, err} ->
        {{:error, err}, nil}
    end
  end

  defp call_closure(channel, closure) when channel != nil do
    case closure.(channel) do
      {:ok, res} ->
        {{:ok, res}, channel}

      {:error, err = %GRPC.RPCError{status: 4, message: "timeout when waiting for server"}} ->
        {{:error, err}, nil}

      {:error, err} ->
        {{:error, err}, channel}
    end
  end

  def current_user(pid, request, token) do
    ElodinDashboard.AtcAgent.with_channel(pid, fn channel ->
      channel
      |> Elodin.Types.Api.Api.Stub.current_user(request,
        metadata: %{"Authorization" => "Bearer #{token}"},
        timeout: 2000
      )
    end)
  end

  def create_user(pid, request, token) do
    ElodinDashboard.AtcAgent.with_channel(pid, fn channel ->
      channel
      |> Elodin.Types.Api.Api.Stub.create_user(request,
        metadata: %{"Authorization" => "Bearer #{token}"}
      )
    end)
  end

  def update_user(pid, request, token) do
    ElodinDashboard.AtcAgent.with_channel(pid, fn channel ->
      channel
      |> Elodin.Types.Api.Api.Stub.update_user(request,
        metadata: %{"Authorization" => "Bearer #{token}"}
      )
    end)
  end

  def create_billing_account(pid, request, token) do
    ElodinDashboard.AtcAgent.with_channel(pid, fn channel ->
      channel
      |> Elodin.Types.Api.Api.Stub.create_billing_account(request,
        metadata: %{"Authorization" => "Bearer #{token}"}
      )
    end)
  end
end

defmodule ElodinDashboard.Atc do
  @timeout 20_000
  def current_user(request, token) do
    :poolboy.transaction(
      :atc,
      fn pid ->
        ElodinDashboard.AtcAgent.current_user(pid, request, token)
      end,
      @timeout
    )
  end

  def create_user(request, token) do
    :poolboy.transaction(
      :atc,
      fn pid ->
        ElodinDashboard.AtcAgent.create_user(pid, request, token)
      end,
      @timeout
    )
  end

  def update_user(request, token) do
    :poolboy.transaction(
      :atc,
      fn pid ->
        ElodinDashboard.AtcAgent.update_user(pid, request, token)
      end,
      @timeout
    )
  end

  def create_billing_account(request, token) do
    :poolboy.transaction(
      :atc,
      fn pid ->
        ElodinDashboard.AtcAgent.create_billing_account(pid, request, token)
      end,
      @timeout
    )
  end

  def with_channel(closure) do
    :poolboy.transaction(
      :atc,
      fn pid ->
        ElodinDashboard.AtcAgent.with_channel(pid, closure)
      end,
      @timeout
    )
  end
end
