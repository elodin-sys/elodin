defmodule ParacosmDashboard.AtcAgent do
  def start_link(_params) do
    Agent.start_link(fn ->
      addr = Application.get_env(:paracosm_dashboard, ParacosmDashboard.Atc)[:internal_addr]

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
        channel =
          if is_nil(channel) do
            addr = Application.get_env(:paracosm_dashboard, ParacosmDashboard.Atc)[:internal_addr]

            case GRPC.Stub.connect(addr, adapter_opts: [retry_timeout: 4]) do
              {:ok, channel} ->
                case closure.(channel) do
                  {:ok, res} ->
                    {{:ok, res}, channel}

                  {:error,
                   err = %GRPC.RPCError{status: 4, message: "timeout when waiting for server"}} ->
                    {{:error, err}, nil}

                  {:error, err} ->
                    {{:err, err}, channel}
                end

              {:error, err} ->
                {{:error, err}, nil}
            end
          else
            case closure.(channel) do
              {:ok, res} ->
                {{:ok, res}, channel}

              {:error,
               err = %GRPC.RPCError{status: 4, message: "timeout when waiting for server"}} ->
                {{:error, err}, nil}

              {:error, err} ->
                {{:error, err}, channel}
            end
          end
      end,
      20_000
    )
  end

  def current_user(pid, request, token) do
    ParacosmDashboard.AtcAgent.with_channel(pid, fn channel ->
      channel
      |> Paracosm.Types.Api.Api.Stub.current_user(request,
        metadata: %{"Authorization" => "Bearer #{token}"},
        timeout: 2000
      )
    end)
  end

  def create_user(pid, request, token) do
    ParacosmDashboard.AtcAgent.with_channel(pid, fn channel ->
      channel
      |> Paracosm.Types.Api.Api.Stub.create_user(request,
        metadata: %{"Authorization" => "Bearer #{token}"}
      )
    end)
  end

  def create_sandbox(pid, request, token) do
    ParacosmDashboard.AtcAgent.with_channel(pid, fn channel ->
      channel
      |> Paracosm.Types.Api.Api.Stub.create_sandbox(request,
        metadata: %{"Authorization" => "Bearer #{token}"}
      )
    end)
  end

  def update_sandbox(pid, request, token) do
    ParacosmDashboard.AtcAgent.with_channel(pid, fn channel ->
      channel
      |> Paracosm.Types.Api.Api.Stub.update_sandbox(request,
        metadata: %{"Authorization" => "Bearer #{token}"}
      )
    end)
  end

  def boot_sandbox(pid, request, token) do
    ParacosmDashboard.AtcAgent.with_channel(pid, fn channel ->
      channel
      |> Paracosm.Types.Api.Api.Stub.boot_sandbox(request,
        metadata: %{"Authorization" => "Bearer #{token}"}
      )
    end)
  end

  def get_sandbox(pid, request, token) do
    ParacosmDashboard.AtcAgent.with_channel(pid, fn channel ->
      channel
      |> Paracosm.Types.Api.Api.Stub.get_sandbox(request,
        metadata: %{"Authorization" => "Bearer #{token}"}
      )
    end)
  end

  def list_sandboxes(pid, request, token) do
    ParacosmDashboard.AtcAgent.with_channel(pid, fn channel ->
      channel
      |> Paracosm.Types.Api.Api.Stub.list_sandboxes(request,
        metadata: %{"Authorization" => "Bearer #{token}"}
      )
    end)
  end

  def sandbox_events(pid, request, token) do
    ParacosmDashboard.AtcAgent.with_channel(pid, fn channel ->
      channel
      |> Paracosm.Types.Api.Api.Stub.sandbox_events(request,
        metadata: %{"Authorization" => "Bearer #{token}"}
      )
    end)
  end
end

defmodule ParacosmDashboard.Atc do
  @timeout 20_000
  def current_user(request, token) do
    :poolboy.transaction(
      :atc,
      fn pid ->
        ParacosmDashboard.AtcAgent.current_user(pid, request, token)
      end,
      @timeout
    )
  end

  def create_user(request, token) do
    :poolboy.transaction(
      :atc,
      fn pid ->
        ParacosmDashboard.AtcAgent.create_user(pid, request, token)
      end,
      @timeout
    )
  end

  def create_sandbox(request, token) do
    :poolboy.transaction(
      :atc,
      fn pid ->
        ParacosmDashboard.AtcAgent.create_sandbox(pid, request, token)
      end,
      @timeout
    )
  end

  def update_sandbox(request, token) do
    :poolboy.transaction(
      :atc,
      fn pid ->
        ParacosmDashboard.AtcAgent.update_sandbox(pid, request, token)
      end,
      @timeout
    )
  end

  def boot_sandbox(request, token) do
    :poolboy.transaction(
      :atc,
      fn pid ->
        ParacosmDashboard.AtcAgent.boot_sandbox(pid, request, token)
      end,
      @timeout
    )
  end

  def get_sandbox(request, token) do
    :poolboy.transaction(
      :atc,
      fn pid ->
        ParacosmDashboard.AtcAgent.get_sandbox(pid, request, token)
      end,
      @timeout
    )
  end

  def list_sandboxes(request, token) do
    :poolboy.transaction(
      :atc,
      fn pid ->
        ParacosmDashboard.AtcAgent.list_sandboxes(pid, request, token)
      end,
      @timeout
    )
  end

  def with_channel(closure) do
    :poolboy.transaction(
      :atc,
      fn pid ->
        ParacosmDashboard.AtcAgent.with_channel(pid, closure)
      end,
      @timeout
    )
  end
end
