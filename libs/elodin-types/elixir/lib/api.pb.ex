defmodule Elodin.Types.Api.Sandbox.Status do
  @moduledoc false

  use Protobuf, enum: true, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :OFF, 0
  field :VM_BOOTING, 1
  field :ERROR, 2
  field :RUNNING, 3
end

defmodule Elodin.Types.Api.MonteCarloRun.Status do
  @moduledoc false

  use Protobuf, enum: true, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :PENDING, 0
  field :RUNNING, 1
  field :DONE, 2
end

defmodule Elodin.Types.Api.MonteCarloBatch.Status do
  @moduledoc false

  use Protobuf, enum: true, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :PENDING, 0
  field :RUNNING, 1
  field :DONE, 2
end

defmodule Elodin.Types.Api.CurrentUserReq do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"
end

defmodule Elodin.Types.Api.CurrentUserResp do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :id, 1, type: :bytes
  field :email, 2, type: :string
  field :name, 3, type: :string
  field :avatar, 4, type: :string
end

defmodule Elodin.Types.Api.CreateUserReq do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :email, 1, proto3_optional: true, type: :string
  field :name, 2, proto3_optional: true, type: :string
end

defmodule Elodin.Types.Api.CreateUserResp do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :id, 1, type: :bytes
end

defmodule Elodin.Types.Api.CreateSandboxReq do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :name, 1, type: :string
  field :code, 2, type: :string
  field :template, 3, proto3_optional: true, type: :string
end

defmodule Elodin.Types.Api.CreateSandboxResp do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :id, 1, type: :bytes
end

defmodule Elodin.Types.Api.UpdateSandboxReq do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :id, 1, type: :bytes
  field :name, 2, type: :string
  field :code, 3, proto3_optional: true, type: :string
  field :draft_code, 4, proto3_optional: true, type: :string, json_name: "draftCode"
  field :public, 5, type: :bool
end

defmodule Elodin.Types.Api.UpdateSandboxResp do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :errors, 1, repeated: true, type: :string
end

defmodule Elodin.Types.Api.BootSandboxReq do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :id, 1, type: :bytes
end

defmodule Elodin.Types.Api.BootSandboxResp do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"
end

defmodule Elodin.Types.Api.Sandbox do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :id, 1, type: :bytes
  field :name, 2, type: :string
  field :code, 3, type: :string
  field :status, 4, type: Elodin.Types.Api.Sandbox.Status, enum: true
  field :draft_code, 5, type: :string, json_name: "draftCode"
  field :public, 6, type: :bool
  field :user_id, 7, proto3_optional: true, type: :bytes, json_name: "userId"
end

defmodule Elodin.Types.Api.Page do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :last_id, 1, type: :bytes, json_name: "lastId"
  field :count, 2, type: :uint32
end

defmodule Elodin.Types.Api.ListSandboxesReq do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :page, 1, type: Elodin.Types.Api.Page
end

defmodule Elodin.Types.Api.ListSandboxesResp do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :sandboxes, 1, repeated: true, type: Elodin.Types.Api.Sandbox
  field :next_page, 2, type: Elodin.Types.Api.Page, json_name: "nextPage"
end

defmodule Elodin.Types.Api.GetSandboxReq do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :id, 1, type: :bytes
end

defmodule Elodin.Types.Api.DeleteSandboxReq do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :id, 1, type: :bytes
end

defmodule Elodin.Types.Api.ListMonteCarloRunsReq do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"
end

defmodule Elodin.Types.Api.ListMonteCarloRunsResp do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :monte_carlo_runs, 1,
    repeated: true,
    type: Elodin.Types.Api.MonteCarloRun,
    json_name: "monteCarloRuns"
end

defmodule Elodin.Types.Api.CreateMonteCarloRunReq do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :name, 1, type: :string
  field :samples, 2, type: :uint32
  field :max_duration, 3, type: :uint64, json_name: "maxDuration"
end

defmodule Elodin.Types.Api.CreateMonteCarloRunResp do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :id, 1, type: :bytes
  field :upload_url, 2, type: :string, json_name: "uploadUrl"
end

defmodule Elodin.Types.Api.StartMonteCarloRunReq do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :id, 1, type: :bytes
end

defmodule Elodin.Types.Api.StartMonteCarloRunResp do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"
end

defmodule Elodin.Types.Api.GetMonteCarloRunReq do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :id, 1, type: :bytes
end

defmodule Elodin.Types.Api.MonteCarloRun do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :id, 1, type: :bytes
  field :name, 2, type: :string
  field :samples, 3, type: :uint32
  field :max_duration, 4, type: :uint64, json_name: "maxDuration"
  field :status, 5, type: Elodin.Types.Api.MonteCarloRun.Status, enum: true
  field :metadata, 6, type: :string
  field :started, 7, proto3_optional: true, type: :uint64
  field :batches, 8, repeated: true, type: Elodin.Types.Api.MonteCarloBatch
end

defmodule Elodin.Types.Api.MonteCarloBatch do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :run_id, 1, type: :bytes, json_name: "runId"
  field :batch_number, 2, type: :uint32, json_name: "batchNumber"
  field :samples, 3, type: :uint32
  field :failures, 4, type: :bytes
  field :finished_time, 5, proto3_optional: true, type: :uint64, json_name: "finishedTime"
  field :status, 6, type: Elodin.Types.Api.MonteCarloBatch.Status, enum: true
end

defmodule Elodin.Types.Api.GetSampleResultsReq do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :id, 1, type: :bytes
  field :sample_number, 2, type: :uint32, json_name: "sampleNumber"
end

defmodule Elodin.Types.Api.GetSampleResultsResp do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :download_url, 1, type: :string, json_name: "downloadUrl"
end

defmodule Elodin.Types.Api.Api.Service do
  @moduledoc false

  use GRPC.Service, name: "elodin.types.api.Api", protoc_gen_elixir_version: "0.12.0"

  rpc :CreateUser, Elodin.Types.Api.CreateUserReq, Elodin.Types.Api.CreateUserResp

  rpc :CurrentUser, Elodin.Types.Api.CurrentUserReq, Elodin.Types.Api.CurrentUserResp

  rpc :GetSandbox, Elodin.Types.Api.GetSandboxReq, Elodin.Types.Api.Sandbox

  rpc :DeleteSandbox, Elodin.Types.Api.DeleteSandboxReq, Elodin.Types.Api.Sandbox

  rpc :ListSandboxes, Elodin.Types.Api.ListSandboxesReq, Elodin.Types.Api.ListSandboxesResp

  rpc :CreateSandbox, Elodin.Types.Api.CreateSandboxReq, Elodin.Types.Api.CreateSandboxResp

  rpc :UpdateSandbox, Elodin.Types.Api.UpdateSandboxReq, Elodin.Types.Api.UpdateSandboxResp

  rpc :BootSandbox, Elodin.Types.Api.BootSandboxReq, Elodin.Types.Api.BootSandboxResp

  rpc :SandboxEvents, Elodin.Types.Api.GetSandboxReq, stream(Elodin.Types.Api.Sandbox)

  rpc :ListMonteCarloRuns,
      Elodin.Types.Api.ListMonteCarloRunsReq,
      Elodin.Types.Api.ListMonteCarloRunsResp

  rpc :CreateMonteCarloRun,
      Elodin.Types.Api.CreateMonteCarloRunReq,
      Elodin.Types.Api.CreateMonteCarloRunResp

  rpc :StartMonteCarloRun,
      Elodin.Types.Api.StartMonteCarloRunReq,
      Elodin.Types.Api.StartMonteCarloRunResp

  rpc :GetMonteCarloRun, Elodin.Types.Api.GetMonteCarloRunReq, Elodin.Types.Api.MonteCarloRun

  rpc :MonteCarloRunEvents,
      Elodin.Types.Api.GetMonteCarloRunReq,
      stream(Elodin.Types.Api.MonteCarloRun)

  rpc :MonteCarloBatchEvents,
      Elodin.Types.Api.GetMonteCarloRunReq,
      stream(Elodin.Types.Api.MonteCarloBatch)

  rpc :GetSampleResults,
      Elodin.Types.Api.GetSampleResultsReq,
      Elodin.Types.Api.GetSampleResultsResp
end

defmodule Elodin.Types.Api.Api.Stub do
  @moduledoc false

  use GRPC.Stub, service: Elodin.Types.Api.Api.Service
end
