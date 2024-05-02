defmodule Elodin.Types.Sandbox.Status do
  @moduledoc false

  use Protobuf, enum: true, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :Success, 0
  field :Error, 1
end

defmodule Elodin.Types.Sandbox.UpdateCodeReq do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :code, 1, type: :string
end

defmodule Elodin.Types.Sandbox.UpdateCodeResp do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :status, 1, type: Elodin.Types.Sandbox.Status, enum: true
  field :errors, 2, repeated: true, type: :string
end

defmodule Elodin.Types.Sandbox.BuildReq do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :code, 1, type: :string
end

defmodule Elodin.Types.Sandbox.BuildResp do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :artifacts_file, 1, type: :string, json_name: "artifactsFile"
end

defmodule Elodin.Types.Sandbox.TestReq do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :results_file, 2, type: :string, json_name: "resultsFile"
end

defmodule Elodin.Types.Sandbox.TestResp do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :report, 1, type: :string
end

defmodule Elodin.Types.Sandbox.FileChunk do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :name, 1, type: :string
  field :data, 2, type: :bytes
end

defmodule Elodin.Types.Sandbox.FileTransferReq do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :name, 1, type: :string
end

defmodule Elodin.Types.Sandbox.FileTransferResp do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"
end

defmodule Elodin.Types.Sandbox.SandboxControl.Service do
  @moduledoc false

  use GRPC.Service,
    name: "elodin.types.sandbox.SandboxControl",
    protoc_gen_elixir_version: "0.12.0"

  rpc :UpdateCode, Elodin.Types.Sandbox.UpdateCodeReq, Elodin.Types.Sandbox.UpdateCodeResp
end

defmodule Elodin.Types.Sandbox.SandboxControl.Stub do
  @moduledoc false

  use GRPC.Stub, service: Elodin.Types.Sandbox.SandboxControl.Service
end

defmodule Elodin.Types.Sandbox.Sandbox.Service do
  @moduledoc false

  use GRPC.Service, name: "elodin.types.sandbox.Sandbox", protoc_gen_elixir_version: "0.12.0"

  rpc :Build, Elodin.Types.Sandbox.BuildReq, Elodin.Types.Sandbox.BuildResp

  rpc :Test, Elodin.Types.Sandbox.TestReq, Elodin.Types.Sandbox.TestResp

  rpc :SendFile, stream(Elodin.Types.Sandbox.FileChunk), Elodin.Types.Sandbox.FileTransferResp

  rpc :RecvFile, Elodin.Types.Sandbox.FileTransferReq, stream(Elodin.Types.Sandbox.FileChunk)
end

defmodule Elodin.Types.Sandbox.Sandbox.Stub do
  @moduledoc false

  use GRPC.Stub, service: Elodin.Types.Sandbox.Sandbox.Service
end