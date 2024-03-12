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

  field :artifacts, 1, type: :bytes
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

defmodule Elodin.Types.Sandbox.BuildSim.Service do
  @moduledoc false

  use GRPC.Service, name: "elodin.types.sandbox.BuildSim", protoc_gen_elixir_version: "0.12.0"

  rpc :Build, Elodin.Types.Sandbox.BuildReq, Elodin.Types.Sandbox.BuildResp
end

defmodule Elodin.Types.Sandbox.BuildSim.Stub do
  @moduledoc false

  use GRPC.Stub, service: Elodin.Types.Sandbox.BuildSim.Service
end