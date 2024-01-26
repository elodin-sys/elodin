defmodule Elodin.Types.Sandbox.Status do
  @moduledoc false

  use Protobuf, enum: true, protoc_gen_elixir_version: "0.12.0", syntax: :proto3

  field :Success, 0
  field :Error, 1
end

defmodule Elodin.Types.Sandbox.UpdateCodeReq do
  @moduledoc false

  use Protobuf, protoc_gen_elixir_version: "0.12.0", syntax: :proto3

  field :code, 1, type: :string
end

defmodule Elodin.Types.Sandbox.UpdateCodeResp do
  @moduledoc false

  use Protobuf, protoc_gen_elixir_version: "0.12.0", syntax: :proto3

  field :status, 1, type: Elodin.Types.Sandbox.Status, enum: true
  field :errors, 2, repeated: true, type: :string
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