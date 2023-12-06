defmodule Paracosm.Types.Api.Sandbox.Status do
  @moduledoc false

  use Protobuf, enum: true, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :OFF, 0
  field :VM_BOOTING, 1
  field :ERROR, 2
  field :RUNNING, 3
end

defmodule Paracosm.Types.Api.CurrentUserReq do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"
end

defmodule Paracosm.Types.Api.CurrentUserResp do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :id, 1, type: :bytes
  field :email, 2, type: :string
  field :name, 3, type: :string
  field :avatar, 4, type: :string
end

defmodule Paracosm.Types.Api.CreateUserReq do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :email, 1, proto3_optional: true, type: :string
  field :name, 2, proto3_optional: true, type: :string
end

defmodule Paracosm.Types.Api.CreateUserResp do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :id, 1, type: :bytes
end

defmodule Paracosm.Types.Api.CreateSandboxReq do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :name, 1, type: :string
  field :code, 2, type: :string
end

defmodule Paracosm.Types.Api.CreateSandboxResp do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :id, 1, type: :bytes
end

defmodule Paracosm.Types.Api.UpdateSandboxReq do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :id, 1, type: :bytes
  field :name, 2, type: :string
  field :code, 3, type: :string
end

defmodule Paracosm.Types.Api.UpdateSandboxResp do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"
end

defmodule Paracosm.Types.Api.BootSandboxReq do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :id, 1, type: :bytes
end

defmodule Paracosm.Types.Api.BootSandboxResp do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"
end

defmodule Paracosm.Types.Api.Sandbox do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :id, 1, type: :bytes
  field :name, 2, type: :string
  field :code, 3, type: :string
  field :status, 4, type: Paracosm.Types.Api.Sandbox.Status, enum: true
end

defmodule Paracosm.Types.Api.Page do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :last_id, 1, type: :bytes, json_name: "lastId"
  field :count, 2, type: :uint32
end

defmodule Paracosm.Types.Api.ListSandboxesReq do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :page, 1, type: Paracosm.Types.Api.Page
end

defmodule Paracosm.Types.Api.ListSandboxesResp do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :sandboxes, 1, repeated: true, type: Paracosm.Types.Api.Sandbox
  field :next_page, 2, type: Paracosm.Types.Api.Page, json_name: "nextPage"
end

defmodule Paracosm.Types.Api.GetSandboxReq do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :id, 1, type: :bytes
end

defmodule Paracosm.Types.Api.Api.Service do
  @moduledoc false

  use GRPC.Service, name: "paracosm.types.api.Api", protoc_gen_elixir_version: "0.12.0"

  rpc :CreateUser, Paracosm.Types.Api.CreateUserReq, Paracosm.Types.Api.CreateUserResp

  rpc :CurrentUser, Paracosm.Types.Api.CurrentUserReq, Paracosm.Types.Api.CurrentUserResp

  rpc :GetSandbox, Paracosm.Types.Api.GetSandboxReq, Paracosm.Types.Api.Sandbox

  rpc :ListSandboxes, Paracosm.Types.Api.ListSandboxesReq, Paracosm.Types.Api.ListSandboxesResp

  rpc :CreateSandbox, Paracosm.Types.Api.CreateSandboxReq, Paracosm.Types.Api.CreateSandboxResp

  rpc :UpdateSandbox, Paracosm.Types.Api.UpdateSandboxReq, Paracosm.Types.Api.UpdateSandboxResp

  rpc :BootSandbox, Paracosm.Types.Api.BootSandboxReq, Paracosm.Types.Api.BootSandboxResp

  rpc :SandboxEvents, Paracosm.Types.Api.GetSandboxReq, stream(Paracosm.Types.Api.Sandbox)
end

defmodule Paracosm.Types.Api.Api.Stub do
  @moduledoc false

  use GRPC.Stub, service: Paracosm.Types.Api.Api.Service
end