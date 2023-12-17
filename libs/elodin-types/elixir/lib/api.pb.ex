defmodule Elodin.Types.Api.Sandbox.Status do
  @moduledoc false

  use Protobuf, enum: true, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"

  field :OFF, 0
  field :VM_BOOTING, 1
  field :ERROR, 2
  field :RUNNING, 3
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
  field :code, 3, type: :string
end

defmodule Elodin.Types.Api.UpdateSandboxResp do
  @moduledoc false

  use Protobuf, syntax: :proto3, protoc_gen_elixir_version: "0.12.0"
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

defmodule Elodin.Types.Api.Api.Service do
  @moduledoc false

  use GRPC.Service, name: "elodin.types.api.Api", protoc_gen_elixir_version: "0.12.0"

  rpc :CreateUser, Elodin.Types.Api.CreateUserReq, Elodin.Types.Api.CreateUserResp

  rpc :CurrentUser, Elodin.Types.Api.CurrentUserReq, Elodin.Types.Api.CurrentUserResp

  rpc :GetSandbox, Elodin.Types.Api.GetSandboxReq, Elodin.Types.Api.Sandbox

  rpc :ListSandboxes, Elodin.Types.Api.ListSandboxesReq, Elodin.Types.Api.ListSandboxesResp

  rpc :CreateSandbox, Elodin.Types.Api.CreateSandboxReq, Elodin.Types.Api.CreateSandboxResp

  rpc :UpdateSandbox, Elodin.Types.Api.UpdateSandboxReq, Elodin.Types.Api.UpdateSandboxResp

  rpc :BootSandbox, Elodin.Types.Api.BootSandboxReq, Elodin.Types.Api.BootSandboxResp

  rpc :SandboxEvents, Elodin.Types.Api.GetSandboxReq, stream(Elodin.Types.Api.Sandbox)
end

defmodule Elodin.Types.Api.Api.Stub do
  @moduledoc false

  use GRPC.Stub, service: Elodin.Types.Api.Api.Service
end