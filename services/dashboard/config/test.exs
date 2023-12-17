import Config

# Only in tests, remove the complexity from the password hashing algorithm
config :bcrypt_elixir, :log_rounds, 1

# We don't run a server during test. If one is required,
# you can enable the server option below.
config :elodin_dashboard, ElodinDashboardWeb.Endpoint,
  http: [ip: {127, 0, 0, 1}, port: 4002],
  secret_key_base: "XWrHUuJkZCZKVZwM3rfec7YyKWBsE/cS3tjRba7dIy+KMZ0UIkkRDHeKbZpWpd7X",
  server: false

config :elodin_dashboard, ParacasmDashboardWeb.Atc, addr: "localhost:50051"

# In test we don't send emails.
config :elodin_dashboard, ElodinDashboard.Mailer, adapter: Swoosh.Adapters.Test

# Disable swoosh api client as it is only required for production adapters.
config :swoosh, :api_client, false

# Print only warnings and errors during test
config :logger, level: :warning

# Initialize plugs at runtime for faster test compilation
config :phoenix, :plug_init_mode, :runtime
