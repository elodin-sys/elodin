# Paracosm Dashboard

To start your Phoenix server:

  * Run `mix setup` to install and setup dependencies
  * Start Phoenix endpoint with `mix phx.server` or inside IEx with `iex -S mix phx.server`

Now you can visit [`localhost:4000`](http://localhost:4000) from your browser.

# Docker Build
To build a docker image you will need Nix installed, and be on a Linux machine of the correct architecture (since Elixir does not support cross-compiling). I.e if you want to build for x86_64 you will need to be on an x86_64 machine. Then you can run

```
nix build .#docker
cat result | docker load
```
