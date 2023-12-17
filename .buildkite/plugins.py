def elixir_cache_plugin():
  return {
    "cache#v0.6.0": {
      "path": "./services/dashboard/_build",
      "restore": "all",
      "save": "all",
    }
  }


def rust_cache_plugin():
  return {
    "cache#v0.6.0": {
      "path": "./target",
      "restore": "branch",
      "save": "branch",
    }
  }
