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

def gcp_identity_plugin():
  return {
    "gcp-workload-identity-federation#v1.0.0": {
      "audience": "//iam.googleapis.com/projects/802981626435/locations/global/workloadIdentityPools/buildkite-pipeline/providers/buildkite",
      "service-account": "buildkite-802981626435@elodin-dev.iam.gserviceaccount.com",
    }
  }
