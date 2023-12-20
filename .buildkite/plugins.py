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

def gcp_identity():
  return {
    "audience": "//iam.googleapis.com/projects/204492191803/locations/global/workloadIdentityPools/buildkite-pipeline/providers/buildkite",
    "service_account": "buildkite-204492191803@elodin-infra.iam.gserviceaccount.com",
    "cmds": {
      "regenerate_token": "buildkite-agent oidc request-token --audience \"\$GCP_WIF_AUDIENCE\" > \$BUILDKITE_OIDC_TMPDIR/token.json",
      "login": "gcloud --quiet auth login --cred-file=\$GOOGLE_APPLICATION_CREDENTIALS",
    },
  }

def gcp_identity_plugin():
  return {
    "gcp-workload-identity-federation#v1.0.0": {
      "audience": gcp_identity()["audience"],
      "service-account": gcp_identity()["service_account"],
    }
  }
