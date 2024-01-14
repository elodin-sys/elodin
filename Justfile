k8s_overlays := "kubernetes/overlays"
artifact_registry := "us-central1-docker.pkg.dev/elodin-infra"
repo_sim_runner := "elo-sim-runner/x86_64"
repo_atc := "elo-atc/x86_64"
repo_dashboard := "elo-dashboard/x86_64"

decrypt-secrets *FLAGS:
  @ echo "   🔑 Decrypting secrets for kubernetes..."
  op inject {{FLAGS}} -i {{k8s_overlays}}/dev/enc.elo-dashboard-secret.env -o {{k8s_overlays}}/dev/elo-dashboard-secret.env
  op inject {{FLAGS}} -i {{k8s_overlays}}/dev-branch/enc.elo-dashboard-secret.env -o {{k8s_overlays}}/dev-branch/elo-dashboard-secret.env
  op inject {{FLAGS}} -i {{k8s_overlays}}/prod/enc.elo-dashboard-secret.env -o {{k8s_overlays}}/prod/elo-dashboard-secret.env

decrypt-secrets-force:
  just decrypt-secrets --force

re-tag-images old_tag new_tag:
  @ echo "   📌 Adding '{{new_tag}}' tag to images with '{{old_tag}}' tag"
  gcloud artifacts docker tags add {{artifact_registry}}/{{repo_sim_runner}}:{{old_tag}} {{artifact_registry}}/{{repo_sim_runner}}:{{new_tag}}
  gcloud artifacts docker tags add {{artifact_registry}}/{{repo_atc}}:{{old_tag}} {{artifact_registry}}/{{repo_atc}}:{{new_tag}}
  gcloud artifacts docker tags add {{artifact_registry}}/{{repo_dashboard}}:{{old_tag}} {{artifact_registry}}/{{repo_dashboard}}:{{new_tag}}

re-tag-images-main new_tag:
  just re-tag-images $(git rev-parse main) {{new_tag}}

re-tag-images-current new_tag:
  just re-tag-images $(git rev-parse HEAD) {{new_tag}}
