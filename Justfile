k8s_overlays := "kubernetes/overlays"

decrypt-secrets *FLAGS:
  @ echo "   🔑 Decrypting secrets for kubernetes..."
  op inject {{FLAGS}} -i {{k8s_overlays}}/dev/enc.elo-dashboard-secret.env -o {{k8s_overlays}}/dev/elo-dashboard-secret.env
  op inject {{FLAGS}} -i {{k8s_overlays}}/dev-branch/enc.elo-dashboard-secret.env -o {{k8s_overlays}}/dev-branch/elo-dashboard-secret.env
  op inject {{FLAGS}} -i {{k8s_overlays}}/prod/enc.elo-dashboard-secret.env -o {{k8s_overlays}}/prod/elo-dashboard-secret.env

decrypt-secrets-force:
  just decrypt-secrets --force
