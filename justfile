#! /usr/bin/env nix
#! nix develop .#ops --command just --justfile

k8s_overlays := "kubernetes/overlays"
artifact_registry := "us-central1-docker.pkg.dev/elodin-infra"
repo_sim_agent := "elo-sim-agent/x86_64"
repo_atc := "elo-atc/x86_64"
repo_dashboard := "elo-dashboard/x86_64"

project := "elodin-dev"
region := "us-central1"
cluster := "elodin-dev-gke"

[private]
default:
  @just --list

decrypt-secrets *FLAGS:
  @ echo "   🔑 Decrypting secrets for kubernetes..."
  op inject {{FLAGS}} -i {{k8s_overlays}}/dev/enc.elo-dashboard-secret.env -o {{k8s_overlays}}/dev/elo-dashboard-secret.env
  op inject {{FLAGS}} -i {{k8s_overlays}}/dev-branch/enc.elo-dashboard-secret.env -o {{k8s_overlays}}/dev-branch/elo-dashboard-secret.env
  op inject {{FLAGS}} -i {{k8s_overlays}}/prod/enc.elo-dashboard-secret.env -o {{k8s_overlays}}/prod/elo-dashboard-secret.env
  op read op://Infrastructure/GCS/dev/sim-gcs-key.json | jq > {{k8s_overlays}}/dev-branch/sim-gcs-key.json

decrypt-secrets-force:
  just decrypt-secrets --force

re-tag-images old_tag new_tag:
  @ echo "   📌 Adding '{{new_tag}}' tag to images with '{{old_tag}}' tag"
  gcloud artifacts docker tags add {{artifact_registry}}/{{repo_sim_agent}}:{{old_tag}} {{artifact_registry}}/{{repo_sim_agent}}:{{new_tag}}
  gcloud artifacts docker tags add {{artifact_registry}}/{{repo_atc}}:{{old_tag}} {{artifact_registry}}/{{repo_atc}}:{{new_tag}}
  gcloud artifacts docker tags add {{artifact_registry}}/{{repo_dashboard}}:{{old_tag}} {{artifact_registry}}/{{repo_dashboard}}:{{new_tag}}

re-tag-images-main new_tag:
  just re-tag-images $(git rev-parse main) {{new_tag}}

re-tag-images-current new_tag:
  just re-tag-images $(git rev-parse HEAD) {{new_tag}}

clean-dev-branch branch_codename:
  gcloud container clusters get-credentials {{cluster}} --region {{region}} --project {{project}}
  kubectl get namespace elodin-app-{{branch_codename}} &> /dev/null && kubectl delete ns elodin-app-{{branch_codename}} || echo "elodin-app-{{branch_codename}} already deleted"
  kubectl get namespace elodin-vms-{{branch_codename}} &> /dev/null && kubectl delete ns elodin-vms-{{branch_codename}} || echo "elodin-vms-{{branch_codename}} already deleted"

sync-open-source:
  git filter-repo --refs main --path examples --path libs/conduit --path libs/nox --path libs/nox-ecs --path libs/nox-ecs-macros --path libs/nox-py --path libs/xla-rs --prune-empty always --target ../elodin
  cd ../elodin; git fetch origin main; git rebase origin/main --committer-date-is-author-date

[confirm("Are you sure you want to force push to elodin-sys/elodin?")]
force-push-open-source: sync-open-source
  cd ../elodin; git push --force

auto-tag:
  #!/usr/bin/env sh
  if git tag --points-at HEAD | grep -q .; then
    echo "HEAD already has a tag"; exit 0
  fi
  current_tag=$(git describe --tags --abbrev=0)
  new_tag=$(echo $current_tag | awk -F. '{$NF = $NF + 1;} 1' | sed 's/ /./g')
  git tag -a $new_tag -m "Elodin $new_tag"
  git push origin $new_tag
  just re-tag-images $(git rev-parse HEAD) $new_tag

[confirm("Are you sure you want to deploy to prod?")]
release tag:
  #!/usr/bin/env sh
  mkdir -p kubernetes/deploy
  cat << EOF > kubernetes/deploy/kustomization.yaml
  apiVersion: kustomize.config.k8s.io/v1beta1
  kind: Kustomization
  resources:
  - ../overlays/prod
  images:
  - name: elodin-infra/elo-atc
    newName: us-central1-docker.pkg.dev/elodin-infra/elo-atc/x86_64
    newTag: {{tag}}
  - name: elodin-infra/elo-dashboard
    newName: us-central1-docker.pkg.dev/elodin-infra/elo-dashboard/x86_64
    newTag: {{tag}}
  - name: elodin-infra/elo-sim-agent
    newName: us-central1-docker.pkg.dev/elodin-infra/elo-sim-agent/x86_64
    newTag: {{tag}}
  replacements:
  - source:
      kind: Deployment
      name: sim-agent-mc
      fieldPath: spec.template.spec.containers.[name=sim-agent-mc].image
    targets:
    - select:
        kind: Deployment
        name: elo-atc
      fieldPaths:
      - spec.template.spec.containers.[name=elo-atc].env.[name=ELODIN_ORCA.IMAGE_NAME].value
  EOF
  kubectl kustomize kubernetes/deploy | kubectl --cluster gke_elodin-prod_us-central1_elodin-prod-gke apply -f -

promote tag:
  #!/usr/bin/env sh
  dir=$(mktemp -d)
  gh release download {{tag}} --pattern 'elodin-*' --dir $dir
  gsutil -m cp -r "$dir/*" "gs://elodin-releases/{{tag}}/"
  gsutil -m cp -r "gs://elodin-releases/{{tag}}/*" "gs://elodin-releases/latest/"
  twine upload "$dir/*.whl"

public-changelog:
  #!/usr/bin/env sh
  cd {{justfile_directory()}}
  ./scripts/public-changelog.sh CHANGELOG.md > docs/public/updates/changelog.mdx
  old_version=$(jq -r '.versions[0]' docs/public/mint.json)
  new_version="v$(cargo metadata --format-version 1 | jq -r '.packages[] | select(.name == "elodin") | .version')"
  sed -i "" "s/$old_version/$new_version/g" docs/public/mint.json
