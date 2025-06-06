#! /usr/bin/env nix
#! nix develop .#ops --command just --justfile

k8s_overlays := "kubernetes/overlays"
artifact_registry := "elodin.azurecr.io/elodin-infra"
repo_docs := "elo-docs/x86_64"

project := "dev"
cluster := "dev"

[private]
default:
  @just --list

re-tag-images old_tag new_tag:
  @ echo "   📌 Adding '{{new_tag}}' tag to images with '{{old_tag}}' tag"
  gcloud artifacts docker tags add {{artifact_registry}}/{{repo_docs}}:{{old_tag}} {{artifact_registry}}/{{repo_docs}}:{{new_tag}}

re-tag-images-main new_tag:
  just re-tag-images $(git rev-parse main) {{new_tag}}

re-tag-images-current new_tag:
  just re-tag-images $(git rev-parse HEAD) {{new_tag}}

clean-dev-branch branch_codename:
  az login --identity
  az aks get-credentials --resource-group {{project}} --name {{cluster}} --overwrite-existing
  kubectl get namespace elodin-app-{{branch_codename}} &> /dev/null && kubectl delete ns elodin-app-{{branch_codename}} || echo "elodin-app-{{branch_codename}} already deleted"

sync-assets:
  gsutil rsync -r assets gs://elodin-assets

sync-open-source:
  git filter-repo --refs main --path examples --path libs/roci --path libs/conduit --path libs/impeller2 --path libs/nox --path libs/nox-ecs --path libs/nox-ecs-macros --path libs/nox-py --path libs/xla-rs --path libs/noxla --path libs/s10 --path libs/stellarator --path fsw --path libs/db --path images --path rust-toolchain.toml --prune-empty always --target ../elodin
  (cd ../elodin && \
   git checkout -b old-main origin/main && \
   git rebase main --committer-date-is-author-date && \
   git checkout main && \
   git reset --hard old-main && \
   git branch -D old-main)

[confirm("Are you sure you want to force push to elodin-sys/elodin?")]
force-push-open-source: sync-open-source
  cd ../elodin; git push --force

version:
  @echo "v$(cargo metadata --format-version 1 | jq -r '.packages[] | select(.name == "elodin") | .version')"

auto-tag:
  #!/usr/bin/env sh
  current_tag=$(git describe --tags --abbrev=0)
  new_tag=$(just version)
  if [ "$current_tag" = "$new_tag" ]; then
    echo "Latest tag is already '$new_tag'"; exit 0
  fi
  echo "🏷️ Tagging HEAD with '$new_tag'"
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
  - name: elodin-infra/elo-docs
    newName: elodin.azurecr.io/elodin-infra/elo-docs/x86_64
    newTag: {{tag}}
  EOF
  kubectl kustomize kubernetes/deploy | kubectl --cluster gke_elodin-prod_us-central1_elodin-prod-gke apply -f -

promote tag:
  #!/usr/bin/env sh
  dir=$(mktemp -d)
  gh release download {{tag}} --pattern 'elodin-*' --dir $dir
  gsutil -m cp -r "$dir/*" "gs://elodin-releases/{{tag}}/"
  gsutil -m cp -r "gs://elodin-releases/{{tag}}/*" "gs://elodin-releases/latest/"
  uv publish "$dir/*.whl"

public-changelog:
  #!/usr/bin/env sh
  cd {{justfile_directory()}}
  ./scripts/public-changelog.sh CHANGELOG.md > docs/public/content/releases/changelog.md
  old_version=$(cat ./docs/public/config.toml | yq -p toml '.extra.version')
  new_version=$(just version)
  sed -i "" "s/$old_version/$new_version/g" docs/public/config.toml

install:
  @echo "🚧 Installing elodin and elodin-db to ~/.local/bin"
  cargo build --release --package elodin --package elodin-db
  cp target/release/elodin target/release/elodin-db ~/.local/bin
