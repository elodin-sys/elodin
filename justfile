#! /usr/bin/env nix
#! nix develop --command just --justfile

[private]
default:
  @just --list

version:
  @echo "v$(cargo metadata --format-version 1 | jq -r '.packages[] | select(.name == "elodin") | .version')"

tag new_tag:
  #!/usr/bin/env sh
  current_tag=$(git describe --tags --abbrev=0)
  new_tag="{{new_tag}}"
  if [ "$current_tag" = "$new_tag" ]; then
    echo "error: Latest tag is already '$new_tag'" >&2; exit 1
  fi
  echo "🏷️ Tagging HEAD with '$new_tag'"
  sh -x <<EOF
  git tag -a $new_tag -m "Elodin $new_tag"
  git push origin $new_tag
  EOF

promote tag:
  #!/usr/bin/env sh -x
  if [ -z "$UV_PUBLISH_TOKEN" ]; then
     echo "error: Set UV_PUBLISH_TOKEN to the token." >&2;
     exit 1;
  fi
  dir=$(mktemp -d)
  gh release download {{tag}} --pattern 'elodin-*.whl' --dir $dir
  uv publish "$dir/*.whl" --token "$UV_PUBLISH_TOKEN"

public-changelog:
  #!/usr/bin/env sh
  cd {{justfile_directory()}}
  ./scripts/public-changelog.sh CHANGELOG.md > docs/public/content/releases/changelog.md
  old_version=$(cat ./docs/public/config.toml | yq -p toml '.extra.version')
  new_version=$(just version)
  sed -i "" "s/$old_version/$new_version/g" docs/public/config.toml

install:
  @echo "🚧 Installing elodin and elodin-db to ~/.nix-profile/bin"
  cargo build --release --package elodin --package elodin-db
  mkdir -p ~/.nix-profile 2>/dev/null || true
  cd ~/.nix-profile && mkdir -p bin
  cp target/release/elodin target/release/elodin-db ~/.nix-profile/bin
