#! /usr/bin/env nix
#! nix develop --command just --justfile

[private]
default:
  @just --list

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

promote tag:
  #!/usr/bin/env sh
  dir=$(mktemp -d)
  gh release download {{tag}} --pattern 'elodin-*.whl' --dir $dir
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
  nix develop --command cargo build --release --package elodin --package elodin-db
  cp target/release/elodin target/release/elodin-db ~/.local/bin
