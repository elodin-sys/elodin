#! /usr/bin/env nix
#! nix develop --command just --justfile

# Use `DRY_RUN=1` to do a dry run of any command.
#
# ```sh
# $ DRY_RUN=1 just tag v0.1.2
# DRY_RUN enabled.
# 🏷️ Tagging HEAD with 'v0.1.2'
#   git tag -a v0.1.2 -m "Elodin v0.1.2"
#   git push origin v0.1.2
# ```

[private]
default:
  @just --list

version:
  @echo "v$(cargo metadata --format-version 1 | jq -r '.packages[] | select(.name == "elodin") | .version')"

tag new_tag:
  #!/usr/bin/env sh
  [ -n "$DRY_RUN" ] && echo "DRY_RUN enabled."
  current_tag=$(git describe --tags --abbrev=0)
  new_tag="{{new_tag}}"
  if [ "$current_tag" = "$new_tag" ]; then
    echo "error: Latest tag is already '$new_tag'" >&2; exit 1;
  fi
  current_branch=$(git branch --show-current);
  if [ "$current_branch" != "main" ]; then
    echo "error: Expected 'main' branch but was '$current_branch'." >&2; exit 2;
  fi
  echo "🏷️ Tagging HEAD with '$new_tag'"
  sh -v ${DRY_RUN:+-n} <<EOF
    git tag -a $new_tag -m "Elodin $new_tag"
    git push origin $new_tag
  EOF

promote tag:
  #!/usr/bin/env sh
  [ -n "$DRY_RUN" ] && echo "DRY_RUN enabled."
  if [ -z "$UV_PUBLISH_TOKEN" ]; then
    echo "error: Set UV_PUBLISH_TOKEN to the token." >&2;
    exit 1;
  fi
  sh -v ${DRY_RUN:+-n} <<EOF
    dir=$(mktemp -d)
    gh release download {{tag}} --pattern 'elodin-*.whl' --dir $dir
    uv publish "$dir/*.whl" --token "$UV_PUBLISH_TOKEN"
  EOF

public-changelog:
  #!/usr/bin/env sh
  [ -n "$DRY_RUN" ] && echo "DRY_RUN enabled."
  sh -v ${DRY_RUN:+-n} <<EOF
    cd {{justfile_directory()}}
    ./scripts/public-changelog.sh CHANGELOG.md > docs/public/content/releases/changelog.md
    old_version=$(cat ./docs/public/config.toml | yq -p toml '.extra.version')
    new_version=$(just version)
    sed -i "" "s/$old_version/$new_version/g" docs/public/config.toml
  EOF

install:
  #!/usr/bin/env sh
  [ -n "$DRY_RUN" ] && echo "DRY_RUN enabled."
  echo "🚧 Installing elodin and elodin-db to ~/.nix-profile/bin"
  sh -v ${DRY_RUN:+-n} <<EOF
    cargo build --release --package elodin --package elodin-db
    mkdir -p ~/.nix-profile/bin 2>/dev/null || true
    cp target/release/elodin target/release/elodin-db ~/.nix-profile/bin
  EOF
