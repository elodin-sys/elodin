#!/bin/sh -e

# Detect the OS
OS=$(uname -s)
# Detect the architecture
ARCH=$(uname -m)

# Detect the shell, default to bash if not available
SHELL=$(basename "${SHELL:-/bin/bash}")

# Check if $HOME/.local/bin is in the PATH. If not, add it to the PATH.
add_local_bin_to_path() {
  shell=$1
  case ":$PATH:" in
    *":$HOME/.local/bin:"*)
      echo "\$HOME/.local/bin is already in your PATH"
      ;;
    *)
      case "$shell" in
        bash)
          echo "Adding \$HOME/.local/bin to your PATH in $HOME/.bashrc"
          echo "export PATH=\$HOME/.local/bin:\$PATH" >> "$HOME/.bashrc"
          echo "Please run 'source \$HOME/.bashrc' or start a new terminal to update your PATH"
          ;;
        zsh)
          echo "Adding \$HOME/.local/bin to your PATH in $HOME/.zshrc"
          echo "export PATH=\$HOME/.local/bin:\$PATH" >> "$HOME/.zshrc"
          echo "Please run 'source \$HOME/.zshrc' or start a new terminal to update your PATH"
          ;;
        fish)
          echo "Adding \$HOME/.local/bin to your PATH in $HOME/.config/fish/config.fish"
          echo "fish_add_path \$HOME/.local/bin" >> "$HOME/.config/fish/config.fish"
          echo "Please run 'source \$HOME/.config/fish/config.fish' or start a new terminal to update your PATH"
          ;;
        *)
          echo "Unsupported shell: $shell"
          echo "Please add \$HOME/.local/bin to your PATH manually and restart your terminal"
          ;;
      esac
      ;;
  esac
}

install_db() {
  os=$1
  arch=$2
  version=${3:-latest}

  if [ "$os" = "Darwin" ] && [ "$arch" = "arm64" ]; then
    download_url="https://storage.googleapis.com/elodin-releases/$version/elodin-db-aarch64-apple-darwin.tar.gz"
  elif [ "$os" = "Linux" ] && [ "$arch" = "aarch64" ]; then
    download_url="https://storage.googleapis.com/elodin-releases/$version/elodin-db-aarch64-unknown-linux-musl.tar.gz"
  elif [ "$os" = "Linux" ] && [ "$arch" = "x86_64" ]; then
    download_url="https://storage.googleapis.com/elodin-releases/$version/elodin-db-x86_64-unknown-linux-musl.tar.gz"
  else
    echo "Unsupported (OS, arch): ($os, $arch)"
    exit 1
  fi

  echo "Downloading $download_url"
  curl -L -# "$download_url" | tar xz --strip-components=1 -C "$HOME/.local/bin"
  version=$("$HOME/.local/bin/elodin-db" --version)
  echo "Installed $version to \$HOME/.local/bin"
}

mkdir -p "$HOME/.local/bin"
install_db "$OS" "$ARCH"
add_local_bin_to_path "$SHELL"
