# Nix Setup

Paracosm uses Nix for building docker images and for CI dependencies. We heavily use Nix flakes which requires a little more setup. The easiest way to use Nix on macOS or Linux is to use the installer located here: https://zero-to-nix.com/start/install

If you want to use the official Nix installer, you will need to follow the instructions located here: https://nixos.wiki/wiki/Flakes

# Flake Errors
If you receive an error that looks like `cannot fetch input 'path:../../.?lastModified=0&narHash=sha256-GwiMX0tMqRYHeABWRWUIB6%2BLAA2yYtQqF8l1C5QkLTo%3D' because it uses a relative path` you need to run:

```
nix flake lock --update-input paracosm
```
