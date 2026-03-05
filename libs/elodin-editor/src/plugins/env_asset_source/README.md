# env_asset_source

Configures Bevy's default asset source from environment.

## What it does
- Reads `ELODIN_ASSETS_DIR`.
- Falls back to `assets` when the variable is not set.
- Resolves relative paths from current working directory.
- Registers the resulting path as default `AssetSource`.
- Logs warnings for invalid/missing/non-directory paths.

## Main API
- `plugin(app: &mut App)`

## Status
Active.
