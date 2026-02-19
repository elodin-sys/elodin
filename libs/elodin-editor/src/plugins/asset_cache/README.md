# asset_cache

Utilities for HTTP asset caching used by `web_asset`.

## What it does
- Defines the `AssetCache` trait (`get` / `put`).
- Defines `CachedAsset` (`data` + `etag`).
- Provides `cache()` factory:
  - `FsCache` on native targets.
  - `NoCache` on `wasm`.

## Main API
- `AssetCache`
- `CachedAsset`
- `cache()`

## Status
Active (internal infrastructure).
