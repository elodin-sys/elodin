# web_asset

HTTP/HTTPS asset source integration for Bevy with ETag-aware caching.

## What it does
- Registers `http://` and `https://` Bevy asset sources.
- Downloads remote assets through `reqwest`.
- Uses `asset_cache` to store/reuse bytes and ETags.
- Returns cached data on HTTP `304 Not Modified`.

## Main API
- `WebAssetPlugin`

## Status
Active.
