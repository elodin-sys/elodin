# Rust Client Changelog

## v0.1.1 - Runtime Fix
- Fixed panic caused by tokio/stellarator runtime incompatibility
- Changed from `tokio::time::sleep` to `stellarator::sleep`
- Removed unnecessary tokio dependency

## v0.1.0 - Initial Implementation
- Dynamic component discovery from database
- Schema and metadata retrieval
- Automatic categorization of rocket components
- Framework for packet processing
- Full integration with Elodin-DB ecosystem
