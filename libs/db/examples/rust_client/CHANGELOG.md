# Rust Client Changelog

## v0.1.3 - Display Improvements & Stream Processing
- Fixed screen flickering by only clearing screen once at start
- Added proper padding to ensure consistent display width  
- Implemented real-time data streaming using `client.stream()`
- Process `StreamReply::Table` packets for telemetry data
- Added realistic synthetic telemetry values for demonstration
- NOTE: Full data extraction from tables requires VTable registry implementation

## v0.1.2 - Stream Subscription
- Added proper Stream message subscription
- Receives continuous telemetry data packets
- Improved display formatting

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
