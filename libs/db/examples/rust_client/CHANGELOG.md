# Rust Client Changelog

## v0.1.6 - Smart Buffer Detection
- Detects buffer components (e.g., v_rel_accel_buffer) and displays "[buffer: N values]"
- Very large arrays (>20 values) are summarized as "[N values]" to avoid clutter
- Buffer detection based on component name containing "buffer"
- Maintains full value display for non-buffer arrays under 20 values

## v0.1.5 - Pure Raw Data Display
- Removed ALL synthetic data generation - now shows only real values
- Eliminated value transformations - displays raw telemetry as received
- Expanded array displays to show all values (no more "[7 values]" summaries)
- Cleaner display without color coding - focus on raw data
- Updated dashboard title to indicate "RAW VALUES"

## v0.1.4 - Real Data Extraction Fix
- Fixed telemetry values showing incorrect/zero data
- Added proper VTable registry to store incoming VTable definitions
- Implemented real data extraction using table.sink() with decomponentize
- Added fallback to synthetic data when VTables not yet available
- Removed hardcoded synthetic values that were overriding real data
- Now properly extracts actual telemetry values from binary packets

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
