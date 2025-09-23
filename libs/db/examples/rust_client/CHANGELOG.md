# Rust Client Changelog

## v0.4.1 - Command Line Enhancement & Documentation Improvements
- **Host:Port Notation Support**: Added support for combined host:port notation in -H flag
- Now accepts formats like `-H [::]:2240` to match elodin-db standard
- Maintains backwards compatibility with separate `-H` and `-p` flags
- Supports IPv4, IPv6, and hostname formats with optional port specification
- Default changed to "127.0.0.1:2240" to be consistent with the new format
- **Documentation**: Clarified control examples to show complete packet sending flow

## v0.4.0 - The Dirty Flag Bug Fix (MAJOR)
- **CRITICAL FIX**: External control components now marked as dirty when updated from DB
- Fixed GPU execution always using initial values instead of DB updates
- Components changed in `copy_db_to_world` now properly flagged for GPU sync
- External control finally works end-to-end!
- See THE-DIRTY-FLAG-BUG.md for complete analysis

## v0.3.2 - External Control Pipeline Fix (Attempt - Incomplete)
- Added explicit `read_external_trim` system to ensure component is read each tick
- Attempted to fix external control flow but missed the dirty flag issue
- Diagnostic systems added to trace the problem

## v0.3.1 - Rocket Physics Fix
- Fixed rocket physics to respond to trim control
- Added roll moment generation from fin_control_trim
- Roll moment was previously hardcoded to zero
- Trim now creates visible roll effect on rocket
- Added visualization graphs for trim, deflection, and aero coefficients
- Roll effectiveness tunable via parameter

## v0.3.0 - External Control Component Solution
- Properly fixed time travel errors with metadata-based external control mechanism
- Components marked with metadata={"external_control": "true"} are not written back
- Modified nox-ecs to check component metadata instead of hardcoding names
- Control client uses normal timestamps (no future timestamp hacks)
- Clean separation of control authority
- Simulation reads but doesn't write external control components
- Scalable solution - any component can be marked as external control
- Components can be initialized with default values for use before client connects
- Requires rebuilding nox-ecs and nox-py

## v0.2.2 - Future Timestamp Workaround (Deprecated)
- Attempted to resolve time travel errors using future timestamps
- Proved to be hacky and brittle
- Replaced by proper external control solution in v0.3.0

## v0.2.1 - Timestamp Synchronization Fix (Partial)
- Added proper timestamps to VTable definition and packets
- Fixed timestamp structure but conflict with simulation persisted
- Led to discovery of write-back issue in v0.2.2

## v0.2.0 - Bidirectional Control Support
- Added control module for sending commands to simulation
- Implemented dual-client architecture (telemetry + control)
- Sends sinusoidal trim commands to `rocket.fin_control_trim`
- Control parameters: ±2° amplitude at 0.25Hz (4-second period)
- 60Hz update rate for smooth control
- Concurrent operation with telemetry streaming
- Created test script for easy validation
- Demonstrates full read/write capability with elodin-db

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
