use crate::config::CoordinateFrame;
use crate::osd_grid::OsdGrid;
use crate::telemetry::{SystemStatus, TelemetryState};

/// Main render function that lays out all OSD elements
/// Uses only position, orientation, and velocity inputs
pub fn render(grid: &mut OsdGrid, state: &TelemetryState, coordinate_frame: CoordinateFrame) {
    grid.clear();

    // Derive display values from core telemetry
    // Altitude and climb rate need coordinate frame conversion:
    // - ENU: Z is up, positive z = altitude, positive vz = climbing
    // - NED: Z is down, positive z = depth, positive vz = descending
    let altitude_m = coordinate_frame.to_display_altitude(state.altitude_m() as f32);
    let ground_speed_ms = state.ground_speed_ms() as f32;
    let climb_rate_ms = coordinate_frame.to_display_climb_rate(state.climb_rate_ms() as f32);
    let heading_raw = state.heading_deg() as f32;
    let roll_deg = state.roll_deg() as f32;
    let pitch_deg = state.pitch_deg() as f32;

    // Convert heading to aviation convention (0°=North, 90°=East)
    // based on the configured coordinate frame
    let heading_deg = coordinate_frame.to_aviation_heading(heading_raw);

    // Top bar - Compass (derived from orientation, in aviation convention)
    render_compass(grid, heading_deg);

    // Left side - Altitude ladder (from position.z)
    render_altitude(grid, altitude_m, climb_rate_ms);

    // Right side - Speed indicator (from velocity magnitude)
    render_speed(grid, ground_speed_ms);

    // Center - Artificial horizon (from orientation)
    render_horizon(grid, roll_deg, pitch_deg, coordinate_frame);

    // Bottom bar - Status
    render_status_bar(grid, state);
}

/// Render compass at the top of the screen
fn render_compass(grid: &mut OsdGrid, heading: f32) {
    let center_col = grid.cols / 2;
    let compass_width = 40.min(grid.cols - 2);
    let start_col = center_col.saturating_sub(compass_width / 2);

    // Compass scale with tick marks
    let mut compass_line = String::new();

    // Calculate what part of the compass to show based on heading
    // Round heading to nearest 2 degrees to prevent flickering
    let heading_rounded = (heading / 2.0).round() * 2.0;
    // Each column represents 2 degrees, so subtract half the total range to center
    let start_heading = (heading_rounded - (compass_width as f32)).rem_euclid(360.0) as i32;

    for i in 0..compass_width {
        let deg = (start_heading + i as i32 * 2) % 360;

        let ch = match deg {
            0 => 'N',
            90 => 'E',
            180 => 'S',
            270 => 'W',
            d if d % 30 == 0 => '|',
            d if d % 10 == 0 => '.',
            _ => '-',
        };
        compass_line.push(ch);
    }

    // Draw compass line
    grid.write_text(0, start_col, &compass_line);

    // Draw heading indicator (center arrow pointing down)
    grid.set_char(1, center_col, 'v');

    // Draw heading value
    let heading_text = format!("{:03.0}°", heading);
    grid.write_text(1, center_col.saturating_sub(2), &heading_text);
}

/// Render altitude on the left side
fn render_altitude(grid: &mut OsdGrid, altitude_m: f32, _climb_rate: f32) {
    let start_row = 3;

    // Altitude value
    let alt_text = format!("ALT:{:5.0}m", altitude_m);
    grid.write_text(start_row, 0, &alt_text);
}

/// Render speed on the right side
fn render_speed(grid: &mut OsdGrid, speed_ms: f32) {
    let start_row = 3;
    let col = grid.cols.saturating_sub(12);

    // Speed in meters per second
    let speed_text = format!("SPD:{:4.1}m/s", speed_ms);
    grid.write_text(start_row, col, &speed_text);
}

/// Render artificial horizon in the center
/// Uses FPV/OSD convention: horizon moves opposite to aircraft pitch
/// - Climbing (nose up): horizon moves DOWN, more sky visible
/// - Diving (nose down): horizon moves UP, more ground visible
fn render_horizon(
    grid: &mut OsdGrid,
    roll_deg: f32,
    pitch_deg: f32,
    coordinate_frame: CoordinateFrame,
) {
    let center_row = grid.rows / 2;
    let center_col = grid.cols / 2;
    let horizon_width = 30.min(grid.cols.saturating_sub(20));
    let horizon_height = 7.min(grid.rows.saturating_sub(10));

    let start_col = center_col.saturating_sub(horizon_width / 2);
    let start_row = center_row.saturating_sub(horizon_height / 2);
    let end_col = start_col + horizon_width;
    let end_row = start_row + horizon_height;

    // Calculate horizon line position based on pitch
    // Negative sign: pitch up → horizon moves down (higher row number) → more sky above
    let pitch_offset = (-pitch_deg / 10.0).clamp(-3.0, 3.0);

    // Calculate roll angle based on coordinate frame:
    // - ENU (Elodin): positive roll = left wing up, needs negation for OSD
    // - NED (aviation): positive roll = right wing down, already correct for OSD
    let roll_rad = match coordinate_frame {
        CoordinateFrame::Enu => (-roll_deg).to_radians(),
        CoordinateFrame::Ned => roll_deg.to_radians(),
    };

    // Use sin/cos instead of tan() to handle all roll angles smoothly,
    // including 90° (vertical) and inverted (180°) orientations.
    // tan() has asymptotes at ±90° causing horizon to "flip" at those angles.
    let (roll_sin, roll_cos) = roll_rad.sin_cos();

    // Draw sky and ground regions using signed distance from tilted horizon line
    for row in start_row..=end_row {
        for col in start_col..=end_col {
            // Signed distance from the tilted horizon line through center point
            // d > 0: above horizon (sky), d < 0: below horizon (ground)
            let rel_col = (col as i8 - center_col as i8) as f32;
            let rel_row = row as f32 - (center_row as f32 + pitch_offset);
            let d = roll_sin * rel_col - roll_cos * rel_row;

            let ch = if d < -0.5 {
                '.' // Ground (minimal dot pattern for least camera occlusion)
            } else if d > 0.5 {
                ' ' // Sky
            } else {
                '-' // Horizon line
            };

            grid.set_char(row, col, ch);
        }
    }

    // Draw aircraft reference symbol (stays centered)
    grid.set_char(center_row, center_col.saturating_sub(3), '<');
    grid.set_char(center_row, center_col, '+');
    grid.set_char(center_row, center_col + 3, '>');

    // Draw pitch ladder marks
    // Generate marks every 10 degrees, visible range depends on current pitch
    // Scale: approximately 2-3 rows per 10 degrees for better spacing
    let rows_per_10deg = 2.5;
    let pitch_range = 50; // Show marks from -50 to +50 degrees relative to current pitch

    for pitch_mark in (-pitch_range..=pitch_range).step_by(10) {
        if pitch_mark == 0 {
            continue; // Skip zero, the horizon line shows that
        }

        // Calculate row position: offset from center based on pitch difference
        // Positive pitch marks appear ABOVE center (lower row numbers)
        // Sign matches the horizon movement direction
        let pitch_diff = pitch_mark as f32 - pitch_deg;
        let row_offset = pitch_diff / 10.0 * rows_per_10deg;
        let ladder_row = center_row as f32 + row_offset;

        // Only draw if within visible area
        if ladder_row >= start_row as f32 && ladder_row <= end_row as f32 {
            let ladder_row_u8 = ladder_row.round() as u8;
            let mark = format!("{:+2}", pitch_mark);
            grid.write_text(ladder_row_u8, start_col.saturating_sub(3), &mark);
            grid.write_text(ladder_row_u8, end_col + 1, &mark);
        }
    }
}

/// Render status bar at the bottom
fn render_status_bar(grid: &mut OsdGrid, state: &TelemetryState) {
    let bottom_row = grid.rows - 1;

    // System status
    let status_text = match &state.system_status {
        SystemStatus::Initializing => "INIT",
        SystemStatus::Ready => "READY",
    };

    let status_display = format!("SYS:{}", status_text);
    grid.write_centered(bottom_row, &status_display);

    // Time since last update (staleness indicator)
    let stale_secs = state.last_update.elapsed().as_secs();
    if stale_secs > 2 {
        let stale_text = format!("STALE:{}s", stale_secs);
        grid.write_text(bottom_row - 1, 0, &stale_text);
    }
}
