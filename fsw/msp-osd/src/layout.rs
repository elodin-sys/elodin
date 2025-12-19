use crate::config::{CoordinateFrame, OsdConfig};
use crate::osd_grid::OsdGrid;
use crate::telemetry::{SystemStatus, TelemetryState};

/// Main render function that lays out all OSD elements
/// Uses only position, orientation, and velocity inputs
pub fn render(grid: &mut OsdGrid, state: &TelemetryState, osd_config: &OsdConfig) {
    grid.clear();

    let coordinate_frame = osd_config.coordinate_frame;

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
    render_horizon(grid, roll_deg, pitch_deg, osd_config);

    // Target indicator (if target is being tracked)
    render_target(grid, state, coordinate_frame);

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
fn render_horizon(grid: &mut OsdGrid, roll_deg: f32, pitch_deg: f32, osd_config: &OsdConfig) {
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
    // pitch_scale is degrees per row (lower = more sensitive)
    // Max offset is half the horizon height to keep horizon visible
    let max_pitch_offset = (horizon_height as f32) / 2.0;
    let pitch_offset =
        (-pitch_deg / osd_config.pitch_scale).clamp(-max_pitch_offset, max_pitch_offset);

    // Calculate roll angle based on coordinate frame:
    // - ENU (Elodin): positive roll = left wing up, needs negation for OSD
    // - NED (aviation): positive roll = right wing down, already correct for OSD
    let roll_rad = match osd_config.coordinate_frame {
        CoordinateFrame::Enu => (-roll_deg).to_radians(),
        CoordinateFrame::Ned => roll_deg.to_radians(),
    };

    // Use sin/cos instead of tan() to handle all roll angles smoothly,
    // including 90° (vertical) and inverted (180°) orientations.
    // tan() has asymptotes at ±90° causing horizon to "flip" at those angles.
    let (roll_sin, roll_cos) = roll_rad.sin_cos();

    // Character aspect ratio compensation for HD OSD systems (e.g., Walksnail Avatar).
    // OSD characters are taller than wide (~12x18 pixels = 1.5:1 aspect ratio).
    // Without this correction, the horizon line appears more steeply tilted than
    // the actual roll angle because each row step is visually larger than each column step.
    let char_aspect_ratio = osd_config.char_aspect_ratio;

    // Draw sky and ground regions using signed distance from tilted horizon line
    for row in start_row..=end_row {
        for col in start_col..=end_col {
            // Signed distance from the tilted horizon line through center point
            // d > 0: above horizon (sky), d < 0: below horizon (ground)
            // The row coordinate is scaled by char_aspect_ratio so the visual angle
            // on screen matches the actual roll angle.
            let rel_col = (col as i8 - center_col as i8) as f32;
            let rel_row = row as f32 - (center_row as f32 + pitch_offset);
            let d = roll_sin * rel_col - roll_cos * char_aspect_ratio * rel_row;

            let ch = if d.abs() > 0.5 {
                ' ' // Ground/Sky (keep empty to minimize camera occlusion)
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
    // Use the same pitch_scale as the horizon for consistency
    let pitch_range = 50; // Show marks from -50 to +50 degrees relative to current pitch

    for pitch_mark in (-pitch_range..=pitch_range).step_by(10) {
        if pitch_mark == 0 {
            continue; // Skip zero, the horizon line shows that
        }

        // Calculate row position: offset from center based on pitch difference
        // Positive pitch marks appear ABOVE center (lower row numbers)
        // Sign matches the horizon movement direction
        let pitch_diff = pitch_mark as f32 - pitch_deg;
        let row_offset = -pitch_diff / osd_config.pitch_scale;
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

/// Render target indicator on the OSD
/// Projects the target's 3D position in body frame to 2D screen coordinates.
/// Shows 'O' with distance when on-screen, directional arrows when off-screen.
fn render_target(grid: &mut OsdGrid, state: &TelemetryState, coordinate_frame: CoordinateFrame) {
    // Get target position in body frame
    let Some(rel_body) = state.target_relative_body_frame() else {
        return; // No target configured or no data
    };

    let Some(distance_m) = state.target_distance_m() else {
        return;
    };

    // Body frame convention (Elodin):
    // X = forward, Y = left, Z = up
    let forward = rel_body.x as f32; // positive = ahead
    let left = rel_body.y as f32; // positive = left of aircraft
    let up = rel_body.z as f32; // positive = above aircraft

    // Apply coordinate frame adjustment for up/down
    let up = match coordinate_frame {
        CoordinateFrame::Enu => up,
        CoordinateFrame::Ned => -up, // NED has Z down
    };

    // Screen parameters
    let center_row = grid.rows / 2;
    let center_col = grid.cols / 2;

    // Define the target display area (avoid compass and status bar)
    let min_row: u8 = 3;
    let max_row = grid.rows.saturating_sub(2);
    let min_col: u8 = 1;
    let max_col = grid.cols.saturating_sub(2);

    // FOV-based projection parameters
    // Assume ~90 degree horizontal FOV, adjusted for character aspect ratio
    // Characters are typically ~2x taller than wide, so we scale accordingly
    let h_fov_scale = (grid.cols as f32) / 2.0; // half-width at unit distance
    let v_fov_scale = (grid.rows as f32) / 2.0 * 1.5; // account for char aspect ratio

    // Format distance string
    let distance_text = if distance_m < 1000.0 {
        format!("{}m", distance_m as i32)
    } else {
        format!("{:.1}km", distance_m / 1000.0)
    };

    // Check if target is in front of aircraft
    if forward > 0.1 {
        // Target is ahead - use perspective projection
        // Project: screen position = (offset / forward_distance) * fov_scale
        // Negate left for screen X (screen right = positive)
        let screen_x = (-left / forward) * h_fov_scale + center_col as f32;
        // Negate up for screen Y (screen down = positive row)
        let screen_y = (-up / forward) * v_fov_scale + center_row as f32;

        let col = screen_x.round() as i32;
        let row = screen_y.round() as i32;

        // Check if on-screen (within display bounds)
        if row >= min_row as i32
            && row <= max_row as i32
            && col >= min_col as i32
            && col <= max_col as i32
        {
            // On-screen: draw target marker 'O' with distance
            let row = row as u8;
            let col = col as u8;
            grid.set_char(row, col, 'O');

            // Draw distance label (try below, then above if no room)
            let label_col = col.saturating_sub(distance_text.len() as u8 / 2);
            if row < max_row {
                grid.write_text(row + 1, label_col, &distance_text);
            } else if row > min_row {
                grid.write_text(row - 1, label_col, &distance_text);
            }
        } else {
            // Off-screen but in front: draw perimeter indicator
            draw_offscreen_indicator(
                grid,
                screen_x,
                screen_y,
                &distance_text,
                min_row,
                max_row,
                min_col,
                max_col,
                false, // not behind
            );
        }
    } else {
        // Target is behind aircraft
        // Calculate direction for perimeter indicator
        // Use the lateral (left/right) and vertical (up/down) components
        let angle = up.atan2(-left); // angle in screen space

        // Place at the back edge, but show direction
        // Use addition for screen_x to match the "ahead" case convention:
        // -left positive (target to right) → angle.cos() positive → screen_x > center (right side)
        let screen_x = center_col as f32 + angle.cos() * (grid.cols as f32 / 2.0 - 2.0);
        let screen_y = center_row as f32 - angle.sin() * (grid.rows as f32 / 2.0 - 2.0);

        draw_offscreen_indicator(
            grid,
            screen_x,
            screen_y,
            &distance_text,
            min_row,
            max_row,
            min_col,
            max_col,
            true, // behind
        );
    }
}

/// Draw an off-screen indicator at the perimeter pointing toward the target
#[allow(clippy::too_many_arguments)]
fn draw_offscreen_indicator(
    grid: &mut OsdGrid,
    screen_x: f32,
    screen_y: f32,
    distance_text: &str,
    min_row: u8,
    max_row: u8,
    min_col: u8,
    max_col: u8,
    is_behind: bool,
) {
    let center_row = grid.rows as f32 / 2.0;
    let center_col = grid.cols as f32 / 2.0;

    // Calculate angle from center to target direction
    let dx = screen_x - center_col;
    let dy = screen_y - center_row;
    let angle = dy.atan2(dx);

    // Find intersection with screen boundary
    // Clamp the position to the perimeter
    let (perimeter_col, perimeter_row) = clamp_to_perimeter(
        screen_x, screen_y, min_row, max_row, min_col, max_col, center_col, center_row,
    );

    let col = perimeter_col as u8;
    let row = perimeter_row as u8;

    // Choose arrow character based on direction
    // Using angle to determine which arrow to show
    let arrow = if is_behind {
        // Target is behind - use 'X' to indicate rear
        'X'
    } else {
        // Determine arrow based on angle (in radians)
        // 0 = right, π/2 = down, π = left, -π/2 = up
        let angle_deg = angle.to_degrees();
        match angle_deg {
            a if (-22.5..22.5).contains(&a) => '>',     // right
            a if (22.5..67.5).contains(&a) => '\\',     // down-right
            a if (67.5..112.5).contains(&a) => 'v',     // down
            a if (112.5..157.5).contains(&a) => '/',    // down-left
            a if !(-157.5..157.5).contains(&a) => '<',  // left
            a if (-157.5..-112.5).contains(&a) => '\\', // up-left
            a if (-112.5..-67.5).contains(&a) => '^',   // up
            a if (-67.5..-22.5).contains(&a) => '/',    // up-right
            _ => '*',
        }
    };

    // Draw the arrow
    grid.set_char(row, col, arrow);

    // Draw distance near the arrow
    // Position label adjacent to arrow, toward the center
    let label_col = if col < grid.cols / 2 {
        col.saturating_add(2).min(max_col)
    } else {
        col.saturating_sub(distance_text.len() as u8 + 1)
            .max(min_col)
    };

    let label_row = if row < grid.rows / 2 {
        row.saturating_add(1).min(max_row)
    } else {
        row.saturating_sub(1).max(min_row)
    };

    grid.write_text(label_row, label_col, distance_text);
}

/// Clamp a point to the perimeter rectangle
#[allow(clippy::too_many_arguments)]
fn clamp_to_perimeter(
    x: f32,
    y: f32,
    min_row: u8,
    max_row: u8,
    min_col: u8,
    max_col: u8,
    center_x: f32,
    center_y: f32,
) -> (f32, f32) {
    let dx = x - center_x;
    let dy = y - center_y;

    if dx.abs() < 0.001 && dy.abs() < 0.001 {
        return (center_x, min_row as f32); // Default to top
    }

    // Calculate intersection with each edge
    let min_x = min_col as f32;
    let max_x = max_col as f32;
    let min_y = min_row as f32;
    let max_y = max_row as f32;

    let mut t_min = f32::MAX;

    // Check each edge
    // Right edge (x = max_x)
    if dx > 0.0 {
        let t = (max_x - center_x) / dx;
        let y_at_t = center_y + t * dy;
        if y_at_t >= min_y && y_at_t <= max_y && t < t_min {
            t_min = t;
        }
    }
    // Left edge (x = min_x)
    if dx < 0.0 {
        let t = (min_x - center_x) / dx;
        let y_at_t = center_y + t * dy;
        if y_at_t >= min_y && y_at_t <= max_y && t < t_min {
            t_min = t;
        }
    }
    // Bottom edge (y = max_y)
    if dy > 0.0 {
        let t = (max_y - center_y) / dy;
        let x_at_t = center_x + t * dx;
        if x_at_t >= min_x && x_at_t <= max_x && t < t_min {
            t_min = t;
        }
    }
    // Top edge (y = min_y)
    if dy < 0.0 {
        let t = (min_y - center_y) / dy;
        let x_at_t = center_x + t * dx;
        if x_at_t >= min_x && x_at_t <= max_x && t < t_min {
            t_min = t;
        }
    }

    if t_min < f32::MAX {
        let final_x = (center_x + t_min * dx).clamp(min_x, max_x);
        let final_y = (center_y + t_min * dy).clamp(min_y, max_y);
        (final_x, final_y)
    } else {
        // Fallback: just clamp
        (x.clamp(min_x, max_x), y.clamp(min_y, max_y))
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
