use crate::config::OsdConfig;
use crate::osd_grid::OsdGrid;
use crate::telemetry::{SystemStatus, TelemetryState};

/// Main render function that lays out all OSD elements
pub fn render(grid: &mut OsdGrid, state: &TelemetryState, _config: &OsdConfig) {
    grid.clear();

    // Top bar - Compass
    render_compass(grid, state.heading_deg);

    // Left side - Altitude ladder
    render_altitude(grid, state.altitude_m, state.climb_rate_ms);

    // Right side - Speed indicator
    render_speed(grid, state.ground_speed_ms);

    // Center - Artificial horizon
    render_horizon(grid, state.roll_deg, state.pitch_deg);

    // Bottom bar - Status information
    render_status_bar(grid, state);

    // Top corners - System info
    render_system_info(grid, state);
}

/// Render compass at the top of the screen
fn render_compass(grid: &mut OsdGrid, heading: f32) {
    let center_col = grid.cols / 2;
    let compass_width = 40.min(grid.cols - 2);
    let start_col = center_col.saturating_sub(compass_width / 2);

    // Compass scale with tick marks
    let mut compass_line = String::new();
    
    // Calculate what part of the compass to show based on heading
    let start_heading = (heading - 20.0).rem_euclid(360.0) as i32;
    
    for i in 0..compass_width {
        let deg = (start_heading + i as i32 * 2) % 360;
        
        // Major tick marks every 30 degrees
        let ch = match deg {
            0 | 360 => 'N',
            90 => 'E',
            180 => 'S',
            270 => 'W',
            d if d % 30 == 0 => '|',
            d if d % 10 == 0 => '·',
            _ => '-',
        };
        compass_line.push(ch);
    }

    // Draw compass line
    grid.write_text(0, start_col, &compass_line);

    // Draw heading indicator (center arrow pointing up)
    grid.set_char(1, center_col, '▼');
    
    // Draw heading value
    let heading_text = format!("{:03.0}°", heading);
    grid.write_text(1, center_col.saturating_sub(2), &heading_text);
}

/// Render altitude on the left side
fn render_altitude(grid: &mut OsdGrid, altitude_m: f32, climb_rate: f32) {
    let start_row = 3;
    let end_row = grid.rows.saturating_sub(3);
    
    // Altitude value
    let alt_text = format!("ALT:{:5.0}m", altitude_m);
    grid.write_text(start_row, 0, &alt_text);
    
    // Climb rate indicator
    let climb_text = if climb_rate.abs() < 0.1 {
        format!("VS: ---")
    } else {
        format!("VS:{:+4.1}", climb_rate)
    };
    grid.write_text(start_row + 1, 0, &climb_text);
    
    // Visual climb rate indicator (ladder)
    let mid_row = (start_row + end_row) / 2;
    for row in (start_row + 3)..end_row {
        let ch = if row == mid_row {
            '═'
        } else if (row as i8 - mid_row as i8).abs() <= 3 {
            '─'
        } else {
            '·'
        };
        grid.set_char(row, 10, ch);
    }
    
    // Climb rate arrow
    if climb_rate.abs() > 0.5 {
        let arrow_offset = (climb_rate.clamp(-3.0, 3.0) * 2.0) as i8;
        let arrow_row = (mid_row as i8 - arrow_offset).clamp(start_row as i8 + 3, end_row as i8 - 1) as u8;
        let arrow = if climb_rate > 0.0 { '↑' } else { '↓' };
        grid.set_char(arrow_row, 11, arrow);
    }
}

/// Render speed on the right side
fn render_speed(grid: &mut OsdGrid, speed_ms: f32) {
    let start_row = 3;
    let col = grid.cols.saturating_sub(12);
    
    // Speed in knots (1 m/s ≈ 1.94384 knots)
    let speed_kts = speed_ms * 1.94384;
    
    let speed_text = format!("SPD:{:4.0}kt", speed_kts);
    grid.write_text(start_row, col, &speed_text);
    
    // Speed bar graph
    let bar_length = ((speed_kts / 10.0).min(10.0)) as usize;
    let bar = "▮".repeat(bar_length) + &"·".repeat(10_usize.saturating_sub(bar_length));
    grid.write_text(start_row + 1, col, &bar);
}

/// Render artificial horizon in the center
fn render_horizon(grid: &mut OsdGrid, roll_deg: f32, pitch_deg: f32) {
    let center_row = grid.rows / 2;
    let center_col = grid.cols / 2;
    let horizon_width = 30.min(grid.cols.saturating_sub(20));
    let horizon_height = 7.min(grid.rows.saturating_sub(10));
    
    let start_col = center_col.saturating_sub(horizon_width / 2);
    let start_row = center_row.saturating_sub(horizon_height / 2);
    let end_col = start_col + horizon_width;
    let end_row = start_row + horizon_height;
    
    // Calculate horizon line position based on pitch
    let pitch_offset = (pitch_deg / 10.0).clamp(-3.0, 3.0) as i8;
    let _horizon_row = (center_row as i8 + pitch_offset) as u8;
    
    // Calculate roll tilt for the horizon line
    let roll_rad = roll_deg.to_radians();
    let roll_slope = roll_rad.tan();
    
    // Draw sky and ground regions
    for row in start_row..=end_row {
        for col in start_col..=end_col {
            // Calculate if this point is above or below the tilted horizon
            let rel_col = (col as i8 - center_col as i8) as f32;
            let horizon_y = center_row as f32 + pitch_offset as f32 + (rel_col * roll_slope);
            
            let ch = if row as f32 > horizon_y {
                '▒' // Ground
            } else if (row as f32 - horizon_y).abs() < 0.5 {
                '═' // Horizon line
            } else {
                ' ' // Sky
            };
            
            grid.set_char(row, col, ch);
        }
    }
    
    // Draw aircraft reference symbol (stays centered)
    grid.set_char(center_row, center_col.saturating_sub(3), '◄');
    grid.set_char(center_row, center_col, '✈');
    grid.set_char(center_row, center_col + 3, '►');
    
    // Draw pitch ladder marks
    for i in [-20, -10, 10, 20] {
        let ladder_row = (center_row as i8 + (i / 10) - pitch_offset) as u8;
        if ladder_row >= start_row && ladder_row <= end_row {
            let mark = format!("{:+2}", i);
            grid.write_text(ladder_row, start_col.saturating_sub(3), &mark);
            grid.write_text(ladder_row, end_col + 1, &mark);
        }
    }
    
    // Display roll and pitch values
    let attitude_text = format!("R:{:+4.0}° P:{:+4.0}°", roll_deg, pitch_deg);
    grid.write_centered(end_row + 1, &attitude_text);
}

/// Render status bar at the bottom
fn render_status_bar(grid: &mut OsdGrid, state: &TelemetryState) {
    let bottom_row = grid.rows - 1;
    
    // GPS satellites
    let gps_text = format!("GPS:{:2}", state.gps_sats);
    grid.write_text(bottom_row, 0, &gps_text);
    
    // System status
    let status_text = match &state.system_status {
        SystemStatus::Initializing => "INIT",
        SystemStatus::Ready => "READY",
        SystemStatus::Armed => "ARMED",
        SystemStatus::Flying => "FLY",
        SystemStatus::Warning(msg) => msg.as_str(),
        SystemStatus::Error(msg) => msg.as_str(),
    };
    
    let status_display = format!("SYS:{}", status_text);
    grid.write_centered(bottom_row, &status_display);
    
    // Battery info (if available)
    if state.battery_voltage > 0.0 {
        let battery_text = format!("{:.1}V {:.1}A", state.battery_voltage, state.current_amps);
        grid.write_right_aligned(bottom_row, &battery_text);
    }
    
    // Time since last update (staleness indicator)
    let stale_secs = state.last_update.elapsed().as_secs();
    if stale_secs > 2 {
        let stale_text = format!("STALE:{}s", stale_secs);
        grid.write_text(bottom_row - 1, 0, &stale_text);
    }
}

/// Render system info in top corners
fn render_system_info(grid: &mut OsdGrid, state: &TelemetryState) {
    // Top-left: Angular rates
    let rates_text = format!(
        "ω:{:+3.0},{:+3.0},{:+3.0}",
        state.angular_vel.x.to_degrees(),
        state.angular_vel.y.to_degrees(),
        state.angular_vel.z.to_degrees()
    );
    grid.write_text(2, 0, &rates_text);
    
    // Top-right: Acceleration magnitude
    let accel_mag = state.linear_accel.magnitude();
    let g_force = accel_mag / 9.81;
    let accel_text = format!("G:{:.2}", g_force);
    grid.write_right_aligned(2, &accel_text);
}
