use nalgebra::{Quaternion, Vector3};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Core telemetry state: position, orientation, velocity in world frame
#[derive(Debug, Clone)]
pub struct TelemetryState {
    /// Position in world frame (x, y, z) in meters
    pub position: Vector3<f64>,

    /// Orientation as quaternion in nalgebra format (w, i, j, k)
    /// Note: Elodin stores quaternions as [x, y, z, w], converted on input
    pub orientation: Quaternion<f64>,

    /// Velocity in world frame (x, y, z) in m/s
    pub velocity: Vector3<f64>,

    /// System status
    pub system_status: SystemStatus,

    /// Database connection status
    pub db_connected: bool,

    /// Number of telemetry updates received
    pub update_count: u64,

    /// Timestamp of last update
    pub last_update: std::time::Instant,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SystemStatus {
    Initializing,
    Ready,
}

impl Default for TelemetryState {
    fn default() -> Self {
        Self {
            position: Vector3::zeros(),
            orientation: Quaternion::identity(),
            velocity: Vector3::zeros(),
            system_status: SystemStatus::Initializing,
            db_connected: false,
            update_count: 0,
            last_update: std::time::Instant::now(),
        }
    }
}

impl TelemetryState {
    /// Get altitude (z position) in meters
    pub fn altitude_m(&self) -> f64 {
        self.position.z
    }

    /// Get ground speed (horizontal velocity magnitude) in m/s
    pub fn ground_speed_ms(&self) -> f64 {
        (self.velocity.x.powi(2) + self.velocity.y.powi(2)).sqrt()
    }

    /// Get climb rate (vertical velocity) in m/s
    pub fn climb_rate_ms(&self) -> f64 {
        self.velocity.z
    }

    /// Get raw heading in degrees (0-360), derived from orientation
    /// Uses 3-2-1 Euler sequence (yaw-pitch-roll)
    ///
    /// Note: This returns heading in mathematical/ENU convention where:
    /// - 0° = East (positive X axis)
    /// - 90° = North (positive Y axis)
    ///
    /// For aviation display, use `CoordinateFrame::to_aviation_heading()` to convert.
    pub fn heading_deg(&self) -> f64 {
        let (_, _, yaw) = self.quat_to_euler_321();
        // Convert yaw to heading (0-360 degrees, ENU: 0=East, 90=North)
        let heading = yaw.to_degrees();
        if heading < 0.0 {
            heading + 360.0
        } else if heading >= 360.0 {
            heading - 360.0
        } else {
            heading
        }
    }

    /// Get roll angle in degrees
    /// Uses 3-2-1 Euler sequence (yaw-pitch-roll)
    pub fn roll_deg(&self) -> f64 {
        let (roll, _, _) = self.quat_to_euler_321();
        let mut roll_deg = roll.to_degrees();

        // Normalize to [-180, 180] range
        while roll_deg > 180.0 {
            roll_deg -= 360.0;
        }
        while roll_deg < -180.0 {
            roll_deg += 360.0;
        }

        roll_deg
    }

    /// Get pitch angle in degrees
    /// Uses 3-2-1 Euler sequence matching rc jet simulation
    pub fn pitch_deg(&self) -> f64 {
        let (_, pitch, _) = self.quat_to_euler_321();
        pitch.to_degrees()
    }

    /// Convert quaternion to Euler angles (roll, pitch, yaw) in 3-2-1 sequence
    /// Returns angles in radians: (roll, pitch, yaw)
    /// - roll: rotation about x-axis (body roll)
    /// - pitch: rotation about y-axis (nose up/down)
    /// - yaw: rotation about z-axis (heading)
    ///   See: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    fn quat_to_euler_321(&self) -> (f64, f64, f64) {
        let q = &self.orientation;
        // nalgebra quaternion: q.w is scalar, q.i/j/k are x/y/z components
        let x = q.i;
        let y = q.j;
        let z = q.k;
        let w = q.w;

        // Roll (rotation about x-axis)
        let sinr_cosp = 2.0 * (w * x + y * z);
        let cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
        let roll = sinr_cosp.atan2(cosr_cosp);

        // Pitch (rotation about y-axis)
        let sinp = (1.0 + 2.0 * (w * y - x * z)).sqrt();
        let cosp = (1.0 - 2.0 * (w * y - x * z)).sqrt();
        let pitch = 2.0 * sinp.atan2(cosp) - std::f64::consts::PI / 2.0;

        // Yaw (rotation about z-axis)
        let siny_cosp = 2.0 * (w * z + x * y);
        let cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
        let yaw = siny_cosp.atan2(cosy_cosp);

        (roll, pitch, yaw)
    }
}

/// Thread-safe telemetry processor
pub struct TelemetryProcessor {
    state: Arc<RwLock<TelemetryState>>,
}

impl TelemetryProcessor {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(TelemetryState::default())),
        }
    }

    pub async fn get_state(&self) -> TelemetryState {
        self.state.read().await.clone()
    }

    /// Update position (x, y, z) in world frame
    pub async fn update_position(&self, x: f64, y: f64, z: f64) {
        let mut state = self.state.write().await;
        state.position = Vector3::new(x, y, z);
        state.last_update = std::time::Instant::now();
    }

    /// Update orientation quaternion from Elodin database
    /// Elodin stores quaternions as [x, y, z, w] (scalar w is last)
    /// nalgebra::Quaternion::new expects (w, i, j, k) order
    pub async fn update_orientation(&self, qx: f64, qy: f64, qz: f64, qw: f64) {
        let mut state = self.state.write().await;
        // Reorder: Elodin gives (x, y, z, w), nalgebra expects (w, x, y, z)
        state.orientation = Quaternion::new(qw, qx, qy, qz);
        state.last_update = std::time::Instant::now();
    }

    /// Update velocity (x, y, z) in world frame
    pub async fn update_velocity(&self, vx: f64, vy: f64, vz: f64) {
        let mut state = self.state.write().await;
        state.velocity = Vector3::new(vx, vy, vz);
        state.last_update = std::time::Instant::now();
    }

    /// Update system status
    pub async fn update_status(&self, status: SystemStatus) {
        let mut state = self.state.write().await;
        state.system_status = status;
    }

    /// Set database connection status
    pub async fn set_db_connected(&self, connected: bool) {
        let mut state = self.state.write().await;
        state.db_connected = connected;
        if connected {
            tracing::info!("Database connected");
        } else {
            tracing::warn!("Database disconnected");
        }
    }

    /// Increment telemetry update counter
    pub async fn increment_update_count(&self) {
        let mut state = self.state.write().await;
        state.update_count += 1;
    }
}
