use nalgebra::{Quaternion, Vector3};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Core telemetry state: position, orientation, velocity in world frame
#[derive(Debug, Clone)]
pub struct TelemetryState {
    /// Position in world frame (x, y, z) in meters
    pub position: Vector3<f64>,

    /// Orientation as quaternion (w, i, j, k)
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

    /// Get heading in degrees (0-360), derived from orientation
    /// Uses 3-2-1 Euler sequence matching rc jet simulation
    pub fn heading_deg(&self) -> f64 {
        // Note: roll and yaw are swapped due to coordinate system
        let (roll, _, _) = self.quat_to_euler_321();
        // Negate to get correct heading direction (left yaw = increasing heading)
        let heading = -roll.to_degrees();
        if heading < 0.0 {
            heading + 360.0
        } else if heading >= 360.0 {
            heading - 360.0
        } else {
            heading
        }
    }

    /// Get roll angle in degrees
    /// Uses 3-2-1 Euler sequence matching rc jet simulation
    pub fn roll_deg(&self) -> f64 {
        // Note: roll and yaw are swapped due to coordinate system
        let (_, _, yaw) = self.quat_to_euler_321();
        let mut roll = yaw.to_degrees();
        
        // Normalize to [-180, 180] range, wrapping properly around 0
        while roll > 180.0 {
            roll -= 360.0;
        }
        while roll < -180.0 {
            roll += 360.0;
        }
        
        roll
    }

    /// Get pitch angle in degrees
    /// Uses 3-2-1 Euler sequence matching rc jet simulation
    pub fn pitch_deg(&self) -> f64 {
        let (_, pitch, _) = self.quat_to_euler_321();
        pitch.to_degrees()
    }

    /// Convert quaternion to Euler angles (roll, pitch, yaw) in 3-2-1 sequence
    /// Matches the rc jet simulation's conversion from examples/rc-jet/util.py
    /// See: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_(in_3-2-1_sequence)_conversion
    fn quat_to_euler_321(&self) -> (f64, f64, f64) {
        let q = &self.orientation;
        let q0 = q.i; // x
        let q1 = q.j; // y
        let q2 = q.k; // z
        let s = q.w;  // w (scalar)

        // Roll (rotation about x-axis)
        let sinr_cosp = 2.0 * (s * q0 + q1 * q2);
        let cosr_cosp = 1.0 - 2.0 * (q0 * q0 + q1 * q1);
        let roll = sinr_cosp.atan2(cosr_cosp);

        // Pitch (rotation about y-axis)
        let sinp = (1.0 + 2.0 * (s * q1 - q0 * q2)).sqrt();
        let cosp = (1.0 - 2.0 * (s * q1 - q0 * q2)).sqrt();
        let pitch = 2.0 * sinp.atan2(cosp) - std::f64::consts::PI / 2.0;

        // Yaw (rotation about z-axis)
        let siny_cosp = 2.0 * (s * q2 + q0 * q1);
        let cosy_cosp = 1.0 - 2.0 * (q1 * q1 + q2 * q2);
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

    /// Update orientation quaternion (w, x, y, z)
    pub async fn update_orientation(&self, q0: f64, q1: f64, q2: f64, q3: f64) {
        let mut state = self.state.write().await;
        // q0 is scalar (w), q1,q2,q3 are vector (i,j,k)
        state.orientation = Quaternion::new(q0, q1, q2, q3);
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
