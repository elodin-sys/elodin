use nalgebra::{Quaternion, Vector3};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Telemetry state aggregated from various components
#[derive(Debug, Clone)]
pub struct TelemetryState {
    /// Attitude quaternion (w, x, y, z)
    pub attitude: Quaternion<f32>,
    
    /// Angular velocity in body frame (rad/s)
    pub angular_vel: Vector3<f32>,
    
    /// Linear acceleration in body frame (m/s²)
    pub linear_accel: Vector3<f32>,
    
    /// Altitude above sea level (meters)
    pub altitude_m: f32,
    
    /// Ground speed (m/s)
    pub ground_speed_ms: f32,
    
    /// Heading in degrees (0-360)
    pub heading_deg: f32,
    
    /// Climb rate (m/s, positive up)
    pub climb_rate_ms: f32,
    
    /// Roll angle in degrees
    pub roll_deg: f32,
    
    /// Pitch angle in degrees
    pub pitch_deg: f32,
    
    /// Number of GPS satellites
    pub gps_sats: u8,
    
    /// System health/status
    pub system_status: SystemStatus,
    
    /// Battery voltage
    pub battery_voltage: f32,
    
    /// Current draw in amps
    pub current_amps: f32,
    
    /// Timestamp of last update
    pub last_update: std::time::Instant,
    
    /// Database connection status
    pub db_connected: bool,
    
    /// Number of telemetry updates received
    pub update_count: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SystemStatus {
    Initializing,
    Ready,
    Armed,
    Flying,
    Warning(String),
    Error(String),
}

impl Default for TelemetryState {
    fn default() -> Self {
        Self {
            attitude: Quaternion::identity(),
            angular_vel: Vector3::zeros(),
            linear_accel: Vector3::new(0.0, 0.0, -9.81),
            altitude_m: 0.0,
            ground_speed_ms: 0.0,
            heading_deg: 0.0,
            climb_rate_ms: 0.0,
            roll_deg: 0.0,
            pitch_deg: 0.0,
            gps_sats: 0,
            system_status: SystemStatus::Initializing,
            battery_voltage: 0.0,
            current_amps: 0.0,
            last_update: std::time::Instant::now(),
            db_connected: false,
            update_count: 0,
        }
    }
}

/// Processes incoming telemetry data and maintains current state
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

    /// Update gyro data (angular velocity)
    pub async fn update_gyro(&self, x: f32, y: f32, z: f32) {
        let mut state = self.state.write().await;
        state.angular_vel = Vector3::new(x, y, z);
        state.last_update = std::time::Instant::now();
    }

    /// Update accelerometer data
    pub async fn update_accel(&self, x: f32, y: f32, z: f32) {
        let mut state = self.state.write().await;
        state.linear_accel = Vector3::new(x, y, z);
        state.last_update = std::time::Instant::now();
    }

    /// Update magnetometer data and calculate heading
    pub async fn update_magnetometer(&self, x: f32, y: f32, _z: f32) {
        let mut state = self.state.write().await;
        // Simple heading calculation from mag x,y
        let heading_rad = y.atan2(x);
        state.heading_deg = heading_rad.to_degrees();
        if state.heading_deg < 0.0 {
            state.heading_deg += 360.0;
        }
        state.last_update = std::time::Instant::now();
    }

    /// Update attitude from quaternion
    pub async fn update_attitude(&self, w: f32, x: f32, y: f32, z: f32) {
        let mut state = self.state.write().await;
        state.attitude = Quaternion::new(w, x, y, z);
        
        // Calculate roll and pitch from quaternion
        // Using standard aerospace convention
        let (roll, pitch, _yaw) = quaternion_to_euler(&state.attitude);
        state.roll_deg = roll.to_degrees();
        state.pitch_deg = pitch.to_degrees();
        
        state.last_update = std::time::Instant::now();
    }

    /// Update position data
    pub async fn update_position(&self, _x: f64, _y: f64, z: f64) {
        let mut state = self.state.write().await;
        state.altitude_m = z as f32;
        state.last_update = std::time::Instant::now();
    }

    /// Update velocity data
    pub async fn update_velocity(&self, vx: f64, vy: f64, vz: f64) {
        let mut state = self.state.write().await;
        // Ground speed from horizontal velocity components
        state.ground_speed_ms = ((vx * vx + vy * vy).sqrt()) as f32;
        // Climb rate from vertical velocity
        state.climb_rate_ms = vz as f32;
        state.last_update = std::time::Instant::now();
    }

    /// Update system status
    pub async fn update_status(&self, status: SystemStatus) {
        let mut state = self.state.write().await;
        state.system_status = status;
        state.last_update = std::time::Instant::now();
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

    /// Process a generic component update
    pub async fn process_component(&self, component_name: &str, data: &[u8]) {
        // This would parse the actual component data based on the schema
        // For now, we'll handle known component names
        match component_name {
            "gyro" => {
                if data.len() >= 24 {
                    // Assuming f64 values (8 bytes each)
                    let x = f64::from_le_bytes(data[0..8].try_into().unwrap_or_default()) as f32;
                    let y = f64::from_le_bytes(data[8..16].try_into().unwrap_or_default()) as f32;
                    let z = f64::from_le_bytes(data[16..24].try_into().unwrap_or_default()) as f32;
                    self.update_gyro(x, y, z).await;
                }
            }
            "accel" => {
                if data.len() >= 24 {
                    let x = f64::from_le_bytes(data[0..8].try_into().unwrap_or_default()) as f32;
                    let y = f64::from_le_bytes(data[8..16].try_into().unwrap_or_default()) as f32;
                    let z = f64::from_le_bytes(data[16..24].try_into().unwrap_or_default()) as f32;
                    self.update_accel(x, y, z).await;
                }
            }
            "magnetometer" => {
                if data.len() >= 24 {
                    let x = f64::from_le_bytes(data[0..8].try_into().unwrap_or_default()) as f32;
                    let y = f64::from_le_bytes(data[8..16].try_into().unwrap_or_default()) as f32;
                    let z = f64::from_le_bytes(data[16..24].try_into().unwrap_or_default()) as f32;
                    self.update_magnetometer(x, y, z).await;
                }
            }
            "world_pos" => {
                if data.len() >= 24 {
                    let x = f64::from_le_bytes(data[0..8].try_into().unwrap_or_default());
                    let y = f64::from_le_bytes(data[8..16].try_into().unwrap_or_default());
                    let z = f64::from_le_bytes(data[16..24].try_into().unwrap_or_default());
                    self.update_position(x, y, z).await;
                }
            }
            "world_vel" => {
                if data.len() >= 24 {
                    let vx = f64::from_le_bytes(data[0..8].try_into().unwrap_or_default());
                    let vy = f64::from_le_bytes(data[8..16].try_into().unwrap_or_default());
                    let vz = f64::from_le_bytes(data[16..24].try_into().unwrap_or_default());
                    self.update_velocity(vx, vy, vz).await;
                }
            }
            _ => {
                // Unknown component, ignore for now
                tracing::trace!("Unknown component: {}", component_name);
            }
        }
    }
}

/// Convert quaternion to Euler angles (roll, pitch, yaw) in radians
fn quaternion_to_euler(q: &Quaternion<f32>) -> (f32, f32, f32) {
    let w = q.w;
    let x = q.i;
    let y = q.j;
    let z = q.k;

    // Roll (x-axis rotation)
    let sinr_cosp = 2.0 * (w * x + y * z);
    let cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
    let roll = sinr_cosp.atan2(cosr_cosp);

    // Pitch (y-axis rotation)
    let sinp = 2.0 * (w * y - z * x);
    let pitch = if sinp.abs() >= 1.0 {
        std::f32::consts::FRAC_PI_2.copysign(sinp)
    } else {
        sinp.asin()
    };

    // Yaw (z-axis rotation)
    let siny_cosp = 2.0 * (w * z + x * y);
    let cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    let yaw = siny_cosp.atan2(cosy_cosp);

    (roll, pitch, yaw)
}
