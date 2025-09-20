use iocraft::prelude::*;
use std::{
    collections::{VecDeque, HashMap},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use anyhow::Result;
use iocraft::Props;

#[derive(Clone, Debug)]
pub struct TelemetryRow {
    pub timestamp: Instant,
    pub component_name: String,
    pub values: Vec<f64>,
    pub unit: String,
    pub is_waiting: bool,  // Indicates this is a "waiting for telemetry" placeholder
}

impl TelemetryRow {
    /// Create a "waiting for connection" placeholder row
    pub fn waiting_for_connection() -> Self {
        TelemetryRow {
            timestamp: Instant::now(),
            component_name: "waiting_for_connection".to_string(),
            values: Vec::new(),
            unit: String::new(),
            is_waiting: true,
        }
    }
    
    /// Create a "waiting for reconnection" placeholder row
    pub fn waiting_for_reconnection() -> Self {
        TelemetryRow {
            timestamp: Instant::now(),
            component_name: "waiting_for_reconnection".to_string(),
            values: Vec::new(),
            unit: String::new(),
            is_waiting: true,
        }
    }
    
    /// Create a "connected" indicator row
    pub fn connected() -> Self {
        TelemetryRow {
            timestamp: Instant::now(),
            component_name: "connected".to_string(),
            values: Vec::new(),
            unit: String::new(),
            is_waiting: true,
        }
    }
    
    /// Create a generic waiting row
    pub fn waiting() -> Self {
        Self::waiting_for_connection()
    }
}

const CAPACITY: usize = 20_000;      // ring buffer size
const FRAME_MS: u64 = 33;            // ~30 FPS

// Categorize components for display
fn categorize_component(name: &str) -> &'static str {
    if name.contains("thrust") || name.contains("motor") {
        "🔥 Propulsion"
    } else if name.contains("fin") || name.contains("pid") {
        "🎯 Control"
    } else if name.contains("aero") || name.contains("mach") || name.contains("pressure") || name.contains("angle") || name.contains("wind") {
        "💨 Aerodynamics"
    } else if name.contains("pos") || name.contains("vel") || name.contains("accel") || name.contains("gravity") {
        "📍 Position/Motion"
    } else {
        "📊 Other"
    }
}

fn format_values(values: &[f64]) -> String {
    if values.is_empty() {
        return String::from("No data");
    } else if values.len() == 1 {
        return format!("{:8.2}", values[0]);
    } else if values.len() <= 3 {
        let formatted: Vec<String> = values.iter()
            .map(|v| format!("{:8.2}", v))
            .collect();
        return format!("[{}]", formatted.join(", "));
    } else if values.len() <= 20 {
        // Show first few and last few for moderate arrays
        let first: Vec<String> = values[..3.min(values.len())].iter()
            .map(|v| format!("{:6.1}", v))
            .collect();
        let last: Vec<String> = values[values.len().saturating_sub(2)..].iter()
            .map(|v| format!("{:6.1}", v))
            .collect();
        return format!("[{} ... {}] ({} values)", 
            first.join(", "), last.join(", "), values.len());
    } else {
        return format!("[{} values]", values.len());
    }
}

#[derive(Props, Clone)]
pub struct TelemetryAppProps {
    pub receiver: async_channel::Receiver<TelemetryRow>,
}

impl Default for TelemetryAppProps {
    fn default() -> Self {
        // Create a dummy channel for default
        let (_, receiver) = async_channel::bounded(1);
        Self { receiver }
    }
}

#[component]
pub fn TelemetryApp(
    props: &TelemetryAppProps,
    mut hooks: Hooks,
) -> impl Into<AnyElement<'static>> {
    let receiver = props.receiver.clone();
    // --- Shared ring buffer (no re-render on every message) ---
    let ring = hooks.use_const(|| Arc::new(Mutex::new(VecDeque::<TelemetryRow>::with_capacity(CAPACITY))));
    
    // Telemetry consumer from channel
    {
        let ring = ring.clone();
        hooks.use_future(async move {
            loop {
                match receiver.recv().await {
                    Ok(row) => {
                        let mut q = ring.lock().unwrap();
                        q.push_back(row);
                        if q.len() > CAPACITY {
                            q.pop_front();
                        }
                    },
                    Err(_) => {
                        // Channel closed
                        break;
                    }
                }
            }
        });
    }

    // Scrollback and frame ticker to coalesce renders
    let mut scroll_back = hooks.use_state(|| 0usize);
    let mut tick = hooks.use_state(|| 0u64);
    let mut should_exit = hooks.use_state(|| false);
    
    hooks.use_future(async move {
        loop {
            smol::Timer::after(Duration::from_millis(FRAME_MS)).await;
            tick += 1; // trigger a render
        }
    });

    // Simple keyboard handling without SystemContext
    hooks.use_terminal_events({
        move |event| match event {
            TerminalEvent::Key(KeyEvent { code, kind, .. }) if kind != KeyEventKind::Release => {
                match code {
                    KeyCode::Char('q') => should_exit.set(true),
                    KeyCode::Up | KeyCode::Char('k')   => scroll_back.set(scroll_back.get().saturating_add(1)),
                    KeyCode::Down | KeyCode::Char('j') => scroll_back.set(scroll_back.get().saturating_sub(1)),
                    KeyCode::Home => scroll_back.set(0),
                    _ => {}
                }
            }
            _ => {}
        }
    });
    
    // Exit if requested
    if should_exit.get() {
        let mut system = hooks.use_context_mut::<SystemContext>();
        system.exit();
    }

    // Virtualize: only draw what fits in the terminal height
    let (_w, h) = hooks.use_terminal_size();
    let rows_visible = h.saturating_sub(10) as usize; // header + padding

    // Group telemetry by category for better display
    let display_text = {
        let q = ring.lock().unwrap();
        let mut latest_values: HashMap<String, TelemetryRow> = HashMap::new();
        
        // Check if we have any real data or just waiting messages
        let has_real_data = q.iter().any(|r| !r.is_waiting);
        
        if !has_real_data && q.iter().any(|r| r.is_waiting) {
            // Get the latest waiting state
            let waiting_state = q.iter()
                .rev()
                .find(|r| r.is_waiting)
                .map(|r| r.component_name.as_str())
                .unwrap_or("waiting_for_connection");
            
            let lines = match waiting_state {
                "waiting_for_connection" => vec![
                    String::new(),
                    String::new(),
                    "🔌 Waiting for database connection...".to_string(),
                    String::new(),
                    "The client is waiting to connect to the database.".to_string(),
                    String::new(),
                    "Start the simulation with: python rocket.py run 0.0.0.0:2240".to_string(),
                    "Or start elodin-db separately, then the simulation.".to_string(),
                ],
                "waiting_for_reconnection" => vec![
                    String::new(),
                    String::new(),
                    "🔄 Reconnecting to database...".to_string(),
                    String::new(),
                    "Connection lost, attempting to reconnect automatically.".to_string(),
                    String::new(),
                    "The client will reconnect when the database becomes available.".to_string(),
                ],
                "connected" => vec![
                    String::new(),
                    String::new(),
                    "✅ Connected! Waiting for telemetry data...".to_string(),
                    String::new(),
                    "Database connected, waiting for simulation to send data.".to_string(),
                    String::new(),
                    "The simulation should start sending telemetry shortly.".to_string(),
                ],
                _ => vec![
                    String::new(),
                    String::new(),
                    "⏳ Waiting for telemetry data...".to_string(),
                    String::new(),
                    "Please start the simulation.".to_string(),
                    String::new(),
                    "The client will automatically connect when data becomes available.".to_string(),
                ],
            };
            (lines.join("\n"), 0)
        } else {
            // Get the latest value for each component (skip waiting messages)
            for row in q.iter().rev().take(1000) {  // Look at last 1000 entries max
                if !row.is_waiting {
                    latest_values.entry(row.component_name.clone()).or_insert(row.clone());
                }
            }
            
            // Group by category
            let mut by_category: HashMap<&str, Vec<TelemetryRow>> = HashMap::new();
            for row in latest_values.values() {
                let category = categorize_component(&row.component_name);
                by_category.entry(category).or_insert_with(Vec::new).push(row.clone());
            }
            
            // Sort within each category
            for rows in by_category.values_mut() {
                rows.sort_by(|a, b| a.component_name.cmp(&b.component_name));
            }
            
            // Build display text
            let mut lines = Vec::new();
            let category_order = ["🔥 Propulsion", "🎯 Control", "💨 Aerodynamics", "📍 Position/Motion"];
            
            for category_name in &category_order {
                if let Some(rows) = by_category.get(category_name) {
                    lines.push(String::new()); // spacer
                    lines.push(format!("{}", category_name));
                    lines.push("════════════════════════════════════════════════════════════════════════════════".to_string());
                    
                    for row in rows.iter().skip(scroll_back.get()).take(rows_visible) {
                        lines.push(format!(
                            "  {:<25}: {:>40} {}",
                            row.component_name.replace("rocket.", ""),
                            format_values(&row.values),
                            row.unit
                        ));
                    }
                }
            }
            
            (lines.join("\n"), q.iter().filter(|r| !r.is_waiting).count())
        }
    };
    
    let (telemetry_text, total_len) = display_text;
    let packet_count = total_len / 20; // Rough estimate of packets
    
    element! {
        View(
            flex_direction: FlexDirection::Column,
            width: 100pct,
            height: 100pct,
            border_style: BorderStyle::Round,
            border_color: Color::Blue,
        ) {
            // Title header
            View(padding: 1, flex_direction: FlexDirection::Column) {
                Text(
                    weight: Weight::Bold, 
                    color: Color::Cyan,
                    content: "╔═══════════════════════════════════════════════════════════════════════════════╗".to_string()
                )
                Text(
                    weight: Weight::Bold,
                    color: Color::Cyan, 
                    align: TextAlign::Center,
                    content: if total_len == 0 {
                        "║  🚀 ROCKET TELEMETRY DASHBOARD - WAITING FOR DATA  ║".to_string()
                    } else {
                        "║  🚀 ROCKET TELEMETRY DASHBOARD - REAL-TIME VALUES  ║".to_string()
                    }
                )
                Text(
                    weight: Weight::Bold,
                    color: Color::Cyan,
                    content: "╚═══════════════════════════════════════════════════════════════════════════════╝".to_string()
                )
                Text(content: "".to_string()) // Spacer
                Text(
                    color: if total_len == 0 { Color::Yellow } else { Color::Green },
                    content: if total_len == 0 {
                        "📡 Connected to database | ⏳ Waiting for simulation...".to_string()
                    } else {
                        format!("📡 Connected | 📦 Packets: ~{} | ⏱️  Tick: {}", packet_count, tick.get())
                    }
                )
            }

            // Body - Show categorized telemetry as a single text block
            View(height: 100pct, padding_left: 1, padding_right: 1, overflow_y: Some(Overflow::Clip)) {
                Text(
                    wrap: TextWrap::NoWrap,
                    content: telemetry_text
                )
            }

            // Footer / status  
            View(padding: 1, border_style: BorderStyle::Single, border_color: Color::DarkGrey) {
                View(flex_direction: FlexDirection::Column) {
                    Text(
                        color: Color::DarkGrey,
                        content: "─────────────────────────────────────────────────────────────────".to_string()
                    )
                    Text(
                        color: Color::Cyan,
                        content: format!(
                            "💡 Press 'q' to exit | ↑↓ or j/k to scroll | Home for top | Scroll: {}",
                            scroll_back.get()
                        )
                    )
                }
            }
        }
    }
}

pub async fn run_tui(receiver: async_channel::Receiver<TelemetryRow>) -> Result<()> {
    element!(TelemetryApp(receiver))
        .fullscreen()
        .await?;
    Ok(())
}