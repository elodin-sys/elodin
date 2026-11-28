use std::env;
use std::io;
use std::net::{Ipv4Addr, UdpSocket};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const STATE_SIZE: usize = 1024;
const COMMAND_SIZE: usize = 104;
/// Updates the helper thread makes for each tick.
const HELPER_STEPS: u64 = 64;

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn read_f64(bytes: &[u8], index: usize) -> f64 {
    let offset = 16 + index * 8;
    f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn write_f64(bytes: &mut [u8], index: usize, value: f64) {
    let offset = 16 + index * 8;
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[derive(Default)]
struct Shared {
    /// The tick the controller wants an update for.
    request: AtomicU64,
    count: AtomicU64,
    /// The last tick the helper finished.
    done: AtomicU64,
}

/// State kept on a thread of its own, which the controller adds to its commands from the fault
/// tick on. In `completion-race` it reads the state as soon as the helper starts updating it, so
/// what it reads depends on scheduling. In `causal-token` it waits for the helper to finish the
/// tick it asked for.
struct Helper {
    thread: std::thread::Thread,
    shared: Arc<Shared>,
    causal: bool,
}

impl Helper {
    fn spawn(causal: bool) -> Self {
        let shared = Arc::new(Shared::default());
        shared.request.store(u64::MAX, Ordering::Relaxed);
        shared.done.store(u64::MAX, Ordering::Relaxed);
        let worker = shared.clone();
        let thread = std::thread::spawn(move || {
            loop {
                let tick = worker.request.load(Ordering::Acquire);
                if tick == u64::MAX || worker.done.load(Ordering::Acquire) == tick {
                    std::thread::park();
                    continue;
                }
                for _ in 0..HELPER_STEPS {
                    worker.count.fetch_add(1, Ordering::AcqRel);
                    std::thread::yield_now();
                }
                worker.done.store(tick, Ordering::Release);
            }
        });
        Self {
            thread: thread.thread().clone(),
            shared,
            causal,
        }
    }

    fn state(&self, tick: u64) -> u64 {
        let shared = &self.shared;
        let before = shared.count.load(Ordering::Acquire);
        shared.request.store(tick, Ordering::Release);
        self.thread.unpark();
        loop {
            if self.causal {
                if shared.done.load(Ordering::Acquire) == tick {
                    return shared.count.load(Ordering::Acquire);
                }
            } else {
                // No token says the helper finished this tick: whatever it reached is used.
                let count = shared.count.load(Ordering::Acquire);
                if count != before {
                    return count;
                }
            }
            std::thread::yield_now();
        }
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: many-agent-controller STATE_PORT COMMAND_PORT AGENT");
        std::process::exit(2);
    }
    let state_port: u16 = args[1].parse().expect("invalid state port");
    let command_port: u16 = args[2].parse().expect("invalid command port");
    let agent: u64 = args[3].parse().expect("invalid agent");
    let fault_tick: u64 = env_or("ELODIN_MANY_AGENT_FAULT_TICK", 4096);
    let helper = match env::var("ELODIN_MANY_AGENT_FAULT_MODE").as_deref() {
        _ if env_or("ELODIN_MANY_AGENT_FAULT_AGENT", 7) != agent => None,
        Ok("completion-race") => Some(Helper::spawn(false)),
        Ok("causal-token") => Some(Helper::spawn(true)),
        _ => None,
    };
    let socket = UdpSocket::bind((Ipv4Addr::LOCALHOST, state_port))?;
    let command_address = (Ipv4Addr::LOCALHOST, command_port);
    let mut state = [0u8; STATE_SIZE];
    loop {
        let (size, _) = socket.recv_from(&mut state)?;
        let received = Instant::now();
        if size != STATE_SIZE
            || &state[0..4] != b"MALS"
            || read_u16(&state, 4) != 1
            || read_u16(&state, 6) != 0
        {
            continue;
        }
        let tick = read_u64(&state, 8);
        let mut command = [0.0; 3];
        for axis in 0..3 {
            let position_error = read_f64(&state, 6 + axis) - read_f64(&state, axis);
            let velocity_error = read_f64(&state, 9 + axis) - read_f64(&state, 3 + axis);
            command[axis] = (1.4 * position_error + 0.45 * velocity_error).clamp(-12.0, 12.0);
        }
        if let Some(helper) = &helper
            && tick >= fault_tick
        {
            command[0] += helper.state(tick) as f64 * 1e-9;
        }
        let mut response = [0u8; COMMAND_SIZE];
        response[0..4].copy_from_slice(b"MALC");
        response[4..6].copy_from_slice(&1u16.to_le_bytes());
        response[6..8].copy_from_slice(&0u16.to_le_bytes());
        response[8..16].copy_from_slice(&tick.to_le_bytes());
        for (axis, value) in command.into_iter().enumerate() {
            write_f64(&mut response, axis, value);
        }
        let service_ns = received.elapsed().as_nanos() as u64;
        response[COMMAND_SIZE - 8..].copy_from_slice(&service_ns.to_le_bytes());
        socket.send_to(&response, command_address)?;
    }
}
