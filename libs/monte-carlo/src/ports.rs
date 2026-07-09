//! Worker port planning: up-front validation of the whole static plan,
//! per-run dynamic allocation (`"auto"` ports), kernel ephemeral-range
//! checks, preflight squatter detection with pid attribution, and reaping of
//! processes still bound to campaign ports.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::net::{IpAddr, TcpListener, UdpSocket};

use miette::{Result, miette};

use crate::{CONTEXT_ENV, PortSpec, ResourceConfig};

/// One statically planned port across the campaign, used for validation,
/// ephemeral-range warnings, and startup reaping.
#[derive(Clone, Debug)]
pub struct PlannedPort {
    pub worker_id: usize,
    pub name: String,
    pub port: u16,
}

/// A worker's port template: static ports are fixed for the campaign, `None`
/// entries are allocated fresh for every run.
#[derive(Clone, Debug)]
pub struct SlotTemplate {
    pub worker_id: usize,
    pub bind_ip: IpAddr,
    pub db_port: Option<u16>,
    pub ports: BTreeMap<String, Option<u16>>,
}

/// Keeps a dynamically allocated port bound until just before the run's
/// processes spawn, shrinking the window in which another process could steal
/// it. Both protocols are held since campaign ports are used for TCP and UDP.
pub struct PortGuard {
    pub port: u16,
    _tcp: TcpListener,
    _udp: UdpSocket,
}

/// Compute a worker's static port template, failing on u16 overflow with the
/// exact worker/name/port. Auto ports come back as `None`.
pub fn slot_template(worker_id: usize, resources: &ResourceConfig) -> Result<SlotTemplate> {
    let offset = worker_id
        .checked_mul(resources.port_stride as usize)
        .ok_or_else(|| miette!("worker {worker_id}: port offset overflows"))?;
    let shift = |name: &str, base: u16| -> Result<u16> {
        u16::try_from(base as usize + offset).map_err(|_| {
            miette!(
                "worker {worker_id}: port `{name}` overflows 16 bits ({base} + {offset}); \
                 lower [resources] bases or port_stride, or use \"auto\" ports"
            )
        })
    };
    let db_port = match resources.db_port {
        PortSpec::Static(base) => Some(shift("db_port", base)?),
        PortSpec::Auto(_) => None,
    };
    let mut ports = BTreeMap::new();
    for (name, spec) in &resources.ports {
        let port = match spec {
            PortSpec::Static(base) => Some(shift(name, *base)?),
            PortSpec::Auto(_) => None,
        };
        ports.insert(name.clone(), port);
    }
    // elodin-db always serves editor/render-server assets on `db_port + 1`
    // (the headless sensor-camera renderer fetches scene assets from it), so
    // the assets port is part of every slot: validated, preflight-probed, and
    // exported as ELODIN_MC_PORT_DB_ASSETS.
    if let Some(db) = db_port {
        let assets = db
            .checked_add(1)
            .ok_or_else(|| miette!("worker {worker_id}: assets port (db_port + 1) overflows"))?;
        ports.insert("db_assets".to_string(), Some(assets));
    }
    let mut seen: HashMap<u16, String> = HashMap::new();
    if let Some(db) = db_port {
        seen.insert(db, "db_port".to_string());
    }
    for (name, port) in &ports {
        let Some(port) = port else { continue };
        if let Some(other) = seen.insert(*port, name.clone()) {
            return Err(miette!(
                "worker {worker_id}: port collision at {port} between `{name}` and `{other}`; \
                 spread [resources] bases or raise port_stride"
            ));
        }
    }
    Ok(SlotTemplate {
        worker_id,
        bind_ip: resources.bind_ip,
        db_port,
        ports,
    })
}

/// Validate the whole static port plan for `workers` workers before the
/// campaign starts, so a bad stride/base combination fails fast instead of as
/// an opaque mid-campaign worker death. Returns every planned static port.
pub fn validate_port_plan(resources: &ResourceConfig, workers: usize) -> Result<Vec<PlannedPort>> {
    let mut planned = Vec::new();
    let mut seen: HashMap<u16, (usize, String)> = HashMap::new();
    for worker_id in 0..workers {
        let template = slot_template(worker_id, resources)?;
        if let Some(port) = template.db_port {
            check_global_collision(&mut seen, worker_id, "db_port", port)?;
            planned.push(PlannedPort {
                worker_id,
                name: "db_port".to_string(),
                port,
            });
        }
        for (name, port) in template.ports {
            if let Some(port) = port {
                check_global_collision(&mut seen, worker_id, &name, port)?;
                planned.push(PlannedPort {
                    worker_id,
                    name,
                    port,
                });
            }
        }
    }
    Ok(planned)
}

fn check_global_collision(
    seen: &mut HashMap<u16, (usize, String)>,
    worker_id: usize,
    name: &str,
    port: u16,
) -> Result<()> {
    if let Some((other_worker, other_name)) = seen.insert(port, (worker_id, name.to_string())) {
        return Err(miette!(
            "port collision at {port}: worker {other_worker} `{other_name}` and worker {worker_id} `{name}` both plan this port; \
             adjust [resources] bases or port_stride"
        ));
    }
    Ok(())
}

/// The kernel's local (ephemeral) port range: any outbound connection may be
/// assigned a source port in this window, silently stealing it from a
/// listener that binds later.
#[cfg(target_os = "linux")]
pub fn ephemeral_range() -> Option<(u16, u16)> {
    let text = std::fs::read_to_string("/proc/sys/net/ipv4/ip_local_port_range").ok()?;
    let mut parts = text.split_whitespace();
    let low = parts.next()?.parse().ok()?;
    let high = parts.next()?.parse().ok()?;
    Some((low, high))
}

#[cfg(not(target_os = "linux"))]
pub fn ephemeral_range() -> Option<(u16, u16)> {
    None
}

/// Human-readable description of planned ports inside the ephemeral range,
/// or `None` when the plan is clean.
pub fn ephemeral_conflicts(planned: &[PlannedPort]) -> Option<String> {
    let (low, high) = ephemeral_range()?;
    let offenders: Vec<&PlannedPort> = planned
        .iter()
        .filter(|planned| planned.port >= low && planned.port <= high)
        .collect();
    let first = offenders.first()?;
    Some(format!(
        "{} planned port(s) fall inside the kernel ephemeral range {low}-{high} \
         (first: worker {} `{}` at {}); outbound connections can steal them. \
         Move [resources] bases below {low} or use \"auto\" ports",
        offenders.len(),
        first.worker_id,
        first.name,
        first.port,
    ))
}

/// Allocate a fresh port that is free for both TCP and UDP, holding guard
/// sockets so it stays ours until the run's processes spawn.
pub fn allocate_port(bind_ip: IpAddr) -> Result<PortGuard> {
    for _ in 0..64 {
        let Ok(udp) = UdpSocket::bind((bind_ip, 0)) else {
            break;
        };
        let Ok(port) = udp.local_addr().map(|addr| addr.port()) else {
            continue;
        };
        if let Ok(tcp) = TcpListener::bind((bind_ip, port)) {
            return Ok(PortGuard {
                port,
                _tcp: tcp,
                _udp: udp,
            });
        }
    }
    Err(miette!(
        "failed to allocate a dynamic port on {bind_ip} (64 attempts)"
    ))
}

/// Allocate a consecutive port pair `(P, P + 1)`, both free for TCP and UDP.
/// Used for `db_port = "auto"`: elodin-db implicitly serves assets on
/// `db_port + 1`, so a dynamically allocated DB port must bring its assets
/// port with it.
pub fn allocate_port_pair(bind_ip: IpAddr) -> Result<(PortGuard, PortGuard)> {
    for _ in 0..64 {
        let Ok(udp) = UdpSocket::bind((bind_ip, 0)) else {
            break;
        };
        let Ok(port) = udp.local_addr().map(|addr| addr.port()) else {
            continue;
        };
        let Some(assets_port) = port.checked_add(1) else {
            continue;
        };
        let Ok(tcp) = TcpListener::bind((bind_ip, port)) else {
            continue;
        };
        let Ok(assets_udp) = UdpSocket::bind((bind_ip, assets_port)) else {
            continue;
        };
        let Ok(assets_tcp) = TcpListener::bind((bind_ip, assets_port)) else {
            continue;
        };
        return Ok((
            PortGuard {
                port,
                _tcp: tcp,
                _udp: udp,
            },
            PortGuard {
                port: assets_port,
                _tcp: assets_tcp,
                _udp: assets_udp,
            },
        ));
    }
    Err(miette!(
        "failed to allocate a dynamic db/assets port pair on {bind_ip} (64 attempts)"
    ))
}

/// A process currently bound to (listening on) a local port.
#[derive(Clone, Debug)]
pub struct PortOwner {
    pub port: u16,
    pub protocol: &'static str,
    pub pid: Option<u32>,
    pub comm: Option<String>,
}

impl std::fmt::Display for PortOwner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.pid, &self.comm) {
            (Some(pid), Some(comm)) => {
                write!(
                    f,
                    "port {} ({}) already bound by pid {pid} ({comm})",
                    self.port, self.protocol
                )
            }
            (Some(pid), None) => write!(
                f,
                "port {} ({}) already bound by pid {pid}",
                self.port, self.protocol
            ),
            _ => write!(
                f,
                "port {} ({}) already bound by another process",
                self.port, self.protocol
            ),
        }
    }
}

/// Find processes bound to any of `ports`. Linux implementation reads
/// `/proc/net/{tcp,tcp6,udp,udp6}` (TCP entries only count in LISTEN state,
/// so TIME_WAIT remnants of a finished run never false-positive) and resolves
/// socket inodes to pids via `/proc/*/fd`.
#[cfg(target_os = "linux")]
pub fn find_port_owners(ports: &HashSet<u16>) -> Vec<PortOwner> {
    const TCP_LISTEN: u32 = 0x0A;
    // UDP sockets are "bound" in state 7 (TCP_CLOSE re-used as unconnected).
    let mut inodes: HashMap<u64, (u16, &'static str)> = HashMap::new();
    for (path, protocol, require_listen) in [
        ("/proc/net/tcp", "tcp", true),
        ("/proc/net/tcp6", "tcp", true),
        ("/proc/net/udp", "udp", false),
        ("/proc/net/udp6", "udp", false),
    ] {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        for line in text.lines().skip(1) {
            let fields: Vec<&str> = line.split_whitespace().collect();
            let (Some(local), Some(state), Some(inode)) =
                (fields.get(1), fields.get(3), fields.get(9))
            else {
                continue;
            };
            let Some((_, port_hex)) = local.rsplit_once(':') else {
                continue;
            };
            let Ok(port) = u16::from_str_radix(port_hex, 16) else {
                continue;
            };
            if !ports.contains(&port) {
                continue;
            }
            let Ok(state) = u32::from_str_radix(state, 16) else {
                continue;
            };
            if require_listen && state != TCP_LISTEN {
                continue;
            }
            let Ok(inode) = inode.parse::<u64>() else {
                continue;
            };
            if inode == 0 {
                continue;
            }
            inodes.insert(inode, (port, protocol));
        }
    }
    if inodes.is_empty() {
        return Vec::new();
    }
    let mut owners: HashMap<u64, (u32, String)> = HashMap::new();
    if let Ok(proc_entries) = std::fs::read_dir("/proc") {
        for entry in proc_entries.flatten() {
            let name = entry.file_name();
            let Some(pid) = name.to_str().and_then(|name| name.parse::<u32>().ok()) else {
                continue;
            };
            let fd_dir = entry.path().join("fd");
            let Ok(fds) = std::fs::read_dir(&fd_dir) else {
                continue;
            };
            for fd in fds.flatten() {
                let Ok(target) = std::fs::read_link(fd.path()) else {
                    continue;
                };
                let target = target.to_string_lossy();
                let Some(inode) = target
                    .strip_prefix("socket:[")
                    .and_then(|rest| rest.strip_suffix(']'))
                    .and_then(|inode| inode.parse::<u64>().ok())
                else {
                    continue;
                };
                if inodes.contains_key(&inode) && !owners.contains_key(&inode) {
                    let comm = std::fs::read_to_string(entry.path().join("comm"))
                        .map(|comm| comm.trim().to_string())
                        .unwrap_or_default();
                    owners.insert(inode, (pid, comm));
                }
            }
        }
    }
    inodes
        .into_iter()
        .map(|(inode, (port, protocol))| {
            let owner = owners.get(&inode);
            PortOwner {
                port,
                protocol,
                pid: owner.map(|(pid, _)| *pid),
                comm: owner
                    .map(|(_, comm)| comm.clone())
                    .filter(|c| !c.is_empty()),
            }
        })
        .collect()
}

#[cfg(not(target_os = "linux"))]
pub fn find_port_owners(ports: &HashSet<u16>) -> Vec<PortOwner> {
    // Fallback: bind probes. TIME_WAIT can false-positive on TCP, so only
    // check UDP (the common leak class) off Linux.
    let mut owners = Vec::new();
    for &port in ports {
        if UdpSocket::bind(("0.0.0.0", port)).is_err() {
            owners.push(PortOwner {
                port,
                protocol: "udp",
                pid: None,
                comm: None,
            });
        }
    }
    owners
}

/// Kill processes still bound to campaign ports (SIGTERM, grace, SIGKILL) —
/// stragglers from a previous campaign whose cgroup could not be reaped.
/// Returns a description of what was reaped.
#[cfg(unix)]
pub fn reap_port_squatters(ports: &HashSet<u16>) -> Result<Vec<String>> {
    let owners = find_port_owners(ports);
    if owners.is_empty() {
        return Ok(Vec::new());
    }
    let me = std::process::id();
    let mut pids = HashSet::new();
    let mut foreign = Vec::new();
    for owner in &owners {
        let Some(pid) = owner.pid else {
            foreign.push(owner.to_string());
            continue;
        };
        if pid == me {
            continue;
        }
        if process_has_campaign_context(pid) {
            pids.insert(pid);
        } else {
            foreign.push(owner.to_string());
        }
    }
    if !foreign.is_empty() {
        return Err(miette!(
            "planned campaign ports are already in use by non-campaign process(es): {}; \
             stop them or change [resources] bases",
            foreign.join("; ")
        ));
    }
    if pids.is_empty() {
        return Ok(Vec::new());
    }
    let mut reaped = Vec::new();
    for owner in &owners {
        let Some(pid) = owner.pid else { continue };
        if !pids.contains(&pid) {
            continue;
        }
        reaped.push(format!(
            "reaped pid {pid} ({}) bound to campaign port {} ({})",
            owner.comm.as_deref().unwrap_or("?"),
            owner.port,
            owner.protocol
        ));
    }
    for pid in &pids {
        if *pid == me {
            continue;
        }
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(*pid as i32),
            nix::sys::signal::Signal::SIGTERM,
        );
    }
    std::thread::sleep(std::time::Duration::from_millis(500));
    for pid in &pids {
        if *pid == me {
            continue;
        }
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(*pid as i32),
            nix::sys::signal::Signal::SIGKILL,
        );
    }
    Ok(reaped)
}

#[cfg(not(unix))]
pub fn reap_port_squatters(_ports: &HashSet<u16>) -> Result<Vec<String>> {
    Ok(Vec::new())
}

#[cfg(target_os = "linux")]
fn process_has_campaign_context(pid: u32) -> bool {
    let Ok(environ) = std::fs::read(format!("/proc/{pid}/environ")) else {
        return false;
    };
    environ_contains_campaign_context(&environ)
}

#[cfg(target_os = "linux")]
fn environ_contains_campaign_context(environ: &[u8]) -> bool {
    let prefix = format!("{CONTEXT_ENV}=").into_bytes();
    environ
        .split(|byte| *byte == 0)
        .any(|entry| entry.starts_with(&prefix))
}

#[cfg(all(unix, not(target_os = "linux")))]
fn process_has_campaign_context(_pid: u32) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv6Addr;

    fn resources(stride: u16, db: u16, ports: &[(&str, PortSpec)]) -> ResourceConfig {
        ResourceConfig {
            bind_ip: IpAddr::V6(Ipv6Addr::UNSPECIFIED),
            port_stride: stride,
            db_port: PortSpec::Static(db),
            ports: ports
                .iter()
                .map(|(name, spec)| (name.to_string(), *spec))
                .collect(),
        }
    }

    /// The customer's original broken plan: `ardupilot = 63333` with stride
    /// 100 overflows u16 at worker 23. Must be rejected up front, naming the
    /// worker and the port.
    #[test]
    fn plan_validation_names_overflowing_worker() {
        let resources = resources(100, 2240, &[("ardupilot", PortSpec::Static(63333))]);
        let err = validate_port_plan(&resources, 96).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("worker 23"), "got: {message}");
        assert!(message.contains("ardupilot"), "got: {message}");
    }

    #[test]
    fn plan_validation_names_colliding_ports() {
        let resources = resources(
            10,
            2000,
            &[("a", PortSpec::Static(3000)), ("b", PortSpec::Static(3000))],
        );
        let err = validate_port_plan(&resources, 4).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("collision"), "got: {message}");
        assert!(message.contains("3000"), "got: {message}");
    }

    #[test]
    fn plan_validation_rejects_cross_worker_port_collisions() {
        let resources = resources(32, 20000, &[("sitl_socket", PortSpec::Static(20032))]);
        let err = validate_port_plan(&resources, 4).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("worker 0"), "got: {message}");
        assert!(message.contains("sitl_socket"), "got: {message}");
        assert!(message.contains("worker 1"), "got: {message}");
        assert!(message.contains("db_port"), "got: {message}");
        assert!(message.contains("20032"), "got: {message}");
    }

    /// elodin-db always serves assets on `db_port + 1`, so a named port
    /// placed there — the customer's original silent-corruption failure — is
    /// rejected up front.
    #[test]
    fn assets_port_is_always_reserved() {
        let colliding = resources(32, 20000, &[("sitl_socket", PortSpec::Static(20001))]);
        let err = validate_port_plan(&colliding, 4).unwrap_err();
        assert!(err.to_string().contains("collision"), "got: {err}");

        // The customer's real plan leaves db_port + 1 free: valid, and the
        // assets port shows up as a planned (probed/reaped) port per worker.
        let valid = resources(32, 20000, &[("sitl_socket", PortSpec::Static(20002))]);
        let planned = validate_port_plan(&valid, 2).unwrap();
        assert!(
            planned
                .iter()
                .any(|planned| planned.name == "db_assets" && planned.port == 20001)
        );
        assert!(
            planned
                .iter()
                .any(|planned| planned.name == "db_assets" && planned.port == 20033)
        );
    }

    #[test]
    fn auto_ports_are_exempt_from_static_validation() {
        let resources = resources(
            100,
            2240,
            &[("ardupilot", PortSpec::Auto(crate::AutoTag::Auto))],
        );
        let planned = validate_port_plan(&resources, 96).unwrap();
        assert!(
            planned
                .iter()
                .all(|planned| planned.name == "db_port" || planned.name == "db_assets")
        );
    }

    #[test]
    fn allocate_port_holds_both_protocols() {
        let guard = allocate_port(IpAddr::V6(Ipv6Addr::UNSPECIFIED)).unwrap();
        assert!(guard.port > 0);
        // While the guard lives, the port cannot be taken.
        assert!(UdpSocket::bind((IpAddr::V6(Ipv6Addr::UNSPECIFIED), guard.port)).is_err());
    }

    #[test]
    fn allocate_port_pair_is_consecutive_and_guarded() {
        let ip = IpAddr::V6(Ipv6Addr::UNSPECIFIED);
        let (db, assets) = allocate_port_pair(ip).unwrap();
        assert_eq!(assets.port, db.port + 1);
        // Both ports are held on both protocols while the guards live.
        assert!(UdpSocket::bind((ip, db.port)).is_err());
        assert!(TcpListener::bind((ip, db.port)).is_err());
        assert!(UdpSocket::bind((ip, assets.port)).is_err());
        assert!(TcpListener::bind((ip, assets.port)).is_err());
        drop((db, assets));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn find_port_owners_names_this_process() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let owners = find_port_owners(&HashSet::from([port]));
        let owner = owners
            .iter()
            .find(|owner| owner.port == port && owner.protocol == "tcp")
            .expect("listener visible in /proc/net");
        assert_eq!(owner.pid, Some(std::process::id()));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn campaign_context_marker_is_required_for_reaping() {
        assert!(environ_contains_campaign_context(
            b"PATH=/bin\0ELODIN_MONTE_CARLO_CONTEXT=/tmp/run/context.json\0"
        ));
        assert!(!environ_contains_campaign_context(
            b"PATH=/bin\0ELODIN_MONTE_CARLO_CONTEXTUAL=yes\0"
        ));
        assert!(!environ_contains_campaign_context(b"PATH=/bin\0"));
    }
}
