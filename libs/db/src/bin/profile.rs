use std::{
    fmt,
    net::SocketAddr,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use clap::{Parser, ValueEnum};
use elodin_db::{Server, profile_stats};
use impeller2::{
    types::{ComponentId, LenPacket, PrimType},
    vtable::builder::{component, raw_field, schema, vtable},
};
use impeller2_stellar::Client;
use impeller2_wkt::{SubscribeLastUpdated, VTableMsg};
use stellarator::{net::TcpListener, sleep, spawn, struc_con::stellar};

#[derive(ValueEnum, Clone, Copy, Default)]
enum SendMode {
    #[default]
    Batch,
    PerComponent,
}

impl fmt::Display for SendMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SendMode::Batch => write!(f, "batch"),
            SendMode::PerComponent => write!(f, "per-component"),
        }
    }
}

#[derive(Parser)]
#[command(about = "elodin-db profiling tool — produces report.md")]
struct Args {
    #[arg(long, default_value_t = 400)]
    components: usize,
    #[arg(long, default_value_t = 250)]
    frequency: u32,
    #[arg(long, default_value_t = 10)]
    duration: u64,
    #[arg(long, default_value_t = 4)]
    clients: usize,
    #[arg(long, default_value = "false")]
    with_reader: bool,
    #[arg(long, value_enum)]
    scenario: Option<Scenario>,
    #[arg(long, value_enum)]
    mode: Option<SendMode>,
    #[arg(long, default_value = "report.md", help = "Output path for the report")]
    output: String,
    #[arg(long, help = "Run all built-in scenarios")]
    all: bool,
}

#[derive(ValueEnum, Clone)]
enum Scenario {
    Customer,
    CustomerBatch,
    HighFreq,
    HighFanout,
    Stress,
}

struct ScenarioConfig {
    name: &'static str,
    components: usize,
    frequency: u32,
    with_reader: bool,
    mode: SendMode,
}

const ALL_SCENARIOS: &[ScenarioConfig] = &[
    ScenarioConfig {
        name: "customer (per-component)",
        components: 400,
        frequency: 250,
        with_reader: true,
        mode: SendMode::PerComponent,
    },
    ScenarioConfig {
        name: "customer (batch)",
        components: 400,
        frequency: 250,
        with_reader: true,
        mode: SendMode::Batch,
    },
    ScenarioConfig {
        name: "high-freq",
        components: 50,
        frequency: 1000,
        with_reader: false,
        mode: SendMode::Batch,
    },
    ScenarioConfig {
        name: "high-fanout",
        components: 1000,
        frequency: 100,
        with_reader: false,
        mode: SendMode::Batch,
    },
    ScenarioConfig {
        name: "stress",
        components: 400,
        frequency: 1000,
        with_reader: true,
        mode: SendMode::Batch,
    },
];

#[stellarator::main]
async fn main() {
    init_tracing();
    let mut args = Args::parse();

    if args.all {
        let mut full_report = String::from("# elodin-db Profiling Report (all scenarios)\n\n");
        full_report.push_str("---\n\n");

        for sc in ALL_SCENARIOS {
            eprintln!("\n=== Running scenario: {} ===\n", sc.name);
            profile_stats::reset_all();

            run_benchmark(
                sc.components,
                sc.frequency,
                args.duration,
                args.clients,
                sc.with_reader,
                sc.mode,
            )
            .await;
            sleep(Duration::from_millis(200)).await;

            let snap = profile_stats::snapshot();
            let report = snap.generate_report(
                sc.name,
                sc.components,
                sc.frequency,
                args.duration,
                &sc.mode.to_string(),
            );
            full_report.push_str(&report);
            full_report.push_str("\n---\n\n");
        }

        std::fs::write(&args.output, &full_report).expect("failed to write report");
        eprintln!("\nReport written to {}", args.output);
        return;
    }

    if let Some(scenario) = &args.scenario {
        match scenario {
            Scenario::Customer => {
                args.components = 400;
                args.frequency = 250;
                args.with_reader = true;
                args.mode = args.mode.or(Some(SendMode::PerComponent));
            }
            Scenario::CustomerBatch => {
                args.components = 400;
                args.frequency = 250;
                args.with_reader = true;
                args.mode = Some(SendMode::Batch);
            }
            Scenario::HighFreq => {
                args.components = 50;
                args.frequency = 1000;
                args.with_reader = false;
            }
            Scenario::HighFanout => {
                args.components = 1000;
                args.frequency = 100;
                args.with_reader = false;
            }
            Scenario::Stress => {
                args.components = 400;
                args.frequency = 1000;
                args.with_reader = true;
            }
        }
    }

    let mode = args.mode.unwrap_or_default();
    let scenario_name = args
        .scenario
        .as_ref()
        .map(|s| match s {
            Scenario::Customer => "customer",
            Scenario::CustomerBatch => "customer-batch",
            Scenario::HighFreq => "high-freq",
            Scenario::HighFanout => "high-fanout",
            Scenario::Stress => "stress",
        })
        .unwrap_or("custom");

    profile_stats::reset_all();

    eprintln!(
        "Running benchmark: {} components, {} Hz, {}, {} s ...",
        args.components, args.frequency, mode, args.duration
    );

    run_benchmark(
        args.components,
        args.frequency,
        args.duration,
        args.clients,
        args.with_reader,
        mode,
    )
    .await;

    sleep(Duration::from_millis(200)).await;

    let snap = profile_stats::snapshot();
    let report = snap.generate_report(
        scenario_name,
        args.components,
        args.frequency,
        args.duration,
        &mode.to_string(),
    );

    std::fs::write(&args.output, &report).expect("failed to write report");
    eprintln!("\nReport written to {}", args.output);
}

fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    let filter = if std::env::var("RUST_LOG").is_ok() {
        EnvFilter::builder().from_env_lossy()
    } else {
        EnvFilter::builder().parse_lossy("warn")
    };
    let _ = tracing_subscriber::fmt::fmt()
        .with_writer(std::io::stderr)
        .with_target(false)
        .with_env_filter(filter)
        .try_init();
}

async fn run_benchmark(
    num_components: usize,
    frequency: u32,
    duration_secs: u64,
    num_clients: usize,
    with_reader: bool,
    mode: SendMode,
) {
    let temp_dir = std::env::temp_dir().join(format!("elodin_db_profile_{}", std::process::id()));
    if temp_dir.exists() {
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = Server::from_listener(listener, &temp_dir).unwrap();

    stellar(move || async move { server.run().await });
    sleep(Duration::from_millis(100)).await;

    if with_reader {
        let reader_addr = addr;
        stellar(move || run_reader(reader_addr));
    }

    let write_counter = Arc::new(AtomicU64::new(0));
    let target_duration = Duration::from_secs(duration_secs);
    let interval = Duration::from_secs_f64(1.0 / frequency as f64);

    let sampler_counter = write_counter.clone();
    let sampler_duration = duration_secs;
    let sampler = spawn(async move {
        for s in 0..sampler_duration {
            let before = sampler_counter.load(Ordering::Relaxed);
            sleep(Duration::from_secs(1)).await;
            let after = sampler_counter.load(Ordering::Relaxed);
            eprintln!("  t={}s  {} writes/s", s + 1, after - before);
        }
    });

    let mut handles = Vec::new();

    match mode {
        SendMode::Batch => {
            let components_per_client = distribute(num_components, num_clients);
            let mut vtable_base: u16 = 1;
            for &n in components_per_client.iter() {
                let base = vtable_base;
                vtable_base += n as u16;
                let counter = write_counter.clone();
                handles.push(spawn(run_writer_batch(
                    addr,
                    n,
                    base,
                    interval,
                    target_duration,
                    counter,
                )));
            }
        }
        SendMode::PerComponent => {
            let components_per_client = distribute(num_components, num_clients);
            let mut comp_base: u16 = 1;
            for &n in components_per_client.iter() {
                let base = comp_base;
                comp_base += n as u16;
                let counter = write_counter.clone();
                handles.push(spawn(run_writer_per_component(
                    addr,
                    base,
                    n,
                    interval,
                    target_duration,
                    counter,
                )));
            }
        }
    }

    for handle in handles {
        let _ = handle.await;
    }

    let _ = sampler.await;
    let _ = std::fs::remove_dir_all(&temp_dir);
}

async fn run_writer_batch(
    addr: SocketAddr,
    num_components: usize,
    vtable_base: u16,
    interval: Duration,
    target_duration: Duration,
    write_counter: Arc<AtomicU64>,
) {
    let mut client = Client::connect(addr).await.unwrap();

    let batched_vtable_id = vtable_base.to_le_bytes();
    let fields: Vec<_> = (0..num_components)
        .map(|i| {
            let comp_name = format!("bench_comp_{}", vtable_base as usize + i);
            let comp_id = ComponentId::new(&comp_name);
            raw_field(
                (i * 8) as u16,
                8,
                schema(PrimType::F64, &[], component(comp_id)),
            )
        })
        .collect();
    let vt = vtable(fields);
    client
        .send(&VTableMsg {
            id: batched_vtable_id,
            vtable: vt,
        })
        .await
        .0
        .unwrap();
    sleep(Duration::from_millis(50)).await;

    let start = Instant::now();
    let mut ticks: u64 = 0;

    while start.elapsed() < target_duration {
        let payload_size = num_components * 8;
        let mut pkt = LenPacket::table(batched_vtable_id, payload_size);
        for i in 0..num_components {
            let value = (ticks * num_components as u64 + i as u64) as f64;
            pkt.extend_aligned(&[value]);
        }
        client.send(pkt).await.0.unwrap();

        ticks += 1;
        write_counter.fetch_add(num_components as u64, Ordering::Relaxed);
        sleep(interval).await;
    }
}

async fn run_writer_per_component(
    addr: SocketAddr,
    comp_base: u16,
    num_components: usize,
    interval: Duration,
    target_duration: Duration,
    write_counter: Arc<AtomicU64>,
) {
    let mut client = Client::connect(addr).await.unwrap();

    let vtable_ids: Vec<[u8; 2]> = (0..num_components)
        .map(|i| (comp_base + i as u16).to_le_bytes())
        .collect();

    for (i, vtable_id) in vtable_ids.iter().enumerate() {
        let idx = comp_base + i as u16;
        let comp_name = format!("bench_comp_{}", idx);
        let comp_id = ComponentId::new(&comp_name);
        let vt = vtable(vec![raw_field(
            0,
            8,
            schema(PrimType::F64, &[], component(comp_id)),
        )]);
        client
            .send(&VTableMsg {
                id: *vtable_id,
                vtable: vt,
            })
            .await
            .0
            .unwrap();
    }
    sleep(Duration::from_millis(50)).await;

    let start = Instant::now();
    let mut ticks: u64 = 0;

    while start.elapsed() < target_duration {
        for vtable_id in &vtable_ids {
            let mut pkt = LenPacket::table(*vtable_id, 8);
            pkt.extend_aligned(&[ticks as f64]);
            client.send(pkt).await.0.unwrap();
        }

        ticks += 1;
        write_counter.fetch_add(num_components as u64, Ordering::Relaxed);
        sleep(interval).await;
    }
}

async fn run_reader(addr: SocketAddr) {
    let mut client = Client::connect(addr).await.unwrap();
    let mut stream = client.stream(&SubscribeLastUpdated).await.unwrap();
    loop {
        if stream.next().await.is_err() {
            break;
        }
    }
}

fn distribute(total: usize, buckets: usize) -> Vec<usize> {
    let base = total / buckets;
    let remainder = total % buckets;
    (0..buckets)
        .map(|i| if i < remainder { base + 1 } else { base })
        .collect()
}
