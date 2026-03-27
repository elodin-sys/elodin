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
use elodin_db::Server;
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
#[command(about = "Elodin-DB throughput benchmark")]
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
    #[arg(long, default_value = "false")]
    json: bool,
    #[arg(long, value_enum)]
    scenario: Option<Scenario>,
    #[arg(long, value_enum)]
    mode: Option<SendMode>,
}

#[derive(ValueEnum, Clone)]
enum Scenario {
    Customer,
    HighFreq,
    HighFanout,
    Stress,
}

struct BenchResult {
    scenario: String,
    mode: SendMode,
    components: usize,
    frequency: u32,
    duration_secs: f64,
    total_writes: u64,
    throughput_writes_per_sec: f64,
    target_writes_per_sec: f64,
    achieved_ratio: f64,
    data_volume_mb: f64,
    data_rate_mb_per_sec: f64,
    effective_freq_per_component: f64,
    with_reader: bool,
    clients: usize,
    per_second_throughput: Vec<u64>,
    send_latency_p50_us: u64,
    send_latency_p95_us: u64,
    send_latency_p99_us: u64,
    send_latency_max_us: u64,
}

impl BenchResult {
    fn print_human(&self) {
        eprintln!("╔══════════════════════════════════════════════╗");
        eprintln!("║          elodin-db benchmark results         ║");
        eprintln!("╠══════════════════════════════════════════════╣");
        eprintln!("║ scenario:      {:<30}║", self.scenario);
        eprintln!("║ mode:          {:<30}║", self.mode);
        eprintln!("║ components:    {:<30}║", self.components);
        eprintln!("║ frequency:     {:<27} Hz ║", self.frequency);
        eprintln!("║ clients:       {:<30}║", self.clients);
        eprintln!("║ with_reader:   {:<30}║", self.with_reader);
        eprintln!("╠══════════════════════════════════════════════╣");
        eprintln!("║ duration:      {:<28.2}s ║", self.duration_secs);
        eprintln!("║ total_writes:  {:<30}║", self.total_writes);
        eprintln!(
            "║ throughput:    {:<23.0} writes/s ║",
            self.throughput_writes_per_sec
        );
        eprintln!(
            "║ target:        {:<23.0} writes/s ║",
            self.target_writes_per_sec
        );
        eprintln!("║ achieved:      {:<28.1}% ║", self.achieved_ratio * 100.0);
        eprintln!("╠══════════════════════════════════════════════╣");
        eprintln!("║ data volume:   {:<27.2} MB ║", self.data_volume_mb);
        eprintln!(
            "║ data rate:     {:<24.2} MB/s ║",
            self.data_rate_mb_per_sec
        );
        eprintln!(
            "║ effective freq:{:<27.1} Hz ║",
            self.effective_freq_per_component
        );
        eprintln!("╠══════════════════════════════════════════════╣");
        eprintln!("║ send latency p50:  {:<23} µs ║", self.send_latency_p50_us);
        eprintln!("║ send latency p95:  {:<23} µs ║", self.send_latency_p95_us);
        eprintln!("║ send latency p99:  {:<23} µs ║", self.send_latency_p99_us);
        eprintln!("║ send latency max:  {:<23} µs ║", self.send_latency_max_us);
        eprintln!("╠══════════════════════════════════════════════╣");
        eprintln!("║ per-second throughput (writes/s):            ║");
        for (i, &t) in self.per_second_throughput.iter().enumerate() {
            eprintln!("║   t={:<3}s  {:<35}║", i + 1, t);
        }
        eprintln!("╚══════════════════════════════════════════════╝");
    }

    fn print_json(&self) {
        let per_sec: Vec<String> = self
            .per_second_throughput
            .iter()
            .map(|v| v.to_string())
            .collect();
        println!(
            concat!(
                "{{",
                "\"scenario\":\"{}\",",
                "\"mode\":\"{}\",",
                "\"components\":{},",
                "\"frequency\":{},",
                "\"clients\":{},",
                "\"with_reader\":{},",
                "\"duration_secs\":{:.3},",
                "\"total_writes\":{},",
                "\"throughput_writes_per_sec\":{:.1},",
                "\"target_writes_per_sec\":{:.1},",
                "\"achieved_ratio\":{:.4},",
                "\"data_volume_mb\":{:.2},",
                "\"data_rate_mb_per_sec\":{:.2},",
                "\"effective_freq_per_component\":{:.1},",
                "\"send_latency_p50_us\":{},",
                "\"send_latency_p95_us\":{},",
                "\"send_latency_p99_us\":{},",
                "\"send_latency_max_us\":{},",
                "\"per_second_throughput\":[{}]",
                "}}"
            ),
            self.scenario,
            self.mode,
            self.components,
            self.frequency,
            self.clients,
            self.with_reader,
            self.duration_secs,
            self.total_writes,
            self.throughput_writes_per_sec,
            self.target_writes_per_sec,
            self.achieved_ratio,
            self.data_volume_mb,
            self.data_rate_mb_per_sec,
            self.effective_freq_per_component,
            self.send_latency_p50_us,
            self.send_latency_p95_us,
            self.send_latency_p99_us,
            self.send_latency_max_us,
            per_sec.join(","),
        );
    }
}

fn init_tracing() {
    use tracing_subscriber::EnvFilter;

    #[cfg(feature = "tracy")]
    {
        use tracing_subscriber::prelude::*;
        // fmt only gets warn+ to avoid flooding stderr; Tracy gets trace-level spans
        let fmt_filter = EnvFilter::builder().parse_lossy("warn");
        let tracy_filter = EnvFilter::builder().parse_lossy("elodin_db=trace");
        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_writer(std::io::stderr)
            .with_target(false)
            .with_filter(fmt_filter);
        let tracy_layer = tracing_tracy::TracyLayer::default().with_filter(tracy_filter);
        let _ = tracing_subscriber::registry()
            .with(fmt_layer)
            .with(tracy_layer)
            .try_init();
    }

    #[cfg(not(feature = "tracy"))]
    {
        let filter = if std::env::var("RUST_LOG").is_ok() {
            EnvFilter::builder().from_env_lossy()
        } else {
            EnvFilter::builder().parse_lossy("elodin_db=info")
        };
        let _ = tracing_subscriber::fmt::fmt()
            .with_writer(std::io::stderr)
            .with_target(false)
            .with_env_filter(filter)
            .try_init();
    }
}

#[stellarator::main]
async fn main() {
    init_tracing();
    let mut args = Args::parse();

    if let Some(scenario) = &args.scenario {
        match scenario {
            Scenario::Customer => {
                args.components = 400;
                args.frequency = 250;
                args.with_reader = true;
                args.mode = args.mode.or(Some(SendMode::PerComponent));
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

    let scenario_name = args
        .scenario
        .as_ref()
        .map(|s| match s {
            Scenario::Customer => "customer",
            Scenario::HighFreq => "high-freq",
            Scenario::HighFanout => "high-fanout",
            Scenario::Stress => "stress",
        })
        .unwrap_or("custom")
        .to_string();

    let result = run_benchmark(
        args.components,
        args.frequency,
        args.duration,
        args.clients,
        args.with_reader,
        args.mode.unwrap_or_default(),
        &scenario_name,
    )
    .await;

    if args.json {
        result.print_json();
    } else {
        result.print_human();
    }
}

async fn run_benchmark(
    num_components: usize,
    frequency: u32,
    duration_secs: u64,
    num_clients: usize,
    with_reader: bool,
    mode: SendMode,
    scenario_name: &str,
) -> BenchResult {
    let temp_dir = std::env::temp_dir().join(format!("elodin_db_bench_{}", std::process::id()));
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
    let latencies = Arc::new(std::sync::Mutex::new(Vec::<u64>::new()));

    let start = Instant::now();
    let target_duration = Duration::from_secs(duration_secs);
    let interval = Duration::from_secs_f64(1.0 / frequency as f64);

    let sampler_counter = write_counter.clone();
    let sampler_duration = duration_secs;
    let sampler = spawn(async move {
        let mut per_second = Vec::new();
        for _ in 0..sampler_duration {
            let before = sampler_counter.load(Ordering::Relaxed);
            sleep(Duration::from_secs(1)).await;
            let after = sampler_counter.load(Ordering::Relaxed);
            per_second.push(after - before);
        }
        per_second
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
                let lat = latencies.clone();
                handles.push(spawn(run_writer_batch(
                    addr,
                    n,
                    base,
                    interval,
                    target_duration,
                    counter,
                    lat,
                )));
            }
        }
        SendMode::PerComponent => {
            for i in 0..num_components {
                let counter = write_counter.clone();
                let lat = latencies.clone();
                let comp_idx = i as u16 + 1;
                handles.push(spawn(run_writer_per_component(
                    addr,
                    comp_idx,
                    interval,
                    target_duration,
                    counter,
                    lat,
                )));
            }
        }
    }

    for handle in handles {
        let _ = handle.await;
    }

    let elapsed = start.elapsed();
    let per_second_throughput = sampler.await.unwrap_or_default();
    let total_writes = count_total_samples(addr, num_components).await;

    let mut lat_samples = latencies.lock().unwrap().clone();
    lat_samples.sort_unstable();
    let (p50, p95, p99, max) = if lat_samples.is_empty() {
        (0, 0, 0, 0)
    } else {
        let len = lat_samples.len();
        (
            lat_samples[len * 50 / 100],
            lat_samples[len * 95 / 100],
            lat_samples[len.saturating_sub(1).min(len * 99 / 100)],
            lat_samples[len - 1],
        )
    };

    let target_writes_per_sec = num_components as f64 * frequency as f64;
    let throughput = total_writes as f64 / elapsed.as_secs_f64();
    let data_bytes = total_writes * 8;
    let data_volume_mb = data_bytes as f64 / (1024.0 * 1024.0);

    let _ = std::fs::remove_dir_all(&temp_dir);

    BenchResult {
        scenario: scenario_name.to_string(),
        mode,
        components: num_components,
        frequency,
        duration_secs: elapsed.as_secs_f64(),
        total_writes,
        throughput_writes_per_sec: throughput,
        target_writes_per_sec,
        achieved_ratio: throughput / target_writes_per_sec,
        data_volume_mb,
        data_rate_mb_per_sec: data_volume_mb / elapsed.as_secs_f64(),
        effective_freq_per_component: throughput / num_components as f64,
        with_reader,
        clients: match mode {
            SendMode::Batch => num_clients,
            SendMode::PerComponent => num_components,
        },
        per_second_throughput,
        send_latency_p50_us: p50,
        send_latency_p95_us: p95,
        send_latency_p99_us: p99,
        send_latency_max_us: max,
    }
}

async fn run_writer_batch(
    addr: SocketAddr,
    num_components: usize,
    vtable_base: u16,
    interval: Duration,
    target_duration: Duration,
    write_counter: Arc<AtomicU64>,
    latencies: Arc<std::sync::Mutex<Vec<u64>>>,
) -> u64 {
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
    let mut local_latencies = Vec::new();
    let sample_every = 10u64;

    while start.elapsed() < target_duration {
        let tick_start = Instant::now();

        let payload_size = num_components * 8;
        let mut pkt = LenPacket::table(batched_vtable_id, payload_size);
        for i in 0..num_components {
            let value = (ticks * num_components as u64 + i as u64) as f64;
            pkt.extend_aligned(&[value]);
        }
        client.send(pkt).await.0.unwrap();

        ticks += 1;
        write_counter.fetch_add(num_components as u64, Ordering::Relaxed);

        if ticks.is_multiple_of(sample_every) {
            local_latencies.push(tick_start.elapsed().as_micros() as u64);
        }

        sleep(interval).await;
    }

    if let Ok(mut global) = latencies.lock() {
        global.extend_from_slice(&local_latencies);
    }

    ticks * num_components as u64
}

/// Each component gets its own TCP connection and VTable (1 field each).
/// This mirrors the pattern used in `db.hpp` where each writer handles a
/// single component independently.
async fn run_writer_per_component(
    addr: SocketAddr,
    comp_idx: u16,
    interval: Duration,
    target_duration: Duration,
    write_counter: Arc<AtomicU64>,
    latencies: Arc<std::sync::Mutex<Vec<u64>>>,
) -> u64 {
    let mut client = Client::connect(addr).await.unwrap();

    let vtable_id = comp_idx.to_le_bytes();
    let comp_name = format!("bench_comp_{}", comp_idx);
    let comp_id = ComponentId::new(&comp_name);
    let vt = vtable(vec![raw_field(
        0,
        8,
        schema(PrimType::F64, &[], component(comp_id)),
    )]);
    client
        .send(&VTableMsg {
            id: vtable_id,
            vtable: vt,
        })
        .await
        .0
        .unwrap();

    sleep(Duration::from_millis(50)).await;

    let start = Instant::now();
    let mut ticks: u64 = 0;
    let mut local_latencies = Vec::new();
    let sample_every = 10u64;

    while start.elapsed() < target_duration {
        let tick_start = Instant::now();

        let mut pkt = LenPacket::table(vtable_id, 8);
        pkt.extend_aligned(&[ticks as f64]);
        client.send(pkt).await.0.unwrap();

        ticks += 1;
        write_counter.fetch_add(1, Ordering::Relaxed);

        if ticks.is_multiple_of(sample_every) {
            local_latencies.push(tick_start.elapsed().as_micros() as u64);
        }

        sleep(interval).await;
    }

    if let Ok(mut global) = latencies.lock() {
        global.extend_from_slice(&local_latencies);
    }

    ticks
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

async fn count_total_samples(addr: SocketAddr, num_components: usize) -> u64 {
    sleep(Duration::from_millis(200)).await;

    let mut client = Client::connect(addr).await.unwrap();
    let mut total: u64 = 0;

    for i in 0..num_components {
        let comp_name = format!("bench_comp_{}", i + 1);
        let comp_id = ComponentId::new(&comp_name);

        use impeller2::types::Timestamp;
        use impeller2_wkt::GetTimeSeries;

        let vtable_id = ((i + 1) as u16).to_le_bytes();
        let query = GetTimeSeries {
            id: vtable_id,
            range: Timestamp(0)..Timestamp(i64::MAX),
            component_id: comp_id,
            limit: None,
        };

        if let Ok(ts) = client.request(&query).await
            && let Ok(timestamps) = ts.timestamps()
        {
            total += timestamps.len() as u64;
        }
    }

    total
}

fn distribute(total: usize, buckets: usize) -> Vec<usize> {
    let base = total / buckets;
    let remainder = total % buckets;
    (0..buckets)
        .map(|i| if i < remainder { base + 1 } else { base })
        .collect()
}
