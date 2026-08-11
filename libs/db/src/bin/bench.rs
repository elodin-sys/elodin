use std::{
    fmt,
    net::SocketAddr,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

#[cfg(feature = "grpc")]
use std::collections::VecDeque;

use clap::{Parser, ValueEnum};
use elodin_db::Server;
use impeller2::{
    types::{ComponentId, LenPacket, PrimType},
    vtable::builder::{component, raw_field, schema, vtable},
};
use impeller2_stellar::Client;
use impeller2_wkt::{SubscribeLastUpdated, VTableMsg};
use stellarator::{net::TcpListener, sleep, spawn, struc_con::stellar};

#[cfg(feature = "grpc")]
use {
    elodin_db::grpc::v1::{
        AckPolicy, ComponentSchema, ComponentValue, IngestRequest, MessageSchema, Row, RowEncoding,
        SchemaSet, SessionOpen, TelemetryBatch, TypedValues, component_value, ingest_request,
        ingest_response, ingest_service_client::IngestServiceClient, row,
    },
    prost::Message,
    sha2::{Digest, Sha256},
    stellarator::struc_con::{Joinable, tokio as tokio_thread},
    tokio::sync::mpsc,
    tokio_stream::wrappers::ReceiverStream,
};

#[derive(ValueEnum, Clone, Copy, Default)]
enum SendMode {
    #[default]
    Batch,
    PerComponent,
    #[cfg(feature = "grpc")]
    GrpcPacked,
    #[cfg(feature = "grpc")]
    GrpcTyped,
}

impl SendMode {
    fn is_grpc(self) -> bool {
        match self {
            Self::Batch | Self::PerComponent => false,
            #[cfg(feature = "grpc")]
            Self::GrpcPacked | Self::GrpcTyped => true,
        }
    }
}

impl fmt::Display for SendMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SendMode::Batch => write!(f, "batch"),
            SendMode::PerComponent => write!(f, "per-component"),
            #[cfg(feature = "grpc")]
            SendMode::GrpcPacked => write!(f, "grpc-packed"),
            #[cfg(feature = "grpc")]
            SendMode::GrpcTyped => write!(f, "grpc-typed"),
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
    #[arg(
        long,
        help = "Use an existing Impeller server instead of an embedded DB"
    )]
    db_addr: Option<SocketAddr>,
    #[cfg(feature = "grpc")]
    #[arg(long, help = "Use an existing gRPC server; requires --db-addr")]
    grpc_addr: Option<SocketAddr>,
}

#[derive(ValueEnum, Clone)]
enum Scenario {
    Customer,
    HighFreq,
    HighFanout,
    Stress,
}

#[derive(Clone, Copy, Default)]
struct Endpoints {
    db: Option<SocketAddr>,
    #[cfg(feature = "grpc")]
    grpc: Option<SocketAddr>,
}

struct BenchmarkConfig<'a> {
    components: usize,
    frequency: u32,
    duration_secs: u64,
    clients: usize,
    with_reader: bool,
    mode: SendMode,
    scenario_name: &'a str,
    endpoints: Endpoints,
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
    ack_latency_p50_us: Option<u64>,
    ack_latency_p95_us: Option<u64>,
    ack_latency_p99_us: Option<u64>,
    ack_latency_max_us: Option<u64>,
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
        if let Some(p50) = self.ack_latency_p50_us {
            eprintln!("║ ack latency p50:   {:<23} µs ║", p50);
            eprintln!(
                "║ ack latency p95:   {:<23} µs ║",
                self.ack_latency_p95_us.unwrap()
            );
            eprintln!(
                "║ ack latency p99:   {:<23} µs ║",
                self.ack_latency_p99_us.unwrap()
            );
            eprintln!(
                "║ ack latency max:   {:<23} µs ║",
                self.ack_latency_max_us.unwrap()
            );
        } else {
            eprintln!("║ ack latency:       {:<23}    ║", "n/a");
        }
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
        let json_number = |value: Option<u64>| {
            value
                .map(|value| value.to_string())
                .unwrap_or_else(|| "null".to_string())
        };
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
                "\"ack_latency_p50_us\":{},",
                "\"ack_latency_p95_us\":{},",
                "\"ack_latency_p99_us\":{},",
                "\"ack_latency_max_us\":{},",
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
            json_number(self.ack_latency_p50_us),
            json_number(self.ack_latency_p95_us),
            json_number(self.ack_latency_p99_us),
            json_number(self.ack_latency_max_us),
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
    let mode = args.mode.unwrap_or_default();
    if args.clients == 0 || args.components == 0 || args.frequency == 0 || args.duration == 0 {
        eprintln!("components, frequency, duration, and clients must be positive");
        std::process::exit(2);
    }
    if mode.is_grpc() && args.clients != 1 {
        eprintln!("gRPC modes use one bidi stream; overriding --clients to 1");
        args.clients = 1;
    }
    #[cfg(feature = "grpc")]
    if mode.is_grpc() && args.db_addr.is_some() != args.grpc_addr.is_some() {
        eprintln!("external gRPC mode requires both --db-addr and --grpc-addr");
        std::process::exit(2);
    }
    #[cfg(feature = "grpc")]
    let endpoints = Endpoints {
        db: args.db_addr,
        grpc: args.grpc_addr,
    };
    #[cfg(not(feature = "grpc"))]
    let endpoints = Endpoints { db: args.db_addr };

    let result = run_benchmark(BenchmarkConfig {
        components: args.components,
        frequency: args.frequency,
        duration_secs: args.duration,
        clients: args.clients,
        with_reader: args.with_reader,
        mode,
        scenario_name: &scenario_name,
        endpoints,
    })
    .await;

    if args.json {
        result.print_json();
    } else {
        result.print_human();
    }
}

async fn run_benchmark(config: BenchmarkConfig<'_>) -> BenchResult {
    let BenchmarkConfig {
        components: num_components,
        frequency,
        duration_secs,
        clients: num_clients,
        with_reader,
        mode,
        scenario_name,
        endpoints,
    } = config;
    let temp_dir = std::env::temp_dir().join(format!("elodin_db_bench_{}", std::process::id()));
    let embedded = endpoints.db.is_none();
    let (addr, _embedded_db) = if let Some(addr) = endpoints.db {
        (addr, None)
    } else {
        if temp_dir.exists() {
            let _ = std::fs::remove_dir_all(&temp_dir);
        }
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = Server::from_listener(listener, &temp_dir).unwrap();
        let db = server.db.clone();
        stellar(move || async move { server.run().await });
        (addr, Some(db))
    };

    #[cfg(feature = "grpc")]
    let grpc_addr = if mode.is_grpc() {
        if let Some(addr) = endpoints.grpc {
            Some(addr)
        } else {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            drop(listener);
            let db = _embedded_db
                .clone()
                .expect("embedded gRPC benchmark requires an embedded DB");
            tokio_thread(move |_| async move { elodin_db::grpc::serve(addr, db).await.unwrap() });
            Some(addr)
        }
    } else {
        None
    };

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
    #[cfg(feature = "grpc")]
    let mut grpc_result = None;

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
            let components_per_client = distribute(num_components, num_clients);
            let mut comp_base: u16 = 1;
            for &n in components_per_client.iter() {
                let base = comp_base;
                comp_base += n as u16;
                let counter = write_counter.clone();
                let lat = latencies.clone();
                handles.push(spawn(run_writer_per_component(
                    addr,
                    base,
                    n,
                    interval,
                    target_duration,
                    counter,
                    lat,
                )));
            }
        }
        #[cfg(feature = "grpc")]
        SendMode::GrpcPacked | SendMode::GrpcTyped => {
            let encoding = match mode {
                SendMode::GrpcPacked => RowEncoding::Packed,
                SendMode::GrpcTyped => RowEncoding::Typed,
                _ => unreachable!(),
            };
            let counter = write_counter.clone();
            let writer = tokio_thread(move |_| {
                run_writer_grpc(
                    grpc_addr.unwrap(),
                    num_components,
                    frequency,
                    interval,
                    target_duration,
                    encoding,
                    counter,
                )
            });
            grpc_result = Some(writer.join().await.unwrap().unwrap());
        }
    }

    for handle in handles {
        let _ = handle.await;
    }

    let elapsed = start.elapsed();
    let per_second_throughput = sampler.await.unwrap_or_default();
    let total_writes = count_total_samples(addr, num_components, mode, num_clients).await;

    let impeller_send_samples = latencies.lock().unwrap().clone();
    #[cfg(feature = "grpc")]
    let (send_samples, ack_samples) = match grpc_result {
        Some(result) => (result.send_latencies, Some(result.ack_latencies)),
        None => (impeller_send_samples, None),
    };
    #[cfg(not(feature = "grpc"))]
    let (send_samples, ack_samples): (Vec<u64>, Option<Vec<u64>>) = (impeller_send_samples, None);
    let send_latency = latency_summary(send_samples);
    let ack_latency = ack_samples.map(latency_summary);

    let target_writes_per_sec = num_components as f64 * frequency as f64;
    let throughput = total_writes as f64 / elapsed.as_secs_f64();
    let data_bytes = total_writes * 8;
    let data_volume_mb = data_bytes as f64 / (1024.0 * 1024.0);

    if embedded {
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

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
        clients: num_clients,
        per_second_throughput,
        send_latency_p50_us: send_latency.p50,
        send_latency_p95_us: send_latency.p95,
        send_latency_p99_us: send_latency.p99,
        send_latency_max_us: send_latency.max,
        ack_latency_p50_us: ack_latency.map(|latency| latency.p50),
        ack_latency_p95_us: ack_latency.map(|latency| latency.p95),
        ack_latency_p99_us: ack_latency.map(|latency| latency.p99),
        ack_latency_max_us: ack_latency.map(|latency| latency.max),
    }
}

#[derive(Clone, Copy)]
struct LatencySummary {
    p50: u64,
    p95: u64,
    p99: u64,
    max: u64,
}

fn latency_summary(mut samples: Vec<u64>) -> LatencySummary {
    if samples.is_empty() {
        return LatencySummary {
            p50: 0,
            p95: 0,
            p99: 0,
            max: 0,
        };
    }
    samples.sort_unstable();
    let len = samples.len();
    LatencySummary {
        p50: samples[len * 50 / 100],
        p95: samples[len * 95 / 100],
        p99: samples[len.saturating_sub(1).min(len * 99 / 100)],
        max: samples[len - 1],
    }
}

#[cfg(feature = "grpc")]
struct GrpcWriterResult {
    send_latencies: Vec<u64>,
    ack_latencies: Vec<u64>,
}

#[cfg(feature = "grpc")]
async fn run_writer_grpc(
    addr: SocketAddr,
    num_components: usize,
    frequency: u32,
    interval: Duration,
    target_duration: Duration,
    encoding: RowEncoding,
    write_counter: Arc<AtomicU64>,
) -> Result<GrpcWriterResult, String> {
    let schema = grpc_schema(num_components, encoding);
    let schema_fingerprint = Sha256::digest(schema.encode_to_vec()).to_vec();
    let open = IngestRequest {
        req: Some(ingest_request::Req::Open(SessionOpen {
            client_name: format!("elodin-db-bench-{encoding:?}"),
            schema_fingerprint,
            schema: Some(schema),
            ack_policy: Some(AckPolicy {
                max_unacked_rows: 256,
                max_ack_delay_ms: 20,
            }),
            client_instance_id: format!(
                "{}-{encoding:?}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_err(|error| error.to_string())?
                    .as_nanos()
            )
            .into_bytes(),
        })),
    };

    let mut client = IngestServiceClient::connect(format!("http://{addr}"))
        .await
        .map_err(|error| error.to_string())?;
    let (tx, rx) = mpsc::channel(64);
    tx.send(open).await.map_err(|error| error.to_string())?;
    let mut responses = client
        .ingest(ReceiverStream::new(rx))
        .await
        .map_err(|error| error.to_string())?
        .into_inner();
    let first = responses
        .message()
        .await
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "gRPC stream closed before SessionAccept".to_string())?;
    let accept = match first.resp {
        Some(ingest_response::Resp::Accept(accept)) => accept,
        Some(ingest_response::Resp::Reject(reject)) => {
            return Err(format!("session rejected: {}", reject.detail));
        }
        _ => return Err("expected SessionAccept".to_string()),
    };
    let message_handle = *accept
        .message_handles
        .get("BenchMessage")
        .ok_or_else(|| "SessionAccept omitted BenchMessage".to_string())?;

    let pending = Arc::new(std::sync::Mutex::new(VecDeque::<(u64, Instant)>::new()));
    let ack_latencies = Arc::new(std::sync::Mutex::new(Vec::new()));
    let row_errors = Arc::new(AtomicU64::new(0));
    let through_seq = Arc::new(AtomicU64::new(accept.resume_from_seq));
    let reader_pending = pending.clone();
    let reader_latencies = ack_latencies.clone();
    let reader_errors = row_errors.clone();
    let reader_through = through_seq.clone();
    let reader = tokio::spawn(async move {
        while let Some(response) = responses
            .message()
            .await
            .map_err(|error| error.to_string())?
        {
            match response.resp {
                Some(ingest_response::Resp::Ack(ack)) => {
                    reader_through.store(ack.through_seq, Ordering::Relaxed);
                    let now = Instant::now();
                    let mut pending = reader_pending.lock().unwrap();
                    let mut latencies = reader_latencies.lock().unwrap();
                    while pending
                        .front()
                        .is_some_and(|(seq, _)| *seq <= ack.through_seq)
                    {
                        let (_, sent_at) = pending.pop_front().unwrap();
                        latencies.push(now.duration_since(sent_at).as_micros() as u64);
                    }
                }
                Some(ingest_response::Resp::Error(error)) => {
                    reader_errors.fetch_add(1, Ordering::Relaxed);
                    eprintln!(
                        "gRPC RowError seq={} component={} detail={}",
                        error.seq, error.component, error.detail
                    );
                }
                Some(ingest_response::Resp::Reject(reject)) => {
                    return Err(format!("session rejected after accept: {}", reject.detail));
                }
                Some(ingest_response::Resp::Accept(_)) | None => {
                    return Err("unexpected gRPC ingest response".to_string());
                }
            }
        }
        Ok::<_, String>(())
    });

    let start = Instant::now();
    let mut deadline = Instant::now();
    let mut seq = accept.resume_from_seq.saturating_add(1);
    let mut tick = 0u64;
    let mut send_latencies = Vec::new();
    while start.elapsed() < target_duration {
        let tick_started = Instant::now();
        let payload = match encoding {
            RowEncoding::Packed => row::Payload::Packed(grpc_packed_values(tick, num_components)),
            RowEncoding::Typed => row::Payload::Typed(grpc_typed_values(tick, num_components)),
            RowEncoding::Unspecified => return Err("unspecified gRPC encoding".to_string()),
        };
        let request = IngestRequest {
            req: Some(ingest_request::Req::Batch(TelemetryBatch {
                first_seq: seq,
                rows: vec![Row {
                    message_handle,
                    time_monotonic_ns: 1_000_000_000
                        + (tick as i64 * 1_000_000_000 / i64::from(frequency)),
                    payload: Some(payload),
                }],
            })),
        };
        let ack_started = Instant::now();
        pending.lock().unwrap().push_back((seq, ack_started));
        tx.send(request).await.map_err(|error| error.to_string())?;
        if tick.is_multiple_of(10) {
            send_latencies.push(tick_started.elapsed().as_micros() as u64);
        }
        write_counter.fetch_add(num_components as u64, Ordering::Relaxed);
        seq += 1;
        tick += 1;
        deadline += interval;
        tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await;
    }

    drop(tx);
    reader
        .await
        .map_err(|error| error.to_string())?
        .map_err(|error| error.to_string())?;
    if row_errors.load(Ordering::Relaxed) != 0 {
        return Err(format!(
            "{} gRPC rows were rejected",
            row_errors.load(Ordering::Relaxed)
        ));
    }
    let expected_through = seq.saturating_sub(1);
    if through_seq.load(Ordering::Relaxed) < expected_through || !pending.lock().unwrap().is_empty()
    {
        return Err(format!(
            "gRPC stream closed before acking sequence {expected_through}"
        ));
    }

    let ack_latencies = ack_latencies.lock().unwrap().clone();
    Ok(GrpcWriterResult {
        send_latencies,
        ack_latencies,
    })
}

#[cfg(feature = "grpc")]
fn grpc_schema(num_components: usize, encoding: RowEncoding) -> SchemaSet {
    let components = (0..num_components)
        .map(|index| ComponentSchema {
            name: format!("bench_comp_{}", index + 1),
            prim_type: elodin_db::grpc::v1::PrimType::F64 as i32,
            dims: Vec::new(),
            element_names: Vec::new(),
            packed_offset: if encoding == RowEncoding::Packed {
                (index * std::mem::size_of::<f64>()) as u32
            } else {
                0
            },
            timestamp_source: false,
        })
        .collect();
    SchemaSet {
        messages: vec![MessageSchema {
            name: "BenchMessage".to_string(),
            encoding: encoding as i32,
            packed_size: if encoding == RowEncoding::Packed {
                (num_components * std::mem::size_of::<f64>()) as u32
            } else {
                0
            },
            components,
        }],
    }
}

#[cfg(feature = "grpc")]
fn grpc_packed_values(tick: u64, num_components: usize) -> Vec<u8> {
    let mut values = Vec::with_capacity(num_components * std::mem::size_of::<f64>());
    for index in 0..num_components {
        values.extend_from_slice(
            &((tick * num_components as u64 + index as u64) as f64).to_le_bytes(),
        );
    }
    values
}

#[cfg(feature = "grpc")]
fn grpc_typed_values(tick: u64, num_components: usize) -> TypedValues {
    TypedValues {
        values: (0..num_components)
            .map(|index| ComponentValue {
                component_index: index as u32,
                value: Some(component_value::Value::F64(
                    (tick * num_components as u64 + index as u64) as f64,
                )),
            })
            .collect(),
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

/// Sends one packet per component per tick over a shared connection.
/// Each component gets its own 1-field VTable, so the server still pays
/// per-packet overhead (protocol parse, write-lock, vtable lookup, mmap)
/// for every component -- but we avoid opening N TCP connections which
/// would exhaust io_uring memory on resource-constrained CI.
async fn run_writer_per_component(
    addr: SocketAddr,
    comp_base: u16,
    num_components: usize,
    interval: Duration,
    target_duration: Duration,
    write_counter: Arc<AtomicU64>,
    latencies: Arc<std::sync::Mutex<Vec<u64>>>,
) -> u64 {
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
    let mut local_latencies = Vec::new();
    let sample_every = 10u64;

    while start.elapsed() < target_duration {
        let tick_start = Instant::now();

        for vtable_id in &vtable_ids {
            let mut pkt = LenPacket::table(*vtable_id, 8);
            pkt.extend_aligned(&[ticks as f64]);
            client.send(pkt).await.0.unwrap();
        }

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

async fn run_reader(addr: SocketAddr) {
    let mut client = Client::connect(addr).await.unwrap();
    let mut stream = client.stream(&SubscribeLastUpdated).await.unwrap();
    loop {
        if stream.next().await.is_err() {
            break;
        }
    }
}

async fn count_total_samples(
    addr: SocketAddr,
    num_components: usize,
    mode: SendMode,
    num_clients: usize,
) -> u64 {
    sleep(Duration::from_millis(200)).await;

    let mut client = Client::connect(addr).await.unwrap();
    let mut total: u64 = 0;

    use impeller2::types::Timestamp;
    use impeller2_wkt::GetTimeSeries;

    // Build the (vtable_id, comp_name) pairs matching writer registration.
    let pairs: Vec<([u8; 2], String)> = match mode {
        SendMode::PerComponent => (0..num_components)
            .map(|i| {
                let idx = (i as u16) + 1;
                (idx.to_le_bytes(), format!("bench_comp_{}", idx))
            })
            .collect(),
        SendMode::Batch => {
            let dist = distribute(num_components, num_clients);
            let mut out = Vec::with_capacity(num_components);
            let mut vtable_base: u16 = 1;
            for &n in &dist {
                let vt_id = vtable_base.to_le_bytes();
                for offset in 0..n {
                    let name = format!("bench_comp_{}", vtable_base as usize + offset);
                    out.push((vt_id, name));
                }
                vtable_base += n as u16;
            }
            out
        }
        #[cfg(feature = "grpc")]
        SendMode::GrpcPacked | SendMode::GrpcTyped => (0..num_components)
            .map(|index| {
                (
                    ((index + 1) as u16).to_le_bytes(),
                    format!("bench_comp_{}", index + 1),
                )
            })
            .collect(),
    };

    for (vtable_id, comp_name) in &pairs {
        let comp_id = ComponentId::new(comp_name);
        let query = GetTimeSeries {
            id: *vtable_id,
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
