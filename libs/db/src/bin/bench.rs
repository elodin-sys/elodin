use std::{
    net::SocketAddr,
    sync::Arc,
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
use stellarator::{net::TcpListener, sleep, spawn, struc_con::stellar, sync::WaitQueue};

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
    components: usize,
    frequency: u32,
    duration_secs: f64,
    total_writes: u64,
    throughput_writes_per_sec: f64,
    with_reader: bool,
    clients: usize,
}

impl BenchResult {
    fn print_human(&self) {
        eprintln!("=== elodin-db benchmark ===");
        eprintln!("scenario:     {}", self.scenario);
        eprintln!("components:   {}", self.components);
        eprintln!("frequency:    {} Hz", self.frequency);
        eprintln!("clients:      {}", self.clients);
        eprintln!("with_reader:  {}", self.with_reader);
        eprintln!("duration:     {:.2}s", self.duration_secs);
        eprintln!("total_writes: {}", self.total_writes);
        eprintln!(
            "throughput:   {:.0} writes/sec",
            self.throughput_writes_per_sec
        );
    }

    fn print_json(&self) {
        println!(
            "{{\
            \"scenario\":\"{}\",\
            \"components\":{},\
            \"frequency\":{},\
            \"duration_secs\":{:.3},\
            \"total_writes\":{},\
            \"throughput_writes_per_sec\":{:.1},\
            \"with_reader\":{},\
            \"clients\":{}\
            }}",
            self.scenario,
            self.components,
            self.frequency,
            self.duration_secs,
            self.total_writes,
            self.throughput_writes_per_sec,
            self.with_reader,
            self.clients,
        );
    }
}

#[stellarator::main]
async fn main() {
    let mut args = Args::parse();

    if let Some(scenario) = &args.scenario {
        match scenario {
            Scenario::Customer => {
                args.components = 400;
                args.frequency = 250;
                args.with_reader = true;
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

    // Optionally connect a reader (simulates an Editor follow stream)
    if with_reader {
        let reader_addr = addr;
        stellar(move || run_reader(reader_addr));
    }

    // Distribute components across clients
    let components_per_client = distribute(num_components, num_clients);

    let done = Arc::new(WaitQueue::new());
    let start = Instant::now();
    let target_duration = Duration::from_secs(duration_secs);
    let interval = Duration::from_secs_f64(1.0 / frequency as f64);

    let mut handles = Vec::new();
    let mut vtable_base: u16 = 1;

    for (client_idx, &n) in components_per_client.iter().enumerate() {
        let base = vtable_base;
        vtable_base += n as u16;
        let done = done.clone();

        handles.push(spawn(run_writer(
            addr,
            client_idx,
            n,
            base,
            interval,
            target_duration,
            done,
        )));
    }

    // Wait for all writers to finish
    for handle in handles {
        let _ = handle.await;
    }

    let elapsed = start.elapsed();

    // Count total writes via the DB
    let total_writes = count_total_samples(addr, num_components).await;

    // Cleanup
    let _ = std::fs::remove_dir_all(&temp_dir);

    BenchResult {
        scenario: scenario_name.to_string(),
        components: num_components,
        frequency,
        duration_secs: elapsed.as_secs_f64(),
        total_writes,
        throughput_writes_per_sec: total_writes as f64 / elapsed.as_secs_f64(),
        with_reader,
        clients: num_clients,
    }
}

async fn run_writer(
    addr: SocketAddr,
    _client_idx: usize,
    num_components: usize,
    vtable_base: u16,
    interval: Duration,
    target_duration: Duration,
    _done: Arc<WaitQueue>,
) -> u64 {
    let mut client = Client::connect(addr).await.unwrap();

    // Register VTables -- one per component, each a scalar f64
    for i in 0..num_components {
        let vtable_id = (vtable_base + i as u16).to_le_bytes();
        let comp_name = format!("bench_comp_{}", vtable_base as usize + i);
        let comp_id = ComponentId::new(&comp_name);
        let vt = vtable([raw_field(
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
    }

    sleep(Duration::from_millis(50)).await;

    let start = Instant::now();
    let mut writes: u64 = 0;

    while start.elapsed() < target_duration {
        for i in 0..num_components {
            let vtable_id = (vtable_base + i as u16).to_le_bytes();
            let value = writes as f64;
            let mut pkt = LenPacket::table(vtable_id, 8);
            pkt.extend_aligned(&[value]);
            client.send(pkt).await.0.unwrap();
            writes += 1;
        }
        sleep(interval).await;
    }

    writes
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
