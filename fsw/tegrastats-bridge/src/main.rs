use futures_concurrency::future::{Join, TryJoin};
use impeller2::types::{LenPacket, PacketId};
use impeller2_stellar::Client;
use roci::{AsVTable, Metadatatize, tcp::SinkExt};
use std::{mem, net::SocketAddr, time::Duration};
use stellarator::{fs::File, rent};
use sysinfo::CpuRefreshKind;
use zerocopy::{Immutable, IntoBytes};

#[derive(AsVTable, Metadatatize, IntoBytes, Immutable, Debug)]
#[roci(entity_id = 1)]
pub struct Output {
    pub cpu_usage: [f32; 8],
    pub cpu_freq: [f32; 8],
    pub thermal_zones: [f32; 10],
    pub gpu_usage: f32,
}

async fn connect() -> anyhow::Result<()> {
    let mut client = Client::connect(SocketAddr::new([127, 0, 0, 1].into(), 2240))
        .await
        .map_err(anyhow::Error::from)?;
    let id: PacketId = fastrand::u16(..).to_le_bytes();
    client.init_world::<Output>(id).await?;
    let mut table = LenPacket::table(id, mem::size_of::<Output>());
    let thermal_zone = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        .map(|i| File::open(format!("/sys/devices/virtual/thermal/thermal_zone{i}/temp")))
        .try_join()
        .await?;
    let cpu_freq = [0, 1, 2, 3, 4, 5, 6, 7]
        .map(|i| {
            File::open(format!(
                "/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq"
            ))
        })
        .try_join()
        .await?;
    let gpu_load = File::open("/sys/devices/platform/gpu.0/load").await?;
    let mut system = sysinfo::System::new_all();
    system.refresh_cpu_specifics(CpuRefreshKind::everything());

    loop {
        table.clear();
        let thermal_zones = thermal_zone.each_ref().map(read_to_float).join();
        let cpu_freq = cpu_freq.each_ref().map(read_to_float).join();
        let gpu_load = read_to_float(&gpu_load);
        let (thermal_zones, cpu_freq, gpu_load) = (thermal_zones, cpu_freq, gpu_load).join().await;
        let thermal_zones = thermal_zones.map(|res| res.unwrap_or(f32::NAN) / 1000.0);
        let cpu_freq = cpu_freq.map(|res| res.unwrap_or(f32::NAN));
        let gpu_load = gpu_load.unwrap_or(f32::NAN) / 1000.0;
        system.refresh_cpu_specifics(CpuRefreshKind::everything());
        let mut cpu_usage = [f32::NAN; 8];
        for (cpu, usage) in system.cpus().iter().zip(cpu_usage.iter_mut()) {
            *usage = cpu.cpu_usage();
        }

        let output = Output {
            thermal_zones,
            cpu_usage,
            gpu_usage: gpu_load,
            cpu_freq,
        };
        table.extend_from_slice(output.as_bytes());
        rent!(client.send(table).await, table)?;
        stellarator::sleep(Duration::from_millis(5)).await;
    }
}

async fn read_to_float(file: &File) -> anyhow::Result<f32> {
    let mut buf = vec![0u8; 32];
    let n = rent!(file.read_at(buf, 0).await, buf)?;
    buf.truncate(n);
    let string = String::from_utf8(buf)?;
    let f = string.trim().parse()?;
    Ok(f)
}

#[stellarator::main]
async fn main() -> anyhow::Result<()> {
    loop {
        if let Err(err) = connect().await {
            eprintln!("error connecting {err:?}")
        }
    }
}
