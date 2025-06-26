use std::{fmt::Display, net::SocketAddr, sync::OnceLock};

use nu_ansi_term::{Color, Style};
use roci::{
    AsVTable,
    tcp::StreamExt,
    zerocopy::{Immutable, KnownLayout, TryFromBytes},
};

use viuer::KittySupport;

static LOGO_PNG: &[u8] = include_bytes!("./logo.png");

#[stellarator::main]
async fn main() {
    print_padding();
    print_logo();
    print_padding();
    print_os_info();
    print_padding();
    print_hw_info();
    print_padding();
    print_sensor_info().await.unwrap();
    print_padding();
    print_soc_telem().await.unwrap();
}

fn print_logo() {
    if viuer::get_kitty_support() == KittySupport::None {
        print_header("ℵ Aleph", Color::Purple);
    } else {
        let img = image::load_from_memory(LOGO_PNG).expect("Data from stdin could not be decoded.");
        let _ = viuer::print(
            &img,
            &viuer::Config {
                absolute_offset: false,
                width: Some(48),
                height: Some(4),
                x: 2,
                ..Default::default()
            },
        );
    }
}

fn divider_line(color: Color) -> &'static String {
    match color {
        Color::Green => {
            static DIVIDER_LINE: OnceLock<String> = OnceLock::new();
            DIVIDER_LINE.get_or_init(|| Color::Green.paint("▌").to_string())
        }
        Color::Yellow => {
            static DIVIDER_LINE: OnceLock<String> = OnceLock::new();
            DIVIDER_LINE.get_or_init(|| Color::Yellow.paint("▌").to_string())
        }

        Color::Purple => {
            static DIVIDER_LINE: OnceLock<String> = OnceLock::new();
            DIVIDER_LINE.get_or_init(|| Color::Purple.paint("▌").to_string())
        }
        Color::Blue => {
            static DIVIDER_LINE: OnceLock<String> = OnceLock::new();
            DIVIDER_LINE.get_or_init(|| Color::Blue.paint("▌").to_string())
        }

        _ => unimplemented!("unsupported divider color"),
    }
}

fn print_padding() {
    println!();
}

fn print_header(text: impl Display, color: Color) {
    println!(
        "{}{}",
        divider_line(color),
        Style::new()
            .bold()
            .on(color)
            .fg(Color::Black)
            .paint(format!(" {text} "))
    );
}

fn print_hw_info() {
    const COLOR: Color = Color::Yellow;
    print_header("HW Info", COLOR);
    let mut system = sysinfo::System::new_all();
    system.refresh_all();
    let core_count = sysinfo::System::physical_core_count().unwrap_or(0);
    let total_ram = system.total_memory();
    let cpu_speed = system
        .cpus()
        .first()
        .map(|c| c.frequency())
        .map(|hz| format!("{hz}MHz"))
        .unwrap_or_else(|| "N/A".to_string());
    println!(
        "{} {} {}",
        divider_line(COLOR),
        COLOR.bold().paint("CPU Cores"),
        core_count
    );

    println!(
        "{} {} {}",
        divider_line(COLOR),
        COLOR.bold().paint("CPU Speed"),
        cpu_speed
    );
    const GIB: f64 = 1000.0 * 1000.0 * 1000.0;
    println!(
        "{} {} {:.3}GB",
        divider_line(COLOR),
        COLOR.bold().paint("RAM"),
        (total_ram as f64) / GIB
    );
}

fn print_os_info() {
    const COLOR: Color = Color::Green;
    print_header("OS Info", COLOR);
    let elodin_version =
        std::fs::read_to_string("/etc/elodin-version").unwrap_or("N/A".to_string());
    let kernel_version = sysinfo::System::kernel_version();
    let kernel_version = kernel_version.as_deref().unwrap_or("N/A");
    let linux_version = sysinfo::System::os_version();
    let linux_version = linux_version.as_deref().unwrap_or("N/A");
    println!(
        "{} {} {}",
        divider_line(COLOR),
        COLOR.bold().paint("OS Version"),
        linux_version,
    );

    println!(
        "{} {} {}",
        divider_line(COLOR),
        COLOR.bold().paint("Kernel Version"),
        kernel_version
    );
    println!(
        "{} {} {}",
        divider_line(COLOR),
        COLOR.bold().paint("Elodin Version"),
        elodin_version
    );
}

async fn print_sensor_info() -> anyhow::Result<()> {
    #[derive(AsVTable, Default, Debug, Clone, TryFromBytes, Immutable, KnownLayout)]
    #[roci(parent = "aleph")]
    struct SensorInfo {
        pub mag: [f32; 3],
        pub gyro: [f32; 3],
        pub accel: [f32; 3],
        pub baro: f32,
        pub q_hat: [f64; 4],
    }

    const COLOR: Color = Color::Purple;

    let mut client =
        impeller2_stellar::Client::connect(SocketAddr::new([127, 0, 0, 1].into(), 2240)).await?;
    let mut sub = client.subscribe::<SensorInfo>().await?;
    let info = sub.next().await?;

    print_header("Sensors", COLOR);

    println!(
        "{} {} {:.3?}",
        divider_line(COLOR),
        COLOR.bold().paint("Mag"),
        info.mag
    );

    println!(
        "{} {} {:.3?}",
        divider_line(COLOR),
        COLOR.bold().paint("Gyro"),
        info.gyro
    );

    println!(
        "{} {} {:.3?}",
        divider_line(COLOR),
        COLOR.bold().paint("Accel"),
        info.accel
    );

    println!(
        "{} {} {:.3?}",
        divider_line(COLOR),
        COLOR.bold().paint("Baro"),
        info.baro
    );
    println!(
        "{} {} {:.3?}",
        divider_line(COLOR),
        COLOR.bold().paint("Attitude (Quat)"),
        info.q_hat
    );

    Ok(())
}

async fn print_soc_telem() -> anyhow::Result<()> {
    #[derive(AsVTable, Default, Debug, Clone, TryFromBytes, Immutable, KnownLayout)]
    #[roci(parent = "aleph")]
    struct HWTelem {
        pub cpu_usage: [f32; 8],
        pub cpu_freq: [f32; 8],
        pub thermal_zones: [f32; 10],
        pub gpu_usage: f32,
    }

    const COLOR: Color = Color::Blue;

    let mut client =
        impeller2_stellar::Client::connect(SocketAddr::new([127, 0, 0, 1].into(), 2240)).await?;
    let mut sub = client.subscribe::<HWTelem>().await?;
    let info = sub.next().await?;

    print_header("SOC Telemetry", COLOR);

    println!(
        "{} {} {:.3?}",
        divider_line(COLOR),
        COLOR.bold().paint("CPU Usage"),
        info.cpu_usage
    );

    println!(
        "{} {} {:.3?}",
        divider_line(COLOR),
        COLOR.bold().paint("CPU Freq"),
        info.cpu_freq
    );

    println!(
        "{} {} {:.3?}",
        divider_line(COLOR),
        COLOR.bold().paint("GPU Usage"),
        info.gpu_usage
    );

    println!(
        "{} {} {:.3?}",
        divider_line(COLOR),
        COLOR.bold().paint("Thermal Zones"),
        info.thermal_zones
    );

    Ok(())
}
