use anyhow::anyhow;
use clap::Parser;
use kdam::{BarExt, term::Colorizer};
use nu_ansi_term::{Color, Style};
use std::{
    fmt::{self, Display},
    io::{self, IsTerminal},
    path::PathBuf,
    process::Command,
    sync::OnceLock,
};
use stellarator::{buf::IoBuf, fs, rent};
use zstd::stream::raw::{Decoder, Operation};

#[derive(Parser, Clone)]
struct Args {
    image: PathBuf,
    #[arg(short, long)]
    disk: Option<String>,
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
        Color::Red => {
            static DIVIDER_LINE: OnceLock<String> = OnceLock::new();
            DIVIDER_LINE.get_or_init(|| Color::Red.paint("▌").to_string())
        }

        _ => unimplemented!("unsupported divider color"),
    }
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

#[stellarator::main]
async fn main() -> anyhow::Result<()> {
    kdam::term::init(io::stderr().is_terminal());

    let args = Args::parse();

    let disks = list_external_disks()?;
    let disk = if let Some(disk) = args.disk {
        disks
            .into_iter()
            .find(|f| f.path == disk)
            .ok_or_else(|| anyhow!("path selected is not an external drive"))?
    } else {
        let mut query =
            promkit::preset::query_selector::QuerySelector::new(disks.clone(), |query, disks| {
                disks
                    .iter()
                    .filter(|disk| disk.contains(query))
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .title("Please select an external drive to flash")
            .listbox_lines(5)
            .prompt()?;
        let disk = query.run()?;
        disks.into_iter().find(|f| f.to_string() == disk).unwrap()
    };
    disk.unmount()?;

    let image_name = args
        .image
        .file_name()
        .and_then(|i| i.to_str())
        .ok_or_else(|| anyhow!("image does not have name"))?;
    let image = fs::File::open(&args.image).await?;
    let image_metadata = std::fs::metadata(&args.image)?;
    let disk_path = PathBuf::from(&disk.path);
    let dest = fs::File::open_with(&disk_path, fs::OpenOptions::new().write(true)).await?;

    let is_zstd = args.image.extension().is_some_and(|ext| ext == "zst");

    let mut buf = vec![0; 1024 * 256];
    let mut cursor = 0;

    println!();
    print_header(
        format!("Flashing {image_name} to {}", disk.identifier),
        Color::Blue,
    );
    println!();

    let total = image_metadata.len() as usize;
    let mut written = 0.0;
    let mut bar = kdam::tqdm!(
        total = total / 1024 / 1024,
        bar_format = format!(
            "{{animation}} {} ",
            "{percentage:3.1}% {rate:.4}{unit}/s|{elapsed human=true}|{remaining human=true}"
                .colorize("#EE6FF8")
        ),
        colour = kdam::Colour::gradient(&["#5A56E0", "#EE6FF8"]),
        dynamic_ncols = true,
        unit = "MB",
        unit_scale = true,
        force_refresh = true
    );
    if is_zstd {
        let mut decoder = Decoder::new()?;
        let mut out_pos = 0;
        let mut temp_buf = vec![0; 1024 * 1024 * 2];

        while cursor < image_metadata.len() {
            let read = rent!(image.read_at(buf, cursor).await, buf)?;
            if read == 0 {
                break;
            }

            let mut in_pos = 0;
            while in_pos < read {
                let in_buffer = &buf[in_pos..read];

                let result = decoder.run_on_buffers(in_buffer, &mut temp_buf)?;

                if result.bytes_written > 0 {
                    let mut write_slice = temp_buf.try_slice(..result.bytes_written).unwrap();
                    rent!(dest.write_at(write_slice, out_pos).await, write_slice)?;
                    temp_buf = write_slice.into_inner();

                    out_pos += result.bytes_written as u64;
                }

                in_pos += result.bytes_read;
            }

            written += read as f64 / 1024.0 / 1024.0;
            let _ = bar.update_to(written as usize);
            cursor += read as u64;
        }
    } else {
        while cursor < image_metadata.len() {
            let read = rent!(image.read_at(buf, cursor).await, buf)?;
            let mut slice = buf.try_slice(..read).unwrap();
            rent!(dest.write_at(slice, cursor).await, slice)?;
            written += read as f64 / 1024.0 / 1024.0;
            let _ = bar.update_to(written as usize);
            cursor += read as u64;
            buf = slice.into_inner();
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct ExternalDisk {
    path: String,
    name: Option<String>,
    size: String,
    identifier: String,
}

impl ExternalDisk {
    #[cfg(target_os = "macos")]
    pub fn unmount(&self) -> anyhow::Result<()> {
        let output = Command::new("diskutil")
            .arg("unmountDisk")
            .arg(&self.identifier)
            .output()?;

        if !output.status.success() {
            return Err(anyhow::anyhow!("Failed to run diskutil list"));
        }
        Ok(())
    }

    #[cfg(target_os = "linux")]
    pub fn unmount(&self) -> anyhow::Result<()> {
        let output = Command::new("umount").arg(&self.path).output()?;

        if !output.status.success() {
            return Err(anyhow::anyhow!("Failed to run diskutil list"));
        }
        Ok(())
    }
}

impl Display for ExternalDisk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} - ({})",
            self.identifier,
            self.name.as_deref().unwrap_or_default(),
            self.size
        )
    }
}

#[cfg(target_os = "macos")]
fn list_external_disks() -> anyhow::Result<Vec<ExternalDisk>> {
    let output = Command::new("diskutil").arg("list").output()?;

    if !output.status.success() {
        return Err(anyhow::anyhow!("Failed to run diskutil list"));
    }

    let stdout = String::from_utf8(output.stdout)?;

    let mut disks = Vec::new();

    for line in stdout.lines() {
        if line.starts_with("/dev/disk") && line.contains("external") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            let path = parts[0].to_string();
            let identifier = path.split('/').last().unwrap_or("").to_string();

            disks.push(ExternalDisk {
                path,
                name: None,
                size: String::new(),
                identifier,
            });
        } else if let Some(disk) = &mut disks.last_mut() {
            let line = line.trim();
            if !line.contains("0:") {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            for (i, part) in parts.iter().enumerate() {
                if part.ends_with("GB") || part.ends_with("MB") || part.ends_with("TB") {
                    disk.size = format!("{} {part}", parts[i - 1].replace("*", "")); // size is in the previous chunk
                    break;
                }
            }

            for (i, part) in parts.iter().enumerate() {
                if *part == "NAME" && i + 1 < parts.len() {
                    disk.name = Some(parts[i + 1].to_string());
                    break;
                }
            }
        }
    }

    Ok(disks.into_iter().filter(|d| !d.path.is_empty()).collect())
}

#[cfg(target_os = "linux")]
fn list_external_disks() -> anyhow::Result<Vec<ExternalDisk>> {
    let output = Command::new("lsblk")
        .args(["-o", "NAME,SIZE,TYPE,MODEL,MOUNTPOINT", "-d", "-n", "-p"])
        .output()?;

    if !output.status.success() {
        return Err(anyhow::anyhow!("Failed to run lsblk"));
    }

    let stdout = String::from_utf8(output.stdout)?;
    let mut disks = Vec::new();

    for line in stdout.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 && parts[2] == "disk" {
            let path = parts[0].to_string();

            let is_external = is_external_disk_linux(&path)?;

            if is_external {
                let identifier = path.split('/').last().unwrap_or("").to_string();
                let size = if parts.len() >= 2 {
                    parts[1].to_string()
                } else {
                    String::new()
                };

                let name = if parts.len() >= 4 {
                    Some(parts[3].to_string())
                } else {
                    None
                };

                disks.push(ExternalDisk {
                    path,
                    name,
                    size,
                    identifier,
                });
            }
        }
    }

    Ok(disks)
}

#[cfg(target_os = "linux")]
fn is_external_disk_linux(path: &str) -> anyhow::Result<bool> {
    let device_name = path.split('/').last().unwrap_or("");
    if device_name.is_empty() {
        return Ok(false);
    }

    let removable_path = format!("/sys/block/{}/removable", device_name);
    if let Ok(removable) = std::fs::read_to_string(removable_path) {
        if removable.trim() == "1" {
            return Ok(true);
        }
    }

    let device_path = format!("/sys/block/{}/device", device_name);
    if let Ok(real_path) = std::fs::read_link(device_path) {
        let path_str = real_path.to_string_lossy();
        if path_str.contains("usb") {
            return Ok(true);
        }
    }

    if device_name == "sda" {
        let output = Command::new("ls").arg("/sys/block/").output()?;
        let output_str = String::from_utf8_lossy(&output.stdout);
        let disk_count = output_str
            .split_whitespace()
            .filter(|d| d.starts_with("sd") || d.starts_with("nvme"))
            .count();

        return Ok(disk_count > 1);
    }

    Ok(!device_name.starts_with("nvme"))
}
