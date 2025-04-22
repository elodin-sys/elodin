use clap::Parser;
use kdam::{BarExt, term::Colorizer};
use nu_ansi_term::Color;
use nu_ansi_term::Style;
use std::fmt::Display;
use std::io;
use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::OnceLock;
use stellarator::buf::IoBuf;
use stellarator::fs;
use stellarator::rent;

#[derive(Parser, Clone)]
struct Args {
    image: PathBuf,
    dest: PathBuf,
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
    let args = Args::parse();
    let image_name = args
        .image
        .file_name()
        .and_then(|i| i.to_str())
        .ok_or_else(|| anyhow::anyhow!("image does not have name"))?;
    let image = fs::File::open(&args.image).await?;
    let image_metadata = std::fs::metadata(&args.image)?;
    let dest = fs::File::open_with(&args.dest, fs::OpenOptions::new().write(true)).await?;
    let dest_name = args
        .dest
        .file_name()
        .and_then(|i| i.to_str())
        .ok_or_else(|| anyhow::anyhow!("dest does not have name"))?;

    let mut buf = vec![0; 1024 * 256];
    let mut cursor = 0;
    kdam::term::init(io::stderr().is_terminal());
    print_header(format!("Flashing {image_name} to {dest_name}"), Color::Blue);
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
    while cursor < image_metadata.len() {
        let read = rent!(image.read_at(buf, cursor).await, buf)?;
        let mut slice = buf.try_slice(..read).unwrap();
        rent!(dest.write_at(slice, cursor).await, slice)?;
        written += read as f64 / 1024.0 / 1024.0;
        let _ = bar.update_to(written as usize);
        cursor += read as u64;
        buf = slice.into_inner();
    }
    Ok(())
}
