use std::{
    env,
    fs,
    path::{Path, PathBuf},
};

use impeller2_wkt::DbConfig;

#[derive(Debug)]
struct Options {
    db_state: PathBuf,
    show_metadata: bool,
    show_schematic_content: bool,
}

fn usage() -> &'static str {
    "Usage:\n  db-config-info --db-dir <path> [--show-metadata] [--show-schematic-content]\n  db-config-info --db-state <path> [--show-metadata] [--show-schematic-content]\n"
}

fn parse_args() -> Result<Options, String> {
    let mut args = env::args().skip(1);
    let mut db_state: Option<PathBuf> = None;
    let mut db_dir: Option<PathBuf> = None;
    let mut show_metadata = false;
    let mut show_schematic_content = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--db-dir" => {
                let dir = args.next().ok_or("--db-dir requires a value")?;
                db_dir = Some(PathBuf::from(dir));
            }
            "--db-state" => {
                let path = args.next().ok_or("--db-state requires a value")?;
                db_state = Some(PathBuf::from(path));
            }
            "--show-metadata" => {
                show_metadata = true;
            }
            "--show-schematic-content" => {
                show_schematic_content = true;
            }
            "-h" | "--help" => {
                return Err("help".to_string());
            }
            _ => {
                return Err(format!("unknown argument: {arg}"));
            }
        }
    }

    let db_state = match (db_state, db_dir) {
        (Some(state), None) => state,
        (None, Some(dir)) => dir.join("db_state"),
        (Some(_), Some(_)) => {
            return Err("use either --db-dir or --db-state (not both)".to_string())
        }
        (None, None) => return Err("missing --db-dir or --db-state".to_string()),
    };

    Ok(Options {
        db_state,
        show_metadata,
        show_schematic_content,
    })
}

fn format_duration(duration: std::time::Duration) -> String {
    let nanos = duration.as_nanos();
    if nanos >= 1_000_000_000 {
        format!("{} s", nanos / 1_000_000_000)
    } else if nanos >= 1_000_000 {
        format!("{} ms", nanos / 1_000_000)
    } else if nanos >= 1_000 {
        format!("{} us", nanos / 1_000)
    } else {
        format!("{} ns", nanos)
    }
}

fn print_key_value(key: &str, value: &str) {
    println!("{key}: {value}");
}

fn print_metadata_summary(config: &DbConfig, show_schematic_content: bool) {
    let meta = &config.metadata;
    println!("metadata keys: {}", meta.len());

    match meta.get("schematic.path") {
        Some(path) => print_key_value("schematic.path", path),
        None => print_key_value("schematic.path", "<missing>"),
    }

    match meta.get("schematic.content") {
        Some(content) => {
            if show_schematic_content {
                print_key_value("schematic.content", content);
            } else {
                print_key_value(
                    "schematic.content",
                    &format!("<{} bytes>", content.len()),
                );
            }
        }
        None => print_key_value("schematic.content", "<missing>"),
    }
}

fn print_metadata_full(config: &DbConfig, show_schematic_content: bool) {
    let meta = &config.metadata;
    if meta.is_empty() {
        println!("metadata: <empty>");
        return;
    }

    let mut keys: Vec<&String> = meta.keys().collect();
    keys.sort();
    for key in keys {
        if key == "schematic.content" && !show_schematic_content {
            let value = meta.get(key).map(|v| v.len()).unwrap_or(0);
            println!("{key}: <{value} bytes>");
        } else if let Some(value) = meta.get(key) {
            println!("{key}: {value}");
        }
    }
}

fn run() -> Result<(), String> {
    let opts = parse_args()?;

    if !Path::new(&opts.db_state).exists() {
        return Err(format!(
            "db_state not found: {}",
            opts.db_state.display()
        ));
    }

    let bytes = fs::read(&opts.db_state)
        .map_err(|e| format!("failed to read {}: {e}", opts.db_state.display()))?;
    let config: DbConfig =
        postcard::from_bytes(&bytes).map_err(|e| format!("decode error: {e}"))?;

    println!("db_state: {}", opts.db_state.display());
    println!("recording: {}", config.recording);
    println!(
        "default_stream_time_step: {}",
        format_duration(config.default_stream_time_step)
    );

    if opts.show_metadata {
        print_metadata_full(&config, opts.show_schematic_content);
    } else {
        print_metadata_summary(&config, opts.show_schematic_content);
    }

    Ok(())
}

fn main() {
    match run() {
        Ok(()) => {}
        Err(err) if err == "help" => {
            print!("{}", usage());
        }
        Err(err) => {
            eprintln!("error: {err}\n");
            eprintln!("{}", usage());
            std::process::exit(1);
        }
    }
}
