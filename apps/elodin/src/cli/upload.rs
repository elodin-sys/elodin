use std::path::PathBuf;

use miette::miette;

use super::Cli;

#[derive(clap::Args, Clone, Debug)]
pub struct Args {
    /// Database directory, or a .tar.zst holding exactly one database directory
    pub path: PathBuf,
    /// Elodin Cloud project name
    #[arg(long, value_name = "NAME")]
    pub project: String,
    /// Run name [default: the database directory's name]
    #[arg(long)]
    pub name: Option<String>,
    /// Run tag as KEY=VALUE (repeatable)
    #[arg(long = "tag", value_name = "KEY=VALUE")]
    pub tags: Vec<String>,
    /// Export one table per component instead of joining components by name prefix
    #[arg(long)]
    pub no_join: bool,
    /// Export only components whose names match this glob
    #[arg(long, value_name = "GLOB")]
    pub pattern: Option<String>,
    /// Return once uploaded instead of waiting for ingest
    #[arg(long)]
    pub no_wait: bool,
    /// Print the run as JSON
    #[arg(long)]
    pub json: bool,
}

impl Cli {
    pub fn upload(self, args: Args, rt: tokio::runtime::Runtime) -> miette::Result<()> {
        let cloud = self.cloud()?;
        let source = elodin_mcp::upload::Source::open(&args.path).map_err(|e| miette!("{e:#}"))?;
        let mut tags = serde_json::Map::new();
        for t in &args.tags {
            let (k, v) = t
                .split_once('=')
                .ok_or_else(|| miette!("--tag {t:?} needs KEY=VALUE"))?;
            tags.insert(k.to_string(), v.into());
        }
        let opts = elodin_mcp::upload::Options {
            name: args.name.clone(),
            tags,
            join: !args.no_join,
            pattern: args.pattern.clone(),
            wait: !args.no_wait,
        };
        let run = rt
            .block_on(async {
                let project = cloud.project(&args.project).await?;
                elodin_mcp::upload::upload(&cloud, &project, source, opts).await
            })
            .map_err(|e| miette!("{e:#}"))?;
        if args.json {
            println!("{}", serde_json::to_string_pretty(&run).expect("json"));
        } else {
            print_run(&run);
        }
        match run["state"].as_str() {
            Some("failed") => Err(miette!(
                "ingest failed: {}",
                run["error"].as_str().unwrap_or("unknown error")
            )),
            _ => Ok(()),
        }
    }
}

fn print_run(run: &serde_json::Value) {
    let s = &run["summary"];
    println!(
        "Run {} ({}): {}",
        run["name"].as_str().unwrap_or("-"),
        run["id"].as_str().unwrap_or("-"),
        run["state"].as_str().unwrap_or("-")
    );
    if s.is_object() {
        println!(
            "  {} tables, {} rows; {} populated and {} empty signals; {} samples",
            s["tables"], s["rows"], s["signals"]["populated"], s["signals"]["empty"], s["samples"]
        );
        println!("  recorder time {} to {} µs", s["first_us"], s["last_us"]);
        for c in s["columns_added"].as_array().into_iter().flatten() {
            println!(
                "  added column {}.{}",
                c["table"].as_str().unwrap_or("?"),
                c["column"].as_str().unwrap_or("?")
            );
        }
    }
}
