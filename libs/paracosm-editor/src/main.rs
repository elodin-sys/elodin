use anyhow::anyhow;
use bevy::prelude::App;
use clap::Parser;
use notify::RecursiveMode;
use notify::Watcher;
use paracosm::sync::ClientTransport;
use paracosm::{
    runner::IntoSimRunner,
    sync::{channel_pair, ServerChannel},
};
use paracosm_editor::EditorPlugin;
use paracosm_py::SimBuilder;
use pyo3::types::PyModule;
use pyo3::Python;
use std::path::PathBuf;
use std::time::Duration;

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    {
        use paracosm_py::paracosm_py;
        pyo3::append_to_inittab!(paracosm_py);
    }

    let (tx, rx) = channel_pair();
    let path = args.path.clone();
    load_python(args.path.clone(), tx.clone())?;
    let watcher_rx = rx.clone();
    let mut watcher =
        notify::recommended_watcher(move |res: notify::Result<notify::Event>| match res {
            Ok(event) => {
                if let notify::EventKind::Modify(_) = event.kind {
                    watcher_rx.send_msg(paracosm::ServerMsg::Exit);
                    std::thread::sleep(Duration::from_millis(32));
                    load_python(path.clone(), tx.clone()).unwrap();
                }
            }
            Err(e) => println!("watch error: {:?}", e),
        })?;

    watcher.watch(&args.path, RecursiveMode::Recursive)?;

    let mut app = App::new();
    app.add_plugins(EditorPlugin::new(rx));
    app.run();

    Ok(())
}

fn load_python(path: PathBuf, tx: ServerChannel) -> anyhow::Result<()> {
    std::thread::spawn(move || -> anyhow::Result<()> {
        let py_file = std::fs::read_to_string(&path)?;
        let file_name = path
            .file_name()
            .ok_or_else(|| anyhow!("file name not found"))?
            .to_str()
            .ok_or_else(|| anyhow!("filename not utf8"))?;
        let file_stem = path
            .file_stem()
            .ok_or_else(|| anyhow!("file name not found"))?
            .to_str()
            .ok_or_else(|| anyhow!("filename not utf8"))?;

        pyo3::prepare_freethreaded_python();
        let builder = Python::with_gil(|py| {
            let sim = PyModule::from_code(py, &py_file, file_name, file_stem)?;
            let callable = sim.getattr("sim")?;
            let builder = callable.call0()?;
            let builder: SimBuilder = builder.extract().unwrap();

            pyo3::PyResult::Ok(builder.0)
        })
        .map_err(|e| {
            Python::with_gil(|py| {
                e.print_and_set_sys_last_vars(py);
                e
            })
        })?;
        let runner = builder.into_runner();
        let mut app = runner
            .run_mode(paracosm::runner::RunMode::RealTime)
            .build(tx);
        app.run();
        println!("thread finished run");
        Ok(())
    });
    Ok(())
}

#[derive(Parser)]
struct Args {
    path: PathBuf,
}
