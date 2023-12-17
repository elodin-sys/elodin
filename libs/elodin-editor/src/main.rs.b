use anyhow::anyhow;
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use clap::Parser;
use futures_lite::future;
use notify::RecursiveMode;
use notify::Watcher;
use elodin::sync::ClientChannel;
use elodin::sync::ClientTransport;
use elodin::{
    runner::IntoSimRunner,
    sync::{channel_pair, ServerChannel},
};
use elodin_editor::EditorPlugin;
use elodin_py::SimBuilder;
use pyo3::types::PyModule;
use pyo3::Python;
use std::path::PathBuf;
use std::time::Duration;
use thread_priority::ThreadBuilderExt;

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    {
        use elodin_py::elodin_py;
        pyo3::append_to_inittab!(elodin_py);
    }

    let (tx, rx) = channel_pair();
    let mut app = App::new();
    app.insert_resource(args);
    app.insert_resource(tx);
    app.add_plugins(EditorPlugin::new(rx));
    app.add_systems(PostStartup, setup_file);
    app.add_systems(Update, poll_file);
    app.run();

    Ok(())
}

#[derive(Component)]
struct FileTask(Task<anyhow::Result<()>>);

fn setup_file(
    mut commands: Commands,
    args: Res<Args>,
    rx: Res<ClientChannel>,
    tx: Res<ServerChannel>,
) {
    let pool = AsyncComputeTaskPool::get();
    let tx = tx.clone();
    let watcher_rx = rx.clone();
    let path = args.path.clone();
    commands.spawn(FileTask(pool.spawn(async move {
        let path = if let Some(ref path) = path {
            path.clone()
        } else {
            dbg!(
                rfd::AsyncFileDialog::new()
                    .add_filter("Python", &["py"])
                    .pick_file()
                    .await
            )
            .unwrap()
            .path()
            .to_path_buf()
        };
        load_python(path.clone(), tx.clone())?;
        let watcher_path = path.clone();
        let mut watcher =
            notify::recommended_watcher(move |res: notify::Result<notify::Event>| match res {
                Ok(event) => {
                    println!("got event");
                    if let notify::EventKind::Modify(_) = event.kind {
                        watcher_rx.send_msg(elodin::ServerMsg::Exit);
                        std::thread::sleep(Duration::from_millis(100));
                        load_python(watcher_path.clone(), tx.clone()).unwrap();
                    }
                }
                Err(e) => println!("watch error: {:?}", e),
            })?;

        watcher.watch(&path, RecursiveMode::Recursive)?;
        std::mem::forget(watcher);
        Ok::<(), anyhow::Error>(())
    })));
}

fn poll_file(mut commands: Commands, mut task: Query<(Entity, &mut FileTask)>) {
    for (entity, mut task) in &mut task {
        if let Some(_res) = future::block_on(future::poll_once(&mut task.0)) {
            commands.entity(entity).remove::<FileTask>();
        }
    }
}

fn load_python(path: PathBuf, tx: ServerChannel) -> anyhow::Result<()> {
    std::thread::Builder::new()
        .name("sim thread".to_owned())
        .spawn_with_priority(
            thread_priority::ThreadPriority::Max,
            move |_res| -> anyhow::Result<()> {
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
                    .run_mode(elodin::runner::RunMode::RealTime)
                    .build(tx);
                app.run();
                println!("thread finished run");
                Ok(())
            },
        )?;
    Ok(())
}

#[derive(Parser, Resource)]
struct Args {
    path: Option<PathBuf>,
}
