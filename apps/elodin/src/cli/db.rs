use bevy::prelude::{error, info, warn};
use miette::miette;
use std::{io, net::SocketAddr, path::PathBuf};
use stellarator::{
    struc_con::{Joinable, Thread, ThreadBuilder},
    util::CancelToken,
};

type ServeThread = Thread<Option<Result<(), elodin_db::Error>>>;

pub struct DbServer {
    thread: Option<ServeThread>,
    cancel_token: CancelToken,
}

impl DbServer {
    pub fn join(mut self) -> miette::Result<()> {
        self.cancel_token.cancel();
        let Some(thread) = self.thread.take() else {
            return Ok(());
        };
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|err| miette!("failed to create database shutdown runtime: {err}"))?;
        match runtime
            .block_on(thread.join())
            .map_err(|err| miette!("failed to join database server thread: {err}"))?
        {
            Some(result) => result.map_err(|err| miette!("embedded database server failed: {err}")),
            None => Ok(()),
        }
    }
}

impl Drop for DbServer {
    fn drop(&mut self) {
        self.cancel_token.cancel();
    }
}

pub fn serve(
    path: PathBuf,
    addr: SocketAddr,
    cancel_token: CancelToken,
) -> miette::Result<DbServer> {
    info!(path = %path.display(), ?addr, "starting embedded database");
    let server = elodin_db::Server::new(&path, addr).map_err(|err| {
        if matches!(&err, elodin_db::Error::Io(io) if io.kind() == io::ErrorKind::AddrInUse) {
            miette!(
                "port {} is already in use — is another elodin instance running? \
                 Kill it with: lsof -tiTCP:{} -sTCP:LISTEN | xargs kill -9",
                addr.port(),
                addr.port()
            )
        } else {
            miette!(
                "failed to open database {} at {addr}: {err}",
                path.display()
            )
        }
    })?;

    if let Some(source) = elodin_db::assets::resolve_assets_root(None) {
        if !source.is_dir() {
            warn!(
                source = %source.display(),
                "asset source path does not exist; starting without asset ingest"
            );
        } else {
            match elodin_db::assets::ingest_asset_dir(&path, &source) {
                Ok(report) if report.skipped => {
                    info!(source = %source.display(), "assets already ingested; skipping");
                }
                Ok(report) => {
                    info!(
                        source = %source.display(),
                        files = report.file_count,
                        bytes = report.byte_count,
                        "ingested assets into db"
                    );
                }
                Err(err) => {
                    warn!(?err, source = %source.display(), "failed to ingest assets");
                }
            }
        }
    }

    if !server
        .db
        .with_state(|state| state.db_config.schematic_active().is_some())
    {
        match server.db.set_active_schematic("schematics/main.kdl") {
            Ok(()) => info!("set active schematic to schematics/main.kdl"),
            Err(err) => {
                bevy::prelude::debug!(?err, "no schematics/main.kdl to set active; skipping");
            }
        }
    }

    let grpc_addr = elodin_db::grpc::grpc_addr(addr);
    let grpc_listener = std::net::TcpListener::bind(grpc_addr)
        .map_err(|err| miette!("failed to bind gRPC server at {grpc_addr}: {err}"))?;
    elodin_db::assets_http::spawn_assets_http(&path, addr, true, Some(server.db.clone()))
        .map_err(|err| miette!("failed to start assets server: {err}"))?;

    let grpc_db = server.db.clone();
    stellarator::struc_con::tokio(move |_| async move {
        if let Err(err) =
            elodin_db::grpc::serve_listener_with_auth(grpc_listener, grpc_db, None).await
        {
            error!(?err, "gRPC server exited");
        }
    });

    let cancel_on_exit = cancel_token.clone();
    let thread = ThreadBuilder::default()
        .cancel_token(cancel_token.clone())
        .stellar(move || async move {
            let result = server.run().await;
            if let Err(err) = &result {
                error!(?err, "embedded database server exited");
                cancel_on_exit.cancel();
            }
            result
        });
    Ok(DbServer {
        thread: Some(thread),
        cancel_token,
    })
}
