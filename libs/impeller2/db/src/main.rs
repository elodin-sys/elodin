use std::{net::SocketAddr, path::PathBuf, time::Duration};

use clap::Parser;
use impeller2::{
    com_de::Decomponentize,
    table::VTableBuilder,
    types::{ComponentId, EntityId, PrimType},
};
use impeller2_stella::PacketSink;
use impeller2_stella::{LenPacket, Msg};
use impeller_db::{
    control::{SetStreamState, Stream, StreamFilter, StreamId, VTableMsg},
    Server,
};
use miette::{miette, IntoDiagnostic};
use stellarator::{io::SplitExt, net::TcpStream};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
enum Args {
    Serve {
        addr: SocketAddr,
        path: PathBuf,
    },
    Send {
        #[arg(required = true)]
        addr: SocketAddr,
        #[arg(required = true)]
        component_id: u64,
        #[arg(required = true)]
        entity_id: u64,
        #[arg(long, value_delimiter = ',', num_args = 1..)]
        shape: Vec<u64>,
        #[arg(long)]
        ty: String,
        #[arg(long)]
        buf: String,
    },
    Stream {
        #[arg(required = true)]
        addr: SocketAddr,
        #[arg(required = true)]
        time_step: f64,
        #[arg(long)]
        component_id: Option<u64>,
        #[arg(long)]
        entity_id: Option<u64>,
        #[arg(long)]
        start_tick: Option<u64>,
    },
    SetStreamState {
        #[arg(required = true)]
        addr: SocketAddr,
        #[arg(required = true)]
        id: StreamId,
        #[arg(long)]
        playing: Option<bool>,
        #[arg(long)]
        tick: Option<u64>,
    },
}

fn main() -> miette::Result<()> {
    tracing_subscriber::fmt::init();
    stellarator::run(|| async {
        let args = Args::parse();
        match args {
            Args::Serve { addr, path } => {
                let server = Server::new(path, addr).into_diagnostic()?;
                server.run().await.into_diagnostic()
            }
            Args::Send {
                addr,
                component_id,
                entity_id,
                shape,
                ty,
                buf,
            } => {
                let component_id = ComponentId(component_id);
                let entity_id = EntityId(entity_id);

                let prim_type = match ty.as_str() {
                    "u8" => PrimType::U8,
                    "u16" => PrimType::U16,
                    "u32" => PrimType::U32,
                    "u64" => PrimType::U64,
                    "i8" => PrimType::I8,
                    "i16" => PrimType::I16,
                    "i32" => PrimType::I32,
                    "i64" => PrimType::I64,
                    "bool" => PrimType::Bool,
                    "f32" => PrimType::F32,
                    "f64" => PrimType::F64,
                    _ => return Err(miette!("unsupported type: {}", ty)),
                };
                let mut vtable: VTableBuilder<Vec<_>, Vec<_>> = VTableBuilder::default();
                vtable.column(component_id, prim_type, &shape, std::iter::once(entity_id))?;
                let vtable = vtable.build();
                let id = fastrand::u64(..).to_le_bytes()[..7]
                    .try_into()
                    .expect("id wrong size");
                let msg = VTableMsg { id, vtable };

                let stream = TcpStream::connect(addr).await?;
                let sink = PacketSink::new(stream);
                sink.send(msg.to_len_packet()).await.0.unwrap();
                let mut table = LenPacket::table(id, 8);
                let values = buf.split(',').map(|s| s.trim());
                match prim_type {
                    PrimType::U8 => {
                        for value in values {
                            let parsed: u8 = value
                                .parse()
                                .map_err(|e| miette!("failed to parse '{}': {}", value, e))?;
                            table.push(parsed);
                        }
                    }
                    PrimType::U16 => {
                        for value in values {
                            let parsed: u16 = value
                                .parse()
                                .map_err(|e| miette!("failed to parse '{}': {}", value, e))?;
                            table.extend_from_slice(&parsed.to_le_bytes());
                        }
                    }
                    PrimType::U32 => {
                        for value in values {
                            let parsed: u32 = value
                                .parse()
                                .map_err(|e| miette!("Failed to parse '{}': {}", value, e))?;
                            table.extend_from_slice(&parsed.to_le_bytes());
                        }
                    }
                    PrimType::U64 => {
                        for value in values {
                            let parsed: u64 = value
                                .parse()
                                .map_err(|e| miette!("failed to parse '{}': {}", value, e))?;
                            table.extend_from_slice(&parsed.to_le_bytes());
                        }
                    }
                    PrimType::I8 => {
                        for value in values {
                            let parsed: i8 = value
                                .parse()
                                .map_err(|e| miette!("failed to parse '{}': {}", value, e))?;
                            table.push(parsed as u8);
                        }
                    }
                    PrimType::I16 => {
                        for value in values {
                            let parsed: i16 = value
                                .parse()
                                .map_err(|e| miette!("failed to parse '{}': {}", value, e))?;
                            table.extend_from_slice(&parsed.to_le_bytes());
                        }
                    }
                    PrimType::I32 => {
                        for value in values {
                            let parsed: i32 = value
                                .parse()
                                .map_err(|e| miette!("failed to parse '{}': {}", value, e))?;
                            table.extend_from_slice(&parsed.to_le_bytes());
                        }
                    }
                    PrimType::I64 => {
                        for value in values {
                            let parsed: i64 = value
                                .parse()
                                .map_err(|e| miette!("failed to parse '{}': {}", value, e))?;
                            table.extend_from_slice(&parsed.to_le_bytes());
                        }
                    }
                    PrimType::F32 => {
                        for value in values {
                            let parsed: f32 = value
                                .parse()
                                .map_err(|e| miette!("failed to parse '{}': {}", value, e))?;
                            table.extend_from_slice(&parsed.to_le_bytes());
                        }
                    }
                    PrimType::F64 => {
                        for value in values {
                            let parsed: f64 = value
                                .parse()
                                .map_err(|e| miette!("failed to parse '{}': {}", value, e))?;
                            table.extend_from_slice(&parsed.to_le_bytes());
                        }
                    }
                    PrimType::Bool => {
                        for value in values {
                            let parsed: bool = value
                                .parse()
                                .map_err(|e| miette!("failed to parse '{}': {}", value, e))?;
                            table.push(parsed as u8);
                        }
                    }
                }
                sink.send(table).await.0.unwrap();
                Ok(())
            }
            Args::Stream {
                addr,
                time_step,
                component_id,
                entity_id,
                start_tick,
            } => {
                let stream = TcpStream::connect(addr).await?;
                let (rx, tx) = stream.split();
                let mut rx = impeller2_stella::PacketStream::new(rx);
                let tx = impeller2_stella::PacketSink::new(tx);
                let id = fastrand::u64(..);
                println!("starting stream {id}");
                let stream = Stream {
                    filter: StreamFilter {
                        component_id: component_id.map(ComponentId),
                        entity_id: entity_id.map(EntityId),
                    },
                    time_step: Duration::from_secs_f64(time_step),
                    start_tick,
                    id,
                };
                tx.send(stream.to_len_packet()).await.0.unwrap();
                let mut buf = vec![0u8; 1024 * 8];
                let mut vtable = None;
                loop {
                    let pkt = rx.next(buf).await.into_diagnostic()?;
                    match &pkt {
                        impeller2_stella::Packet::Msg(msg) if msg.id == VTableMsg::ID => {
                            let msg = msg.parse::<VTableMsg>().into_diagnostic()?;
                            println!("insert vtable {:?}", msg.id);
                            vtable = Some(msg);
                        }
                        impeller2_stella::Packet::Msg(msg) => {
                            println!("msg ({:?}) = {:?}", msg.id, &msg.buf[..]);
                        }
                        impeller2_stella::Packet::Table(table) => {
                            if let Some(vtable) = &vtable {
                                vtable.vtable.parse_table(&table.buf[..], &mut DebugSink)?;
                            } else {
                                println!("table ({:?}) = {:?}", table.id, &table.buf[..]);
                            }
                        }
                    }
                    buf = pkt.into_buf();
                }
            }
            Args::SetStreamState {
                addr,
                id,
                playing,
                tick,
            } => {
                let msg = SetStreamState { id, playing, tick };
                let stream = TcpStream::connect(addr).await?;
                let (_, tx) = stream.split();
                let tx = impeller2_stella::PacketSink::new(tx);
                tx.send(msg.to_len_packet()).await.0.unwrap();
                Ok(())
            }
        }
    })
}

struct DebugSink;

impl Decomponentize for DebugSink {
    fn apply_value(
        &mut self,
        component_id: ComponentId,
        entity_id: EntityId,
        value: impeller2::types::ComponentView<'_>,
    ) {
        println!("({component_id:?},{entity_id:?}) = {value:?}");
    }
}
