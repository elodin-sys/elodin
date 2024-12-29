use std::{collections::HashMap, net::SocketAddr, path::PathBuf, time::Duration};

use clap::Parser;
use impeller2::{
    com_de::Decomponentize,
    table::VTableBuilder,
    types::{ComponentId, EntityId, PrimType},
};
use impeller2_stella::{LenPacket, Msg};
use impeller_db::{
    control::{
        ComponentMetadata, EntityMetadata, GetComponentMetadata, GetEntityMetadata, GetSchema,
        GetTimeSeries, Metadata, MetadataValue, SchemaMsg, SetComponentMetadata, SetEntityMetadata,
        SetStreamState, Stream, StreamFilter, StreamId, VTableMsg,
    },
    Server,
};
use miette::{miette, IntoDiagnostic};
use stellarator::{
    io::{OwnedReader, OwnedWriter, SplitExt},
    net::TcpStream,
};

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
        addr: SocketAddr,
        time_step: f64,
        #[arg(long)]
        component_id: Option<u64>,
        #[arg(long)]
        entity_id: Option<u64>,
        #[arg(long)]
        start_tick: Option<u64>,
    },
    SetStreamState {
        addr: SocketAddr,
        id: StreamId,
        #[arg(long)]
        playing: Option<bool>,
        #[arg(long)]
        tick: Option<u64>,
    },
    GetSchema {
        addr: SocketAddr,
        component_id: u64,
    },
    GetTimeSeries {
        addr: SocketAddr,
        component_id: u64,
        entity_id: u64,
        start: u64,
        stop: u64,
    },
    SetComponentMetadata {
        addr: SocketAddr,
        component_id: u64,
        data: String,
    },
    GetComponentMetadata {
        addr: SocketAddr,
        component_id: u64,
    },
    SetEntityMetadata {
        #[arg(required = true)]
        addr: SocketAddr,
        #[arg(required = true)]
        entity_id: u64,
        #[arg(required = true)]
        data: String,
    },
    GetEntityMetadata {
        #[arg(required = true)]
        addr: SocketAddr,
        #[arg(required = true)]
        entity_id: u64,
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
                let id = fastrand::u64(..).to_le_bytes()[..3]
                    .try_into()
                    .expect("id wrong size");
                let msg = VTableMsg { id, vtable };
                let (_, sink) = client_pair(addr).await?;
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
                let (mut rx, tx) = client_pair(addr).await?;
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
                        impeller2_stella::Packet::TimeSeries(_) => {}
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
                let (_, tx) = client_pair(addr).await?;
                let msg = SetStreamState { id, playing, tick };
                tx.send(msg.to_len_packet()).await.0.unwrap();
                Ok(())
            }
            Args::GetSchema { addr, component_id } => {
                let (mut rx, tx) = client_pair(addr).await?;
                let msg = GetSchema {
                    component_id: ComponentId(component_id),
                };
                tx.send(msg.to_len_packet()).await.0.unwrap();
                let buf = vec![0u8; 1024];
                let pkt = rx.next(buf).await.into_diagnostic()?;
                match &pkt {
                    impeller2_stella::Packet::Msg(msg) if msg.id == SchemaMsg::ID => {
                        let msg = msg.parse::<SchemaMsg>().into_diagnostic()?.0;
                        println!(
                            "got schema component_id = {} prim_type = {} shape = {:?}",
                            msg.component_id(),
                            msg.prim_type(),
                            msg.shape()
                        );
                    }
                    _ => {}
                }

                Ok(())
            }

            Args::GetTimeSeries {
                addr,
                component_id,
                entity_id,
                start,
                stop,
            } => {
                let id = fastrand::u64(..).to_le_bytes()[..7]
                    .try_into()
                    .expect("id wrong size");

                let msg = GetTimeSeries {
                    id,
                    range: start..stop,
                    entity_id: EntityId(entity_id),
                    component_id: ComponentId(component_id),
                };
                let (mut rx, tx) = client_pair(addr).await?;
                tx.send(msg.to_len_packet()).await.0.unwrap();
                let buf = vec![0u8; 8 * 1024];
                let pkt = rx.next(buf).await.into_diagnostic()?;
                if let impeller2_stella::Packet::TimeSeries(time_series) = pkt {
                    println!("time series {:?}", &time_series.buf[..]);
                }
                Ok(())
            }
            Args::SetComponentMetadata {
                addr,
                component_id,
                data,
            } => {
                let data: HashMap<String, serde_json::Value> =
                    serde_json::from_str(&data).into_diagnostic()?;
                let metadata = data
                    .into_iter()
                    .map(|(k, v)| {
                        (
                            k,
                            match v {
                                serde_json::Value::Bool(b) => MetadataValue::Bool(b),
                                serde_json::Value::Number(n) => {
                                    if let Some(n) = n.as_f64() {
                                        MetadataValue::F64(n)
                                    } else if let Some(n) = n.as_u64() {
                                        MetadataValue::U64(n)
                                    } else if let Some(n) = n.as_i64() {
                                        MetadataValue::I64(n)
                                    } else {
                                        panic!("invalid number type")
                                    }
                                }
                                serde_json::Value::String(s) => MetadataValue::String(s),
                                _ => {
                                    panic!("unsupported json type")
                                }
                            },
                        )
                    })
                    .collect::<HashMap<String, MetadataValue>>();
                let (_, tx) = client_pair(addr).await?;
                tx.send(
                    SetComponentMetadata {
                        component_id: ComponentId(component_id),
                        metadata: Metadata { metadata },
                    }
                    .to_len_packet(),
                )
                .await
                .0
                .unwrap();
                Ok(())
            }
            Args::GetComponentMetadata { addr, component_id } => {
                let (mut rx, tx) = client_pair(addr).await?;
                tx.send(
                    GetComponentMetadata {
                        component_id: ComponentId(component_id),
                    }
                    .to_len_packet(),
                )
                .await
                .0?;
                let buf = vec![0u8; 8 * 1024];
                let pkt = rx.next(buf).await.unwrap();
                match &pkt {
                    impeller2_stella::Packet::Msg(m) if m.id == ComponentMetadata::ID => {
                        let m = m.parse::<ComponentMetadata>().into_diagnostic()?;
                        println!("metadata = {:?}", m.metadata);
                    }
                    _ => {}
                }
                Ok(())
            }
            Args::SetEntityMetadata {
                addr,
                entity_id,
                data,
            } => {
                let data: HashMap<String, serde_json::Value> =
                    serde_json::from_str(&data).into_diagnostic()?;
                let metadata = data
                    .into_iter()
                    .map(|(k, v)| {
                        (
                            k,
                            match v {
                                serde_json::Value::Bool(b) => MetadataValue::Bool(b),
                                serde_json::Value::Number(n) => {
                                    if let Some(n) = n.as_f64() {
                                        MetadataValue::F64(n)
                                    } else if let Some(n) = n.as_u64() {
                                        MetadataValue::U64(n)
                                    } else if let Some(n) = n.as_i64() {
                                        MetadataValue::I64(n)
                                    } else {
                                        panic!("invalid number type")
                                    }
                                }
                                serde_json::Value::String(s) => MetadataValue::String(s),
                                _ => {
                                    panic!("unsupported json type")
                                }
                            },
                        )
                    })
                    .collect::<HashMap<String, MetadataValue>>();
                let (_, tx) = client_pair(addr).await?;
                tx.send(
                    SetEntityMetadata {
                        entity_id: EntityId(entity_id),
                        metadata: Metadata { metadata },
                    }
                    .to_len_packet(),
                )
                .await
                .0
                .unwrap();
                Ok(())
            }
            Args::GetEntityMetadata { addr, entity_id } => {
                let (mut rx, tx) = client_pair(addr).await?;
                tx.send(
                    GetEntityMetadata {
                        entity_id: EntityId(entity_id),
                    }
                    .to_len_packet(),
                )
                .await
                .0?;
                let buf = vec![0u8; 8 * 1024];
                let pkt = rx.next(buf).await.unwrap();
                match &pkt {
                    impeller2_stella::Packet::Msg(m) if m.id == EntityMetadata::ID => {
                        let m = m.parse::<EntityMetadata>().into_diagnostic()?;
                        println!("metadata = {:?}", m.metadata);
                    }
                    _ => {}
                }
                Ok(())
            }
        }
    })
}

async fn client_pair(
    addr: SocketAddr,
) -> Result<
    (
        impeller2_stella::PacketStream<OwnedReader<TcpStream>>,
        impeller2_stella::PacketSink<OwnedWriter<TcpStream>>,
    ),
    miette::Error,
> {
    let stream = TcpStream::connect(addr).await?;
    let (rx, tx) = stream.split();
    let tx = impeller2_stella::PacketSink::new(tx);
    let rx = impeller2_stella::PacketStream::new(rx);
    Ok((rx, tx))
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
