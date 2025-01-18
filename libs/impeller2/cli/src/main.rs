use impeller2::{
    com_de::Decomponentize,
    table::VTableBuilder,
    types::{ComponentId, EntityId, Msg, PrimType},
};

use impeller2::types::{LenPacket, MsgExt};
use impeller2_wkt::*;
use mlua::prelude::LuaError;
use mlua::{Error, Lua, LuaSerdeExt, MultiValue, UserData, Value};
use nu_ansi_term::Color;
use rustyline::{
    completion::FilenameCompleter,
    highlight::{CmdKind, Highlighter},
    hint::HistoryHinter,
    history::History,
    validate::MatchingBracketValidator,
    Completer, CompletionType, Editor, Helper, Hinter, Validator,
};
use serde::de::DeserializeOwned;
use std::{
    borrow::Cow::{self, Borrowed, Owned},
    fmt::Display,
    io::{self, Read},
    net::SocketAddr,
    sync::{
        atomic::{self, AtomicBool},
        Arc,
    },
};
use stellarator::{
    io::{OwnedReader, OwnedWriter, SplitExt},
    net::TcpStream,
};
use zerocopy::IntoBytes;

struct Client {
    rx: impeller2_stella::PacketStream<OwnedReader<TcpStream>>,
    tx: impeller2_stella::PacketSink<OwnedWriter<TcpStream>>,
}

impl Client {
    pub async fn connect(addr: SocketAddr) -> anyhow::Result<Self> {
        let stream = TcpStream::connect(addr)
            .await
            .map_err(anyhow::Error::from)?;
        let (rx, tx) = stream.split();
        let tx = impeller2_stella::PacketSink::new(tx);
        let rx = impeller2_stella::PacketStream::new(rx);
        Ok(Client { tx, rx })
    }

    pub async fn send_msg(&self, msg: impl Msg) -> anyhow::Result<()> {
        self.tx.send(msg.to_len_packet()).await.0?;
        Ok(())
    }

    pub async fn send_req_reply<R: Msg + DeserializeOwned>(
        &mut self,
        msg: impl Msg,
    ) -> anyhow::Result<R> {
        self.tx.send(msg.to_len_packet()).await.0?;
        let buf = vec![0u8; 8 * 1024];
        let pkt = self.rx.next(buf).await?;
        match &pkt {
            impeller2::types::OwnedPacket::Msg(m) if m.id == R::ID => {
                let m = m.parse::<R>()?;
                Ok(m)
            }
            _ => Err(anyhow::anyhow!("wrong msg type")),
        }
    }

    pub async fn get_time_series(
        &mut self,
        component_id: u64,
        entity_id: u64,
        start: u64,
        stop: u64,
    ) -> anyhow::Result<()> {
        let id = fastrand::u64(..).to_le_bytes()[..3]
            .try_into()
            .expect("id wrong size");

        let msg = GetTimeSeries {
            id,
            range: start..stop,
            entity_id: EntityId(entity_id),
            component_id: ComponentId(component_id),
        };

        self.tx.send(msg.to_len_packet()).await.0?;
        let buf = vec![0u8; 8 * 1024];
        let pkt = self.rx.next(buf).await.unwrap();
        match &pkt {
            impeller2::types::OwnedPacket::TimeSeries(time_series) => {
                println!("time series {:?}", &time_series.buf[..]);
                Ok(())
            }
            _ => Err(anyhow::anyhow!("wrong msg type")),
        }
    }

    pub async fn send(
        &self,
        lua: &Lua,
        component_id: u64,
        entity_id: u64,
        prim_type: PrimType,
        shape: Vec<u64>,
        buf: Value,
    ) -> anyhow::Result<()> {
        let component_id = ComponentId(component_id);
        let entity_id = EntityId(entity_id);
        let mut vtable: VTableBuilder<Vec<_>, Vec<_>> = VTableBuilder::default();
        vtable.column(component_id, prim_type, &shape, std::iter::once(entity_id))?;
        let vtable = vtable.build();
        let id: [u8; 3] = fastrand::u64(..).to_le_bytes()[..3]
            .try_into()
            .expect("id wrong size");
        let msg = VTableMsg { id, vtable };
        self.tx.send(msg.to_len_packet()).await.0?;
        let mut table = LenPacket::table(id, 8);
        match prim_type {
            PrimType::U8 => {
                let buf: Vec<u8> = lua.from_value(buf)?;
                let buf = buf.as_bytes();
                table.extend_from_slice(buf);
            }
            PrimType::U16 => {
                let buf: Vec<u16> = lua.from_value(buf)?;
                let buf = buf.as_bytes();
                table.extend_from_slice(buf);
            }
            PrimType::U32 => {
                let buf: Vec<u32> = lua.from_value(buf)?;
                let buf = buf.as_bytes();
                table.extend_from_slice(buf);
            }
            PrimType::U64 => {
                let buf: Vec<u64> = lua.from_value(buf)?;
                let buf = buf.as_bytes();
                table.extend_from_slice(buf);
            }
            PrimType::I8 => {
                let buf: Vec<i8> = lua.from_value(buf)?;
                let buf = buf.as_bytes();
                table.extend_from_slice(buf);
            }
            PrimType::I16 => {
                let buf: Vec<i16> = lua.from_value(buf)?;
                let buf = buf.as_bytes();
                table.extend_from_slice(buf);
            }
            PrimType::I32 => {
                let buf: Vec<i32> = lua.from_value(buf)?;
                let buf = buf.as_bytes();
                table.extend_from_slice(buf);
            }
            PrimType::I64 => {
                let buf: Vec<i64> = lua.from_value(buf)?;
                let buf = buf.as_bytes();
                table.extend_from_slice(buf);
            }
            PrimType::Bool => {
                let buf: Vec<bool> = lua.from_value(buf)?;
                let buf = buf.as_bytes();
                table.extend_from_slice(buf);
            }
            PrimType::F32 => {
                let buf: Vec<f32> = lua.from_value(buf)?;
                let buf = buf.as_bytes();
                table.extend_from_slice(buf);
            }
            PrimType::F64 => {
                let buf: Vec<f64> = lua.from_value(buf)?;
                let buf = buf.as_bytes();
                table.extend_from_slice(buf);
            }
        }
        self.tx.send(table).await.0?;
        Ok(())
    }

    pub async fn stream(&mut self, mut stream: Stream) -> anyhow::Result<()> {
        if stream.id == 0 {
            stream.id = fastrand::u64(..);
        }
        self.tx.send(stream.to_len_packet()).await.0?;
        let mut vtable = None;
        let mut buf = vec![0; 1024 * 8];
        let cancel = Arc::new(AtomicBool::new(true));
        let canceler = cancel.clone();
        std::thread::spawn(move || {
            let mut stdin = io::stdin().lock();
            let mut buf = [0u8];
            let _ = stdin.read(&mut buf);
            canceler.store(false, atomic::Ordering::SeqCst);
        });

        while cancel.load(atomic::Ordering::SeqCst) {
            let pkt = self.rx.next(buf).await?;
            match &pkt {
                impeller2::types::OwnedPacket::Msg(msg) if msg.id == VTableMsg::ID => {
                    let msg = msg.parse::<VTableMsg>()?;
                    vtable = Some(msg);
                }
                impeller2::types::OwnedPacket::Msg(msg) => {
                    println!("msg ({:?}) = {:?}", msg.id, &msg.buf[..]);
                }
                impeller2::types::OwnedPacket::Table(table) => {
                    if let Some(vtable) = &vtable {
                        vtable.vtable.parse_table(&table.buf[..], &mut DebugSink)?;
                    } else {
                        println!("table ({:?}) = {:?}", table.id, &table.buf[..]);
                    }
                }
                impeller2::types::OwnedPacket::TimeSeries(_) => {}
            }
            buf = pkt.into_buf();
        }
        Ok(())
    }
}

impl UserData for Client {
    fn add_methods<M: mlua::UserDataMethods<Self>>(methods: &mut M) {
        methods.add_async_method(
            "send",
            |lua, this, (component_id, entity_id, ty, shape, buf): (_, _, _, Vec<u64>, _)| async move {
                let ty: PrimType = lua.from_value(ty)?;
                this.send(&lua, component_id, entity_id, ty, shape, buf).await?;
                Ok(())
            },
        );
        methods.add_async_method_mut(
            "get_time_series",
            |_, mut this, (c_id, e_id, start, stop)| async move {
                this.get_time_series(c_id, e_id, start, stop).await?;
                Ok(())
            },
        );
        methods.add_async_method_mut("stream", |lua, mut this, stream| async move {
            let msg: Stream = lua.from_value(stream)?;
            this.stream(msg).await?;
            Ok(())
        });

        macro_rules! add_send_method {
            ($name:tt, $ty:tt) => {
                methods.add_async_method(stringify!($name), |lua, this, value| async move {
                    let msg: $ty = lua.from_value(value)?;
                    this.send_msg(msg).await?;
                    Ok::<_, LuaError>(())
                });
            };
        }
        macro_rules! add_req_reply_method {
            ($name:tt, $ty:tt, $req:tt) => {
                methods.add_async_method_mut(
                    stringify!($name),
                    |lua, mut this, value| async move {
                        let msg: $ty = lua.from_value(value)?;
                        let res = this.send_req_reply::<$req>(msg).await?;
                        lua.to_value(&res)
                    },
                );
            };
        }
        add_send_method!(set_stream_state, SetStreamState);
        add_send_method!(set_component_metadata, SetComponentMetadata);
        add_send_method!(set_entity_metadata, SetEntityMetadata);
        add_send_method!(set_asset, SetAsset);
        add_req_reply_method!(get_asset, GetAsset, Asset);
        add_req_reply_method!(
            get_component_metadata,
            GetComponentMetadata,
            ComponentMetadata
        );
        add_req_reply_method!(dump_metadata, DumpMetadata, DumpMetadataResp);
        add_req_reply_method!(get_entity_metadata, GetEntityMetadata, EntityMetadata);
        add_req_reply_method!(get_schema, GetSchema, SchemaMsg);
    }
}

#[derive(Helper, Completer, Validator, Hinter)]
struct CliHelper {
    #[rustyline(Completer)]
    completer: FilenameCompleter,
    #[rustyline(Validator)]
    validator: MatchingBracketValidator,
    #[rustyline(Hinter)]
    hinter: HistoryHinter,
}

impl Highlighter for CliHelper {
    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        default: bool,
    ) -> Cow<'b, str> {
        if default {
            Owned(Color::Blue.bold().paint("impeller ❯❯ ").to_string())
        } else {
            Borrowed(prompt)
        }
    }

    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        Owned(Color::Default.dimmed().paint(hint).to_string())
    }

    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        #[cfg(feature = "highlight")]
        let out = syntastica::highlight(
            line,
            syntastica_parsers::Lang::Lua,
            &syntastica_parsers::LanguageSetImpl::new(),
            &mut syntastica::renderer::TerminalRenderer::new(None),
            syntastica_themes::catppuccin::mocha(),
        )
        .unwrap()
        .into();
        #[cfg(not(feature = "highlight"))]
        let out = Cow::Borrowed(line);
        out
    }

    fn highlight_char(&self, _line: &str, _pos: usize, _kind: CmdKind) -> bool {
        false
    }
}

async fn run() -> anyhow::Result<()> {
    let lua = Lua::new();
    let config = rustyline::Config::builder()
        .history_ignore_space(true)
        .completion_type(CompletionType::List)
        .auto_add_history(true)
        .build();
    let h = CliHelper {
        completer: FilenameCompleter::new(),
        hinter: HistoryHinter::new(),
        validator: MatchingBracketValidator::new(),
    };

    let mut history = rustyline::history::FileHistory::with_config(config);
    let dirs = directories::ProjectDirs::from("systems", "elodin", "impeller2-cli")
        .ok_or_else(|| anyhow::anyhow!("dir not found"))?;
    std::fs::create_dir_all(dirs.data_dir())?;
    let history_path = dirs.data_dir().join("impeller2-history");
    if history_path.exists() {
        history.load(&history_path)?;
    }
    let mut editor: Editor<_, _> = Editor::with_history(config, history)?;
    editor.set_helper(Some(h));
    let client = lua.create_async_function(|_lua, addr: String| async move {
        let addr = addr.parse().map_err(anyhow::Error::from)?;
        let c = Client::connect(addr).await?;
        Ok(c)
    })?;
    lua.globals().set("client", client)?;
    lua.globals().set(
        "EntityId",
        lua.create_function(|lua, id: String| {
            let entity_id = EntityId(id.parse().map_err(anyhow::Error::from)?);
            Ok(lua.to_value(&entity_id))
        })?,
    )?;
    lua.globals().set(
        "ComponentId",
        lua.create_function(|_lua, id: String| {
            let component_id: u64 = id.parse().map_err(anyhow::Error::from)?;
            Ok(component_id)
        })?,
    )?;

    loop {
        let mut prompt = "impeller ❯❯ ";
        let mut line = String::new();
        loop {
            match editor.readline(prompt) {
                Ok(input) => line.push_str(&input),
                Err(_) => return Ok(()),
            }

            if line == ":exit" {
                std::process::exit(0);
            }
            if line == ":help" || line == ":h" {
                println!("{}", Color::Yellow.bold().paint("Impeller Lua REPL"));
                print_usage_line(
                    "client(addr)",
                    "Connects to a new database and returns a client",
                );
                print_usage_line(
                    "Client:send(component_id, entity_id, ty, shape, data)",
                    "Sends a new ComponentValue to the db",
                );
                print_usage_line(
                    "Client:set_stream_state(SetStreamState)",
                    format!(
                        "Sets the stream state using {} {{ id, playing, tick }}",
                        Color::Blue.bold().paint("SetStreamState")
                    ),
                );
                print_usage_line(
                    "Client:get_component_metadata(GetComponentMetadata)",
                    format!(
                        "Gets a component's metadata using {} {{ id }}",
                        Color::Blue.bold().paint("GetComponentMetadata")
                    ),
                );
                print_usage_line(
                    "Client:set_component_metadata(SetComponentMetadata)",
                    format!(
                        "Sets a component's metadata using {} {{ id, name, metadata, asset }}",
                        Color::Blue.bold().paint("SetComponentMetadata")
                    ),
                );
                print_usage_line(
                    "Client:get_entity_metadata(GetEntityMetadata)",
                    format!(
                        "Gets a entity's metadata using {} {{ id }}",
                        Color::Blue.bold().paint("GetEntityMetadata")
                    ),
                );
                print_usage_line(
                    "Client:set_entity_metadata(SetEntityMetadata)",
                    format!(
                        "Sets a entity's metadata using {} {{ id, name, metadata, asset }}",
                        Color::Blue.bold().paint("SetEntityMetadata")
                    ),
                );
                print_usage_line(
                    "Client:get_asset(GetAsset)",
                    format!(
                        "Gets a entity's metadata using {} {{ id }}",
                        Color::Blue.bold().paint("GetAsset")
                    ),
                );
                print_usage_line(
                    "Client:set_asset(SetAsset)",
                    format!(
                        "Sets an asset {} {{ id, buf }}",
                        Color::Blue.bold().paint("SetAsset")
                    ),
                );
                print_usage_line("Client:dump_metadata()", "Dumps all metadata from the db ");
                print_usage_line(
                    "Client:get_schema(GetSchema)",
                    format!(
                        "Gets a components schema {} {{ id }}",
                        Color::Blue.bold().paint("GetSchema")
                    ),
                );
                break;
            }
            editor.save_history(&history_path)?;
            editor.add_history_entry(line.clone())?;
            match lua.load(&line).eval_async::<MultiValue>().await {
                Ok(values) => {
                    println!(
                        "{}",
                        values
                            .iter()
                            .map(|value| {
                                #[cfg(not(feature = "highlight"))]
                                let out = format!("{:#?}", value);
                                #[cfg(feature = "highlight")]
                                let out = syntastica::highlight(
                                    format!("{:#?}", value),
                                    syntastica_parsers::Lang::Lua,
                                    &syntastica_parsers::LanguageSetImpl::new(),
                                    &mut syntastica::renderer::TerminalRenderer::new(None),
                                    syntastica_themes::catppuccin::mocha(),
                                )
                                .unwrap()
                                .to_string();
                                out
                            })
                            .collect::<Vec<_>>()
                            .join("\t")
                    );
                    break;
                }
                Err(Error::SyntaxError {
                    incomplete_input: true,
                    ..
                }) => {
                    line.push('\n');
                    prompt = ">> ";
                }
                Err(e) => {
                    let err = e.to_string();
                    let err = Color::Red.paint(&err);
                    eprintln!("{}", err);
                    break;
                }
            }
        }
    }
}
fn main() -> anyhow::Result<()> {
    stellarator::run(run)
}

fn print_usage_line(name: impl Display, desc: impl Display) {
    let name = Color::Green.bold().paint(format!("- `{name}`")).to_string();
    println!("{name}");
    println!("   {desc}",);
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
