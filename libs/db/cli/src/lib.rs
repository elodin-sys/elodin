use anyhow::anyhow;
use arrow::{
    array::RecordBatch,
    error::ArrowError,
    util::display::{ArrayFormatter, FormatOptions},
};
use impeller2::{
    com_de::Decomponentize,
    schema::Schema,
    types::{ComponentId, Msg, OwnedTimeSeries, PacketId, PrimType, Request, Timestamp, msg_id},
    vtable::{
        self, VTable,
        builder::{FieldBuilder, OpBuilder, schema},
    },
};

use impeller2::types::{IntoLenPacket, LenPacket, OwnedPacket};
use impeller2_wkt::*;
use mlua::{
    AnyUserData, Error, Lua, LuaSerdeExt, MultiValue, ObjectLike, UserData, UserDataRef, Value,
};
use nu_ansi_term::Color;
use rustyline::{
    Completer, CompletionType, Editor, Helper, Hinter, Validator,
    completion::FilenameCompleter,
    highlight::{CmdKind, Highlighter},
    hint::HistoryHinter,
    history::History,
    validate::MatchingBracketValidator,
};
use std::{
    borrow::Cow::{self, Borrowed, Owned},
    collections::HashMap,
    fmt::Display,
    io::{self, Read},
    net::ToSocketAddrs,
    ops::Deref,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{self, AtomicBool},
    },
    time::Duration,
};
use stellarator::buf::Slice;
use zerocopy::{Immutable, IntoBytes, TryFromBytes};

pub use mlua;

pub struct Client {
    client: impeller2_stellar::Client,
}

impl Client {
    pub async fn connect<T: ToSocketAddrs>(addr: T) -> anyhow::Result<Self> {
        let addr = addr
            .to_socket_addrs()
            .map_err(anyhow::Error::from)?
            .next()
            .ok_or_else(|| anyhow!("missing socket ip"))?;
        let client = impeller2_stellar::Client::connect(addr).await?;
        Ok(Client { client })
    }

    pub async fn request<M: Request + IntoLenPacket>(
        &mut self,
        msg: M,
    ) -> anyhow::Result<M::Reply<Slice<Vec<u8>>>> {
        let resp = async {
            let resp = self.client.request(msg).await?;
            Ok(resp)
        };
        let timeout = async {
            stellarator::sleep(Duration::from_secs(3)).await;
            Err(anyhow!("request timed out"))
        };
        futures_lite::future::race(timeout, resp).await
    }

    pub async fn get_time_series(
        &mut self,
        lua: &Lua,
        component_id: Value,
        start: Option<i64>,
        stop: Option<i64>,
    ) -> anyhow::Result<()> {
        let start = start.unwrap_or(i64::MIN);
        let stop = stop.unwrap_or(i64::MAX);
        let id = fastrand::u16(..);

        let component_id: ComponentId = lua.from_value(component_id)?;
        let schema = self.client.request(&GetSchema { component_id }).await?;
        let start = Timestamp(start);
        let stop = Timestamp(stop);
        let msg = GetTimeSeries {
            id: id.to_le_bytes(),
            range: start..stop,
            component_id,
            limit: Some(256),
        };

        let time_series = self.request(&msg).await?;

        fn print_time_series_as_table<
            T: Immutable + TryFromBytes + Copy + std::fmt::Display + Default + 'static,
        >(
            time_series: &OwnedTimeSeries<Slice<Vec<u8>>>,
            schema: Schema<Vec<u64>>,
        ) -> Result<(), anyhow::Error> {
            let len = schema.shape().iter().product();
            let data = time_series
                .data()
                .map_err(|err| anyhow!("{err:?} failed to get data"))?;
            let buf = <[T]>::try_ref_from_bytes(data).map_err(|_| anyhow!("failed to get data"))?;
            let mut builder = tabled::builder::Builder::default();
            builder.push_record(["TIME".to_string(), "DATA".to_string()]);
            for (chunk, timestamp) in buf
                .chunks(len)
                .zip(time_series.timestamps().unwrap().iter())
            {
                let view = nox::ArrayView::from_buf_shape_unchecked(chunk, schema.shape());
                let epoch = hifitime::Epoch::from(*timestamp);
                builder.push_record([epoch.to_string(), view.to_string()])
            }
            println!(
                "{}",
                builder
                    .build()
                    .with(tabled::settings::Style::rounded())
                    .with(tabled::settings::style::BorderColor::filled(
                        tabled::settings::Color::FG_BLUE
                    ))
            );
            Ok(())
        }

        let schema = schema.0;
        match schema.prim_type() {
            PrimType::U8 => print_time_series_as_table::<u8>(&time_series, schema),
            PrimType::U16 => print_time_series_as_table::<u16>(&time_series, schema),
            PrimType::U32 => print_time_series_as_table::<u32>(&time_series, schema),
            PrimType::U64 => print_time_series_as_table::<u64>(&time_series, schema),
            PrimType::I8 => print_time_series_as_table::<i8>(&time_series, schema),
            PrimType::I16 => print_time_series_as_table::<i16>(&time_series, schema),
            PrimType::I32 => print_time_series_as_table::<i32>(&time_series, schema),
            PrimType::I64 => print_time_series_as_table::<i64>(&time_series, schema),
            PrimType::Bool => print_time_series_as_table::<bool>(&time_series, schema),
            PrimType::F32 => print_time_series_as_table::<f32>(&time_series, schema),
            PrimType::F64 => print_time_series_as_table::<f64>(&time_series, schema),
        }
    }

    pub async fn sql(&mut self, sql: &str) -> anyhow::Result<()> {
        let stream = self.client.stream(&SQLQuery(sql.to_string())).await?;
        let mut batches = vec![];
        futures_lite::pin!(stream);
        loop {
            let msg = stream.next().await?;
            let Some(batch) = msg.batch else {
                break;
            };
            let mut decoder = arrow::ipc::reader::StreamDecoder::new();
            let mut buffer = arrow::buffer::Buffer::from(batch.into_owned());
            if let Some(batch) = decoder.decode(&mut buffer)? {
                batches.push(batch);
            }
        }
        let mut table = create_table(&batches, &FormatOptions::default())?;
        println!(
            "{}",
            table.with(tabled::settings::Style::rounded()).with(
                tabled::settings::style::BorderColor::filled(tabled::settings::Color::FG_BLUE)
            )
        );
        Ok(())
    }

    pub async fn send(
        &mut self,
        lua: &Lua,
        component_id: u64,
        prim_type: PrimType,
        shape: Vec<u64>,
        buf: Value,
    ) -> anyhow::Result<()> {
        use vtable::builder::*;
        let component_id = ComponentId(component_id);
        let size = shape.iter().product::<u64>() as usize * prim_type.size();
        let vtable = vtable([raw_field(
            0,
            size as u16,
            schema(prim_type, &shape, component(component_id)),
        )]);
        let id: [u8; 2] = fastrand::u16(..).to_le_bytes();
        let msg = VTableMsg { id, vtable };
        self.client.send(&msg).await.0?;
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
        self.client.send(table).await.0?;
        Ok(())
    }

    pub async fn stream(&mut self, mut stream: Stream) -> anyhow::Result<()> {
        if stream.id == 0 {
            stream.id = fastrand::u64(..);
        }
        let stream = self.client.stream(&stream).await?;
        let cancel = Arc::new(AtomicBool::new(true));
        let canceler = cancel.clone();
        let mut vtable: HashMap<PacketId, VTable> = HashMap::new();
        std::thread::spawn(move || {
            let mut stdin = io::stdin().lock();
            let mut buf = [0u8];
            let _ = stdin.read(&mut buf);
            canceler.store(false, atomic::Ordering::SeqCst);
        });

        futures_lite::pin!(stream);
        while cancel.load(atomic::Ordering::SeqCst) {
            let msg = stream.next().await?;
            match msg {
                StreamReply::Table(table) => {
                    if let Some(vtable) = vtable.get(&table.id) {
                        vtable.apply(&table.buf[..], &mut DebugSink)??;
                    } else {
                        println!("table ({:?}) = {:?}", table.id, &table.buf[..]);
                    }
                }
                StreamReply::VTable(msg) => {
                    vtable.insert(msg.id, msg.vtable);
                }
            }
        }
        Ok(())
    }

    pub async fn vtable_stream(&mut self, vtable: VTable) -> anyhow::Result<()> {
        let id = fastrand::u16(..).to_le_bytes();
        let vtable_msg = VTableMsg { vtable, id };
        self.client.send(&vtable_msg).await.0?;
        let stream = self.client.stream(&VTableStream { id }).await?;
        let cancel = Arc::new(AtomicBool::new(true));
        let canceler = cancel.clone();
        std::thread::spawn(move || {
            let mut stdin = io::stdin().lock();
            let mut buf = [0u8];
            let _ = stdin.read(&mut buf);
            canceler.store(false, atomic::Ordering::SeqCst);
        });

        futures_lite::pin!(stream);
        while cancel.load(atomic::Ordering::SeqCst) {
            let msg = stream.next().await?;
            match msg {
                StreamReply::Table(table) => {
                    vtable_msg.vtable.apply(&table.buf[..], &mut DebugSink)??;
                }
                StreamReply::VTable(_) => {}
            }
        }
        Ok(())
    }

    pub async fn stream_msgs(&mut self, stream_msgs: MsgStream) -> anyhow::Result<()> {
        let metadata = self
            .request(&GetMsgMetadata {
                msg_id: stream_msgs.msg_id,
            })
            .await?;

        let request_id = fastrand::u8(..);
        self.client
            .send(stream_msgs.with_request_id(request_id))
            .await
            .0?;

        let cancel = Arc::new(AtomicBool::new(true));
        let canceler = cancel.clone();
        std::thread::spawn(move || {
            let mut stdin = io::stdin().lock();
            let mut buf = [0u8];
            let _ = stdin.read(&mut buf);
            canceler.store(false, atomic::Ordering::SeqCst);
        });

        while cancel.load(atomic::Ordering::SeqCst) {
            let packet = self
                .client
                .recv::<OwnedPacket<Slice<Vec<u8>>>>(request_id)
                .await?;
            if let OwnedPacket::Msg(msg) = packet {
                let data = postcard_dyn::from_slice_dyn(&metadata.schema, &msg.buf[..])
                    .map_err(|e| anyhow!("failed to deserialize msg: {:?}", e))?;
                println!("{:?}", data);
            }
        }
        Ok(())
    }

    pub async fn get_msgs(
        &mut self,
        msg_id: PacketId,
        start: Option<i64>,
        stop: Option<i64>,
    ) -> anyhow::Result<()> {
        let start = Timestamp(start.unwrap_or(i64::MIN));
        let stop = Timestamp(stop.unwrap_or(i64::MAX));
        let metadata = self.request(&GetMsgMetadata { msg_id }).await?;
        let get_msgs = GetMsgs {
            msg_id,
            range: start..stop,
            limit: Some(1000),
        };
        let batch = self.request(&get_msgs).await?;
        let mut builder = tabled::builder::Builder::default();
        for (timestamp, msg) in batch.data {
            let data = postcard_dyn::from_slice_dyn(&metadata.schema, &msg[..])
                .map_err(|e| anyhow!("failed to deserialize msg: {:?}", e))?;

            let epoch = hifitime::Epoch::from(timestamp);
            builder.push_record([epoch.to_string(), data.to_string()]);
        }
        println!(
            "{}",
            builder
                .build()
                .with(tabled::settings::Style::rounded())
                .with(tabled::settings::style::BorderColor::filled(
                    tabled::settings::Color::FG_BLUE
                ))
        );
        Ok(())
    }

    pub async fn send_msg(
        &mut self,
        msg_id: PacketId,
        msg: postcard_dyn::Value,
    ) -> anyhow::Result<()> {
        let metadata = self.request(&GetMsgMetadata { msg_id }).await?;
        let bytes =
            postcard_dyn::to_stdvec_dyn(&metadata.schema, &msg).map_err(|e| anyhow!("{e:?}"))?;
        let mut pkt = LenPacket::msg(msg_id, bytes.len());
        pkt.extend_from_slice(&bytes);
        self.client.send(pkt).await.0?;
        Ok(())
    }
}

fn create_table(
    results: &[RecordBatch],
    options: &FormatOptions,
) -> anyhow::Result<tabled::Table, anyhow::Error> {
    let mut builder = tabled::builder::Builder::default();

    if results.is_empty() {
        return Ok(builder.build());
    }

    let schema = results[0].schema();

    let mut header = Vec::new();
    for field in schema.fields() {
        header.push(field.name());
    }
    builder.push_record(header);

    for batch in results {
        let formatters = batch
            .columns()
            .iter()
            .map(|c| ArrayFormatter::try_new(c.as_ref(), options))
            .collect::<Result<Vec<_>, ArrowError>>()?;

        for row in 0..batch.num_rows() {
            let mut cells = Vec::new();
            for formatter in &formatters {
                cells.push(formatter.value(row).to_string());
            }
            builder.push_record(cells);
        }
    }

    Ok(builder.build())
}

impl UserData for Client {
    fn add_methods<M: mlua::UserDataMethods<Self>>(methods: &mut M) {
        methods.add_async_method_mut(
            "send_table",
            |lua, mut this, (component_id, ty, shape, buf): (Value, _, Vec<u64>, _)| async move {
                let component_id =
                    if let Ok(id) = lua.from_value::<ComponentId>(component_id.clone()) {
                        id
                    } else if let Ok(name) = lua.from_value::<String>(component_id.clone()) {
                        ComponentId::new(&name)
                    } else if let Ok(id) = lua.from_value::<i64>(component_id) {
                        ComponentId(id as u64)
                    } else {
                        return Err(anyhow!("component id must be a ComponentId or String").into());
                    };
                let ty: PrimType = lua.from_value(ty)?;
                this.send(&lua, component_id.0, ty, shape, buf).await?;
                Ok(())
            },
        );
        methods.add_async_method_mut(
            "send_msg",
            |lua, mut this, (msg_or_id, val): (Value, Option<Value>)| async move {
                if let Some(msg) = msg_or_id.as_userdata() {
                    let msg = msg.call_method::<Vec<u8>>("msg", ())?;
                    this.client
                        .send(LenPacket { inner: msg })
                        .await
                        .0
                        .map_err(anyhow::Error::from)?;
                } else if let Some(msg) = val {
                    let id = msg_or_id;
                    let msg_id = if let Ok(id) = lua.from_value::<PacketId>(id.clone()) {
                        id
                    } else if let Ok(name) = lua.from_value::<String>(id) {
                        msg_id(&name)
                    } else {
                        return Err(anyhow!("msg id must be a PacketId or String").into());
                    };
                    let msg = lua.from_value(msg)?;
                    this.send_msg(msg_id, msg).await?;
                } else {
                    return Err(anyhow!(
                        "send_msg requires either a native msg or a id and a table"
                    )
                    .into());
                };
                Ok(())
            },
        );

        methods.add_async_method_mut(
            "send_msgs",
            |_lua, mut this, msgs: Vec<AnyUserData>| async move {
                for msg in msgs {
                    let msg = msg.call_method::<Vec<u8>>("msg", ())?;
                    this.client
                        .send(LenPacket { inner: msg })
                        .await
                        .0
                        .map_err(anyhow::Error::from)?;
                }
                Ok(())
            },
        );

        methods.add_async_method_mut("sql", |_lua, mut this, sql: String| async move {
            this.sql(&sql).await?;
            Ok(())
        });
        methods.add_async_method_mut(
            "get_time_series",
            |lua, mut this, (c_id, start, stop)| async move {
                this.get_time_series(&lua, c_id, start, stop).await?;
                Ok(())
            },
        );
        methods.add_async_method_mut("stream", |lua, mut this, stream| async move {
            let msg: Stream = lua.from_value(stream)?;
            this.stream(msg).await?;
            Ok(())
        });

        methods.add_async_method_mut(
            "vtable_stream",
            |_, mut this, fields: Vec<UserDataRef<LuaFieldBuilder>>| async move {
                let fields = fields.into_iter().map(|field| field.0.clone());
                let vtable = vtable::builder::vtable(fields);
                this.vtable_stream(vtable).await?;
                Ok(())
            },
        );

        methods.add_async_method_mut("stream_msgs", |lua, mut this, id: Value| async move {
            let msg_id = if let Ok(id) = lua.from_value::<PacketId>(id.clone()) {
                id
            } else if let Ok(name) = lua.from_value::<String>(id) {
                msg_id(&name)
            } else {
                return Err(anyhow!("msg id must be a PacketId or String").into());
            };
            this.stream_msgs(MsgStream { msg_id }).await?;
            Ok(())
        });

        methods.add_async_method_mut(
            "get_msgs",
            |lua, mut this, (id, start, stop): (Value, Option<i64>, Option<i64>)| async move {
                let msg_id = if let Ok(id) = lua.from_value::<PacketId>(id.clone()) {
                    id
                } else if let Ok(name) = lua.from_value::<String>(id) {
                    msg_id(&name)
                } else {
                    return Err(anyhow!("msg id must be a PacketId or String").into());
                };
                this.get_msgs(msg_id, start, stop).await?;
                Ok(())
            },
        );
        methods.add_async_method_mut(
            "save_archive",
            |lua, mut this, (path, format): (PathBuf, Option<Value>)| async move {
                let format = if let Some(format) = format {
                    lua.from_value(format)?
                } else {
                    ArchiveFormat::ArrowIpc
                };
                this.request(&SaveArchive { path, format }).await?;
                Ok(())
            },
        );

        macro_rules! add_req_reply_method {
            ($name:tt, $ty:tt, $req:tt) => {
                methods.add_async_method_mut(
                    stringify!($name),
                    |lua, mut this, value| async move {
                        let msg: $ty = lua.from_value(value)?;
                        let res = this.request(&msg).await?;
                        lua.to_value(&res)
                    },
                );
            };
        }
        add_req_reply_method!(
            get_component_metadata,
            GetComponentMetadata,
            ComponentMetadata
        );
        add_req_reply_method!(dump_metadata, DumpMetadata, DumpMetadataResp);
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
            Owned(Color::Blue.bold().paint(prompt).to_string())
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

#[derive(clap::Args, Clone, Debug)]
pub struct Args {
    pub path: Option<PathBuf>,
}

struct LuaMsg<M: Msg>(M);

impl<M: Msg> UserData for LuaMsg<M> {
    fn add_methods<T: mlua::UserDataMethods<Self>>(methods: &mut T) {
        methods.add_method("msg", |_, this, ()| {
            let msg = this.0.into_len_packet().inner;
            Ok(msg)
        });
    }
}

pub fn lua() -> anyhow::Result<Lua> {
    let lua = Lua::new();
    let client = lua.create_async_function(|_lua, addr: String| async move {
        let c = Client::connect(addr).await?;
        Ok(c)
    })?;
    lua.globals().set("connect", client)?;
    lua.globals().set(
        "ComponentId",
        lua.create_function(|lua, name: String| lua.create_ser_userdata(ComponentId::new(&name)))?,
    )?;
    lua.globals().set(
        "SetComponentMetadata",
        lua.create_function(|lua, m: SetComponentMetadata| lua.create_ser_userdata(m))?,
    )?;

    lua.globals().set(
        "Stream",
        lua.create_function(|lua, m: Stream| lua.create_ser_userdata(m))?,
    )?;

    lua.globals().set(
        "Stream",
        lua.create_function(|lua, m: Stream| lua.create_ser_userdata(m))?,
    )?;
    lua.globals().set(
        "UdpUnicast",
        lua.create_function(|lua, m: UdpUnicast| lua.create_ser_userdata(m))?,
    )?;

    lua.globals().set(
        "SQLQuery",
        lua.create_function(|lua, m: SQLQuery| lua.create_ser_userdata(m))?,
    )?;
    lua.globals().set(
        "MsgStream",
        lua.create_function(|lua, m: MsgStream| lua.create_ser_userdata(m))?,
    )?;

    lua.globals().set(
        "table_slice",
        lua.create_function(|_, (offset, len): (u64, u64)| {
            Ok(LuaOpBuilder(vtable::builder::raw_table(
                offset as u16,
                len as u16,
            )))
        })?,
    )?;

    lua.globals().set(
        "field",
        lua.create_function(
            |_, (offset, len, op): (u64, u64, UserDataRef<LuaOpBuilder>)| {
                let op = op.deref().0.clone();
                Ok(LuaFieldBuilder(vtable::builder::raw_field(
                    offset as u16,
                    len as u16,
                    op,
                )))
            },
        )?,
    )?;
    lua.globals().set(
        "component",
        lua.create_function(|lua, component_id: Value| {
            let component_id = if let Ok(id) = lua.from_value::<ComponentId>(component_id.clone()) {
                id
            } else if let Ok(name) = lua.from_value::<String>(component_id.clone()) {
                ComponentId::new(&name)
            } else if let Ok(id) = lua.from_value::<i64>(component_id) {
                ComponentId(id as u64)
            } else {
                return Err(anyhow!("component id must be a ComponentId or String").into());
            };
            Ok(LuaOpBuilder(vtable::builder::component(component_id)))
        })?,
    )?;
    lua.globals().set(
        "schema",
        lua.create_function(
            |lua, (ty, shape, arg): (Value, Vec<u64>, UserDataRef<LuaOpBuilder>)| {
                let ty: PrimType = lua.from_value(ty)?;
                let arg = arg.deref().0.clone();
                Ok(LuaOpBuilder(schema(ty, &shape[..], arg)))
            },
        )?,
    )?;
    lua.globals().set(
        "timestamp",
        lua.create_function(
            |_, (t, arg): (UserDataRef<LuaOpBuilder>, UserDataRef<LuaOpBuilder>)| {
                let t = t.deref().0.clone();
                let arg = arg.deref().0.clone();
                Ok(LuaOpBuilder(vtable::builder::timestamp(t, arg)))
            },
        )?,
    )?;
    lua.globals().set(
        "mean",
        lua.create_function(|_, (window, arg): (u16, UserDataRef<LuaOpBuilder>)| {
            let arg = arg.deref().0.clone();
            Ok(LuaOpBuilder(vtable::builder::ext(MeanOp { window }, arg)))
        })?,
    )?;
    lua.globals().set(
        "vtable_msg",
        lua.create_function(
            |lua, (id, fields): (u16, Vec<UserDataRef<LuaFieldBuilder>>)| {
                let fields = fields.into_iter().map(|field| field.0.clone());
                let vtable = vtable::builder::vtable(fields);
                let id = id.to_le_bytes();
                lua.create_ser_userdata(VTableMsg { id, vtable })
            },
        )?,
    )?;
    lua.globals().set(
        "udp_vtable_stream",
        lua.create_function(|lua, (id, addr): (u16, String)| {
            let id = id.to_le_bytes();
            lua.create_ser_userdata(UdpVTableStream { id, addr })
        })?,
    )?;

    Ok(lua)
}

pub struct LuaOpBuilder(Arc<OpBuilder>);
impl UserData for LuaOpBuilder {}

pub struct LuaFieldBuilder(FieldBuilder);
impl UserData for LuaFieldBuilder {}

pub async fn run(args: Args) -> anyhow::Result<()> {
    let lua = lua()?;
    if let Some(path) = args.path {
        let script = std::fs::read_to_string(path)?;
        lua.load(&script).eval_async::<MultiValue>().await?;
        Ok(())
    } else {
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
            .ok_or_else(|| anyhow!("dir not found"))?;
        std::fs::create_dir_all(dirs.data_dir())?;
        let history_path = dirs.data_dir().join("impeller2-history");
        if history_path.exists() {
            history.load(&history_path)?;
        }
        let mut editor: Editor<_, _> = Editor::with_history(config, history)?;
        editor.set_helper(Some(h));

        let mut mode = Mode::Lua;
        loop {
            let mut prompt = match &mode {
                Mode::Lua => "db ❯❯ ",
                Mode::Sql(..) => "sql ❯❯ ",
            };
            let mut line = String::new();
            loop {
                line.clear();
                match editor.readline(prompt) {
                    Ok(input) => line.push_str(&input),
                    Err(_) => return Ok(()),
                }

                if line == ":exit" {
                    if matches!(mode, Mode::Sql(_)) {
                        mode = Mode::Lua;
                        break;
                    }
                    std::process::exit(0);
                }
                if line.starts_with(":sql") {
                    let addr = &line.strip_prefix(":sql ").unwrap_or_default();
                    let addr = if addr.is_empty() {
                        "localhost:2240"
                    } else {
                        addr
                    };
                    let client = match Client::connect(addr).await {
                        Ok(c) => c,
                        Err(err) => {
                            println!("{err}");
                            continue;
                        }
                    };
                    mode = Mode::Sql(client);
                    break;
                }
                if line == ":help" || line == ":h" {
                    println!("{}", Color::Yellow.bold().paint("Impeller Lua REPL"));
                    print_usage_line(
                        ":sql addr",
                        "Connects to a database and drops you into a sql repl",
                    );
                    print_usage_line(
                        "connect(addr) -> Client",
                        "Connects to a database and returns a client",
                    );
                    print_message("udp_vtable_stream(id, addr) -> UdpVTableStream");
                    print_usage_line(
                        "Client:send_table(component_id, ty, shape, data)",
                        "Sends a new ComponentValue to the db",
                    );
                    print_usage_line("Client:send_msg(msg)", "Sends a raw message to the db");
                    print_usage_line(
                        "Client:send_msgs(msgs)",
                        "Sends a list of raw messages to the db",
                    );
                    print_usage_line(
                        "Client:get_component_metadata(GetComponentMetadata)",
                        format!(
                            "Gets a component's metadata using {} {{ id }}",
                            Color::Blue.bold().paint("GetComponentMetadata")
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
                    print_usage_line(
                        "Client:save_archive(path, format)",
                        r#"Dumps the database to arrow-ipc or parquet files at the specified path
 - path - the path to the folder where the contents will be dumped
 - format - 'arrow-ipc' (default), 'parquet' - the format that will be used"#,
                    );
                    println!("{}", Color::Yellow.bold().paint("Messages"));
                    print_message("SetComponentMetadata { component_id, name, metadata }");
                    print_message(
                        "UdpUnicast { stream = { filter = { component_id }, id }, addr }",
                    );
                    print_message("SetStreamState { id, playing, tick, time_step }");
                    break;
                }
                editor.save_history(&history_path)?;
                editor.add_history_entry(line.clone())?;
                match &mut mode {
                    Mode::Sql(client) => {
                        if line.is_empty() {
                            continue;
                        }
                        if let Err(err) = client.sql(&line).await {
                            let err = err.to_string();
                            println!("{}", Color::Red.paint(&err));
                        }
                    }
                    Mode::Lua => match lua.load(&line).eval_async::<MultiValue>().await {
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
                    },
                }
            }
        }
    }
}

enum Mode {
    Lua,
    Sql(Client),
}

fn print_usage_line(name: impl Display, desc: impl Display) {
    let name = Color::Green.bold().paint(format!("- `{name}`")).to_string();
    println!("{name}");
    println!("   {desc}",);
}

fn print_message(msg: impl Display) {
    let msg = Color::Green.bold().paint(format!("- `{msg}`")).to_string();
    println!("{msg}");
}

struct DebugSink;

impl Decomponentize for DebugSink {
    type Error = core::convert::Infallible;

    fn apply_value(
        &mut self,
        component_id: ComponentId,
        value: impeller2::types::ComponentView<'_>,
        timestamp: Option<Timestamp>,
    ) -> Result<(), Self::Error> {
        let epoch = timestamp.map(hifitime::Epoch::from);
        println!("{component_id:?} @ {epoch:?} = {value:?}");
        Ok(())
    }
}
