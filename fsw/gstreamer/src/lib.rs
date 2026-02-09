use gstreamer as gst;
use gstreamer::glib;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    elodinsink::ElodinSink::register(plugin)?;
    Ok(())
}

gst::plugin_define!(
    elodin,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    env!("CARGO_PKG_VERSION"),
    "MIT",
    "Elodin",
    env!("CARGO_PKG_NAME"),
    "https://github.com/elodin-project/elodin",
    "2025-05-012" // dates make deterministic builds hard
);

mod elodinsink {
    use gstreamer::{self as gst, glib};
    use gstreamer::{prelude::*, subclass::prelude::*};
    use gstreamer_base::subclass::prelude::*;
    use impeller2::types::{msg_id, IntoLenPacket, LenPacket, Timestamp};
    use impeller2_wkt::{
        opaque_bytes_msg_schema, LastUpdated, MsgMetadata, SetMsgMetadata, SubscribeLastUpdated,
    };
    use std::{
        io::{Read, Write},
        net::{SocketAddr, TcpStream},
        str::FromStr,
        sync::Mutex,
    };

    const PACKET_HEADER_LEN: usize = 4; // ty (1) + id (2) + req_id (1)

    glib::wrapper! {
        pub struct ElodinSink(ObjectSubclass<imp::ElodinSink>)
            @extends gstreamer_base::BaseSink, gstreamer::Element, gstreamer::Object;
    }

    impl ElodinSink {
        pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
            gst::Element::register(
                Some(plugin),
                "elodinsink",
                gst::Rank::NONE,
                Self::static_type(),
            )
        }
    }

    // Implementation module
    mod imp {
        use std::sync::LazyLock;

        use super::*;

        pub struct ElodinSink {
            state: Mutex<ElodinSinkState>,
        }

        pub struct ElodinSinkState {
            pub db_addr: SocketAddr,
            pub connection: Option<TcpStream>,
            pub msg_name: String,
            pub base_timestamp: Option<Timestamp>, // DB's last_updated at connect time
            pub first_pts: Option<gst::ClockTime>, // First buffer PTS after connect
        }

        impl Default for ElodinSinkState {
            fn default() -> Self {
                Self {
                    db_addr: SocketAddr::new([127, 0, 0, 1].into(), 2240),
                    connection: None,
                    msg_name: "video".to_string(),
                    base_timestamp: None,
                    first_pts: None,
                }
            }
        }

        impl ElodinSink {
            fn connect(&self) -> Result<(), gst::ErrorMessage> {
                let mut state = self.state.lock().unwrap();

                state.connection = None;
                state.base_timestamp = None;
                state.first_pts = None; // Reset PTS anchor on reconnect

                match TcpStream::connect(state.db_addr) {
                    Ok(mut stream) => {
                        // Query last_updated timestamp (blocking, before setting non-blocking)
                        let pkt = (&SubscribeLastUpdated).into_len_packet();
                        stream.write_all(&pkt.inner).map_err(|e| {
                            gst::error_msg!(
                                gst::ResourceError::Failed,
                                ["Failed to send SubscribeLastUpdated: {}", e]
                            )
                        })?;

                        // Read response length (4 bytes, u32 LE)
                        let mut len_buf = [0u8; 4];
                        stream.read_exact(&mut len_buf).map_err(|e| {
                            gst::error_msg!(
                                gst::ResourceError::Failed,
                                ["Failed to read LastUpdated length: {}", e]
                            )
                        })?;
                        let pkt_len = u32::from_le_bytes(len_buf) as usize;

                        // Read the rest of the packet (header + body)
                        let mut pkt_buf = vec![0u8; pkt_len];
                        stream.read_exact(&mut pkt_buf).map_err(|e| {
                            gst::error_msg!(
                                gst::ResourceError::Failed,
                                ["Failed to read LastUpdated packet: {}", e]
                            )
                        })?;

                        // Skip packet header (4 bytes), deserialize body with postcard
                        if pkt_len < PACKET_HEADER_LEN {
                            return Err(gst::error_msg!(
                                gst::ResourceError::Failed,
                                [
                                    "LastUpdated packet too short: {} bytes (need at least {})",
                                    pkt_len,
                                    PACKET_HEADER_LEN
                                ]
                            ));
                        }
                        let body = &pkt_buf[PACKET_HEADER_LEN..];
                        let last_updated: LastUpdated =
                            postcard::from_bytes(body).map_err(|e| {
                                gst::error_msg!(
                                    gst::ResourceError::Failed,
                                    ["Failed to deserialize LastUpdated: {}", e]
                                )
                            })?;
                        let base_timestamp = last_updated.0;

                        gst::info!(
                            gst::CAT_DEFAULT,
                            "Connected to DB, base_timestamp = {}",
                            base_timestamp.0
                        );

                        // Set message metadata (friendly name) so export-videos and DB have it.
                        // Sent on every connect (including reconnect); DB overwrites idempotently.
                        let set_msg_metadata = SetMsgMetadata {
                            id: msg_id(&state.msg_name),
                            metadata: MsgMetadata {
                                name: state.msg_name.clone(),
                                schema: opaque_bytes_msg_schema(),
                                metadata: std::collections::HashMap::new(),
                            },
                        };
                        let pkt = (&set_msg_metadata).into_len_packet();
                        gst::info!(
                            gst::CAT_DEFAULT,
                            "Sending SetMsgMetadata: name={}, id={:?}, pkt_len={}",
                            state.msg_name,
                            msg_id(&state.msg_name),
                            pkt.inner.len()
                        );
                        stream.write_all(&pkt.inner).map_err(|e| {
                            gst::error_msg!(
                                gst::ResourceError::Failed,
                                ["Failed to send SetMsgMetadata: {}", e]
                            )
                        })?;

                        // Now set non-blocking for video streaming
                        if let Err(e) = stream.set_nonblocking(true) {
                            return Err(gst::error_msg!(
                                gst::ResourceError::Failed,
                                ["failed to set non-blocking mode: {}", e]
                            ));
                        }

                        state.base_timestamp = Some(base_timestamp);
                        state.connection = Some(stream);
                        Ok(())
                    }
                    Err(err) => Err(gst::error_msg!(
                        gst::ResourceError::OpenRead,
                        [
                            "failed to connect to elodin-db at {}: {}",
                            state.db_addr,
                            err
                        ]
                    )),
                }
            }

            fn send_packet(
                &self,
                data: &[u8],
                pts: Option<gst::ClockTime>,
            ) -> Result<(), gst::ErrorMessage> {
                let mut state = self.state.lock().unwrap();

                // Track first PTS after connect for offset calculation
                if state.first_pts.is_none() {
                    state.first_pts = pts;
                }

                let msg_id = msg_id(&state.msg_name);

                // Calculate timestamp: base + (current_pts - first_pts)
                let timestamp = match (state.base_timestamp, pts, state.first_pts) {
                    (Some(base), Some(pts), Some(first_pts)) => {
                        let pts_offset = pts.nseconds().saturating_sub(first_pts.nseconds());
                        Timestamp(base.0 + (pts_offset / 1000) as i64)
                    }
                    _ => Timestamp::now(), // Fallback to wall clock
                };

                let mut packet = LenPacket::msg_with_timestamp(msg_id, timestamp, data.len());
                packet.extend_from_slice(data);

                if let Some(stream) = &mut state.connection {
                    // Drain any pending SubscribeLastUpdated responses from the
                    // read buffer. The DB sends LastUpdated continuously, and since
                    // the socket is non-blocking we must discard these to prevent
                    // the TCP receive buffer from saturating (which would back-
                    // pressure the DB's send task).
                    let mut drain_buf = [0u8; 4096];
                    loop {
                        match stream.read(&mut drain_buf) {
                            Ok(0) => break,                                                    // EOF
                            Ok(_) => continue, // discard, keep draining
                            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => break, // no more data
                            Err(_) => break, // other error
                        }
                    }

                    match stream.write_all(&packet.inner) {
                        Ok(_) => return Ok(()),
                        Err(err) => {
                            gst::warning!(
                                gst::CAT_DEFAULT,
                                "error sending packet to db: {:?}, reconnecting and retrying",
                                err
                            );
                            drop(state); // Release lock before reconnecting
                        }
                    }
                } else {
                    return Err(gst::error_msg!(
                        gst::ResourceError::NotFound,
                        ["No connection to elodin-db"]
                    ));
                }

                // Reconnect and retry sending the packet once.
                // connect() resets base_timestamp and first_pts, so we must
                // recalculate the timestamp and rebuild the packet to avoid
                // sending a stale timestamp from the old connection.
                self.connect()?;
                let mut state = self.state.lock().unwrap();

                if state.first_pts.is_none() {
                    state.first_pts = pts;
                }
                let timestamp = match (state.base_timestamp, pts, state.first_pts) {
                    (Some(base), Some(pts), Some(first_pts)) => {
                        let pts_offset = pts.nseconds().saturating_sub(first_pts.nseconds());
                        Timestamp(base.0 + (pts_offset / 1000) as i64)
                    }
                    _ => Timestamp::now(),
                };
                let mut packet = LenPacket::msg_with_timestamp(msg_id, timestamp, data.len());
                packet.extend_from_slice(data);

                if let Some(stream) = &mut state.connection {
                    stream.write_all(&packet.inner).map_err(|e| {
                        gst::error_msg!(
                            gst::ResourceError::Failed,
                            ["Failed to send packet after reconnect: {}", e]
                        )
                    })?;
                } else {
                    return Err(gst::error_msg!(
                        gst::ResourceError::NotFound,
                        ["No connection to elodin-db after reconnect"]
                    ));
                }

                Ok(())
            }
        }

        #[glib::object_subclass]
        impl ObjectSubclass for ElodinSink {
            const NAME: &'static str = "GstElodinSink";
            type Type = super::ElodinSink;
            type ParentType = gstreamer_base::BaseSink;

            fn new() -> Self {
                Self {
                    state: Mutex::new(ElodinSinkState::default()),
                }
            }
        }

        impl ObjectImpl for ElodinSink {
            fn properties() -> &'static [glib::ParamSpec] {
                static PROPERTIES: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
                    vec![
                        glib::ParamSpecString::builder("db-address")
                            .nick("elodin-db addr")
                            .blurb("The address of the elodin-db instance (e.g., 127.0.0.1:2240)")
                            .default_value(Some("127.0.0.1:2240"))
                            .readwrite()
                            .build(),
                        glib::ParamSpecString::builder("msg-name")
                            .nick("msg name")
                            .blurb("The message name to use (will be hashed to get msg-id) defaults to `video`")
                            .readwrite()
                            .build(),
                    ]
                });

                PROPERTIES.as_ref()
            }

            fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
                let mut state = self.state.lock().unwrap();

                match pspec.name() {
                    "db-address" => {
                        if let Ok(addr_str) = value.get::<String>() {
                            match SocketAddr::from_str(&addr_str) {
                                Ok(addr) => {
                                    state.db_addr = addr;
                                }
                                Err(e) => {
                                    gst::error!(
                                        gst::CAT_DEFAULT,
                                        "Invalid socket address '{}': {}",
                                        addr_str,
                                        e
                                    );
                                }
                            }
                        }
                    }
                    "msg-name" => {
                        if let Some(msg_name) = value.get().ok().flatten() {
                            state.msg_name = msg_name;
                        }
                    }
                    _ => unimplemented!(),
                }
            }

            fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
                let state = self.state.lock().unwrap();

                match pspec.name() {
                    "db-address" => state.db_addr.to_string().to_value(),
                    "msg-name" => state.msg_name.to_value(),
                    _ => unimplemented!(),
                }
            }
        }

        impl GstObjectImpl for ElodinSink {}

        impl ElementImpl for ElodinSink {
            fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
                static ELEMENT_METADATA: LazyLock<gst::subclass::ElementMetadata> =
                    LazyLock::new(|| {
                        gst::subclass::ElementMetadata::new(
                            "Elodin Sink",
                            "Sink/Network",
                            "Send H.264 NAL units to elodin-db",
                            "Elodin",
                        )
                    });

                Some(&*ELEMENT_METADATA)
            }

            fn pad_templates() -> &'static [gst::PadTemplate] {
                static PAD_TEMPLATES: LazyLock<Vec<gst::PadTemplate>> = LazyLock::new(|| {
                    let caps = gst::Caps::builder("video/x-h264")
                        .field("stream-format", "byte-stream")
                        .field("alignment", "au")
                        .build();

                    let sink_pad_template = gst::PadTemplate::new(
                        "sink",
                        gst::PadDirection::Sink,
                        gst::PadPresence::Always,
                        &caps,
                    )
                    .unwrap();

                    vec![sink_pad_template]
                });

                PAD_TEMPLATES.as_ref()
            }
        }

        impl BaseSinkImpl for ElodinSink {
            fn start(&self) -> Result<(), gst::ErrorMessage> {
                self.connect()
            }

            fn stop(&self) -> Result<(), gst::ErrorMessage> {
                let mut state = self.state.lock().unwrap();
                state.connection = None;
                Ok(())
            }

            fn render(&self, buffer: &gst::Buffer) -> Result<gst::FlowSuccess, gst::FlowError> {
                let map = buffer.map_readable().map_err(|_| {
                    gst::error!(gst::CAT_DEFAULT, "Failed to map buffer readable");
                    gst::FlowError::Error
                })?;

                if let Err(e) = self.send_packet(&map, buffer.pts()) {
                    gst::error!(gst::CAT_DEFAULT, "Failed to send data: {}", e);
                    return Err(gst::FlowError::Error);
                }

                Ok(gst::FlowSuccess::Ok)
            }
        }
    }
}
