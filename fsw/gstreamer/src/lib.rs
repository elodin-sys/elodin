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
    use impeller2::types::{msg_id, LenPacket, Timestamp};
    use std::{
        io::Write,
        net::{SocketAddr, TcpStream},
        str::FromStr,
        sync::Mutex,
    };

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
        }

        impl Default for ElodinSinkState {
            fn default() -> Self {
                Self {
                    db_addr: SocketAddr::new([127, 0, 0, 1].into(), 2240),
                    connection: None,
                    msg_name: "video".to_string(),
                }
            }
        }

        impl ElodinSink {
            fn connect(&self) -> Result<(), gst::ErrorMessage> {
                let mut state = self.state.lock().unwrap();

                state.connection = None;

                match TcpStream::connect(state.db_addr) {
                    Ok(stream) => {
                        if let Err(e) = stream.set_nonblocking(true) {
                            return Err(gst::error_msg!(
                                gst::ResourceError::Failed,
                                ["failed to set non-blocking mode: {}", e]
                            ));
                        }
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

            fn send_packet(&self, data: &[u8]) -> Result<(), gst::ErrorMessage> {
                let mut state = self.state.lock().unwrap();

                let msg_id = msg_id(&state.msg_name);

                let mut packet =
                    LenPacket::msg_with_timestamp(msg_id, Timestamp::now(), data.len());
                packet.extend_from_slice(data);

                if let Some(stream) = &mut state.connection {
                    match stream.write_all(&packet.inner) {
                        Ok(_) => {}
                        Err(err) => {
                            gst::warning!(
                                gst::CAT_DEFAULT,
                                "error sending packet to db: {:?}",
                                err
                            );
                            self.connect()?;
                        }
                    }
                } else {
                    return Err(gst::error_msg!(
                        gst::ResourceError::NotFound,
                        ["No connection to elodin-db"]
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

                if let Err(e) = self.send_packet(&map) {
                    gst::error!(gst::CAT_DEFAULT, "Failed to send data: {}", e);
                    return Err(gst::FlowError::Error);
                }

                Ok(gst::FlowSuccess::Ok)
            }
        }
    }
}
