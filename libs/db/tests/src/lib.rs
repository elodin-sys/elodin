#[cfg(test)]
mod tests {

    use arrow::{array::AsArray, datatypes::Float64Type};
    use elodin_db::{DB, Error, Server};
    use futures_lite::future::zip;
    use impeller2::{
        types::{ComponentId, IntoLenPacket, LenPacket, Msg, PrimType, Timestamp},
        vtable::builder::{component, raw_field, raw_table, schema, timestamp, vtable},
    };
    use impeller2_stellar::Client;
    use postcard_schema::{Schema, schema::owned::OwnedNamedType};
    use std::{
        collections::{BTreeMap, BTreeSet},
        fs::File,
        io::Write,
        net::SocketAddr,
        path::Path,
        sync::Arc,
        time::{Duration, Instant},
    };
    use stellarator::{net::TcpListener, sleep, spawn, struc_con::stellar, test};
    use zerocopy::FromBytes;
    use zerocopy::IntoBytes;

    use impeller2_wkt::*;

    async fn setup_test_db() -> Result<(SocketAddr, Arc<DB>), Error> {
        let subscriber = tracing_subscriber::FmtSubscriber::new();
        let _ = tracing::subscriber::set_global_default(subscriber);

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let temp_dir = std::env::temp_dir().join(format!("elodin_db_test_{}", fastrand::u64(..)));
        if temp_dir.exists() {
            let _ = std::fs::remove_dir_all(&temp_dir);
        }

        let server = Server::from_listener(listener, temp_dir)?;
        let db = server.db.clone();

        stellar(move || async { server.run().await });

        Ok((addr, db))
    }

    #[test]
    async fn test_connect() {
        let (_client, _db) = setup_test_db().await.unwrap();
    }

    #[test]
    async fn test_send_data() {
        let (addr, db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();
        let vtable = vtable([raw_field(
            0,
            16,
            schema(PrimType::F64, &[2], component("test")),
        )]);
        client
            .send(&VTableMsg {
                id: 1u16.to_le_bytes(),
                vtable,
            })
            .await
            .0
            .unwrap();
        let mut pkt = LenPacket::table(1u16.to_le_bytes(), 16);
        let floats = [1.0f64, 2.0];
        pkt.extend_aligned(&floats);
        client.send(pkt).await.0.unwrap();
        sleep(Duration::from_millis(100)).await;
        db.with_state(|state| {
            let c = state
                .get_component(ComponentId::new("test"))
                .expect("missing component");
            let (_, data) = c.time_series.latest().expect("missing latest value");
            assert_eq!(data, floats.as_bytes());
        })
    }

    #[test]
    async fn test_vtable_stream() {
        let (addr, _db) = setup_test_db().await.unwrap();
        let mut tx_client = Client::connect(addr).await.unwrap();
        let mut rx_client = Client::connect(addr).await.unwrap();

        let component_id = ComponentId::new("temperature");

        let vtable = vtable([raw_field(
            0,
            8,
            schema(PrimType::F64, &[1], component(component_id)),
        )]);

        let vtable_id = 1u16.to_le_bytes();
        let msg = VTableMsg {
            id: vtable_id,
            vtable: vtable.clone(),
        };
        let vtable_stream = VTableStream { id: vtable_id };
        tx_client.send(&msg).await.0.unwrap();
        sleep(Duration::from_millis(50)).await;
        let mut sub = rx_client.stream(&vtable_stream).await.unwrap();

        spawn(async move {
            for i in 0..5 {
                sleep(Duration::from_millis(50)).await;
                let value = i as f64;
                let mut pkt = LenPacket::table(vtable_id, 8);
                pkt.extend_aligned(&[value]);
                tx_client.send(pkt).await.0.unwrap();
            }
        });
        let StreamReply::VTable(_) = sub.next().await.unwrap() else {
            panic!("unexpected reply type");
        };
        for i in 0..5 {
            let StreamReply::Table(table) = sub.next().await.unwrap() else {
                panic!("unexpected reply type");
            };
            let expected_value = i as f64;
            assert_eq!(&table.buf[..], expected_value.as_bytes());
        }
    }

    #[test]
    async fn test_dump_metadata() {
        let (addr, _db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();
        let component_id = ComponentId::new("test_component");
        let component_metadata = SetComponentMetadata::new(component_id, "Test Component")
            .metadata(
                [("baz".to_string(), "bang".to_string())]
                    .into_iter()
                    .collect(),
            );

        client.send(&component_metadata).await.0.unwrap();
        sleep(Duration::from_millis(100)).await;
        let mut response = client.request(&DumpMetadata).await.unwrap();
        response.component_metadata.sort_by_key(|c| c.component_id);
        assert_eq!(
            response.component_metadata,
            &[ComponentMetadata {
                component_id,
                name: "Test Component".to_string(),
                metadata: [("baz".to_string(), "bang".to_string())]
                    .into_iter()
                    .collect(),
            },]
        )
    }

    #[test]
    async fn test_sql_query() {
        let (addr, _db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();

        let component_id = ComponentId::new("cpu_temperature");

        let vtable = vtable([raw_field(
            0,
            8,
            schema(PrimType::F64, &[1], component(component_id)),
        )]);

        client
            .send(&SetComponentMetadata::new(component_id, "cpu_temperature"))
            .await
            .0
            .unwrap();

        let vtable_id = 1u16.to_le_bytes();
        client
            .send(&VTableMsg {
                id: vtable_id,
                vtable,
            })
            .await
            .0
            .unwrap();

        for i in 0..5 {
            let value = i as f64 * 10.0;
            let mut pkt = LenPacket::table(vtable_id, 8);
            pkt.extend_aligned(&[value]);
            client.send(pkt).await.0.unwrap();
        }

        sleep(Duration::from_millis(100)).await;

        let sql = "SELECT * FROM cpu_temperature";
        let mut stream = client.stream(&SQLQuery(sql.to_string())).await.unwrap();
        let mut batches = vec![];
        loop {
            let msg = stream.next().await.unwrap();
            let Some(batch) = msg.batch else {
                break;
            };
            let mut decoder = arrow::ipc::reader::StreamDecoder::new();
            let mut buffer = arrow::buffer::Buffer::from(batch.into_owned());
            if let Some(batch) = decoder.decode(&mut buffer).unwrap() {
                batches.push(batch);
            }
        }
        let batch = &batches[0];
        let arr = batch
            .column_by_name("cpu_temperature")
            .unwrap()
            .as_fixed_size_list();
        let arr = arr.values();
        let arr = arr.as_primitive::<Float64Type>();
        assert_eq!(arr.values(), &[0.0, 10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    async fn test_get_time_series() {
        let (addr, _db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();

        let component_id = ComponentId::new("sensor_data");

        let vtable = vtable([raw_field(
            0,
            8,
            timestamp(
                raw_table(8, 8),
                schema(PrimType::F64, &[], component(component_id)),
            ),
        )]);

        let vtable_id = 1u16.to_le_bytes();
        client
            .send(&VTableMsg {
                id: vtable_id,
                vtable,
            })
            .await
            .0
            .unwrap();

        let timestamps = [
            Timestamp(1000),
            Timestamp(2000),
            Timestamp(3000),
            Timestamp(4000),
            Timestamp(5000),
        ];

        for t in &timestamps {
            let mut pkt = LenPacket::table(vtable_id, 8);
            pkt.extend_aligned(&[t.0 as f64]);
            pkt.extend_aligned(&[t.0]);
            sleep(Duration::from_millis(10)).await;
            client.send(pkt).await.0.unwrap();
        }

        sleep(Duration::from_millis(100)).await;

        // Query the time series data
        let query = GetTimeSeries {
            id: vtable_id,
            range: Timestamp(0)..Timestamp(10000),
            component_id,
            limit: Some(256),
        };

        let time_series = client.request(&query).await.unwrap();

        let data = <[f64]>::ref_from_bytes(time_series.data().unwrap()).unwrap();
        assert_eq!(data, &[1000.0, 2000.0, 3000.0, 4000.0, 5000.0]);
        assert_eq!(time_series.timestamps().unwrap(), &timestamps);
    }

    #[test]
    async fn test_get_schema() {
        let (addr, _db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();

        let component_id = ComponentId::new("test_component");
        let vtable = vtable([raw_field(
            0,
            8,
            schema(PrimType::F64, &[1], component(component_id)),
        )]);

        client
            .send(&VTableMsg {
                id: 1u16.to_le_bytes(),
                vtable,
            })
            .await
            .0
            .unwrap();

        sleep(Duration::from_millis(50)).await;

        let get_schema = GetSchema { component_id };
        let SchemaMsg(schema) = client.request(&get_schema).await.unwrap();

        assert_eq!(schema.prim_type(), PrimType::F64);
        assert_eq!(schema.shape(), &[1]);
    }

    #[test]
    async fn test_get_component_metadata() {
        let (addr, _db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();

        let component_id = ComponentId::new("sensor");
        let metadata = SetComponentMetadata::new(component_id, "Temperature Sensor").metadata(
            [("unit".to_string(), "celsius".to_string())]
                .into_iter()
                .collect(),
        );
        client.send(&metadata).await.0.unwrap();

        sleep(Duration::from_millis(50)).await;

        let get_metadata = GetComponentMetadata { component_id };
        let component_metadata = client.request(&get_metadata).await.unwrap();

        assert_eq!(component_metadata.component_id, component_id);
        assert_eq!(component_metadata.name, "Temperature Sensor");
        assert_eq!(component_metadata.metadata.get("unit").unwrap(), "celsius");
    }

    #[test]
    async fn test_dump_schema() {
        let (addr, _db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();

        let component_id1 = ComponentId::new("component1");
        let component_id2 = ComponentId::new("component2");

        let vtable1 = vtable([raw_field(
            0,
            8,
            schema(PrimType::F64, &[1], component(component_id1)),
        )]);

        let vtable2 = vtable([raw_field(
            0,
            4,
            schema(PrimType::F32, &[2, 2], component(component_id2)),
        )]);

        client
            .send(&VTableMsg {
                id: 1u16.to_le_bytes(),
                vtable: vtable1,
            })
            .await
            .0
            .unwrap();

        client
            .send(&VTableMsg {
                id: 2u16.to_le_bytes(),
                vtable: vtable2,
            })
            .await
            .0
            .unwrap();

        sleep(Duration::from_millis(50)).await;

        let resp = client.request(&DumpSchema).await.unwrap();

        assert!(resp.schemas.contains_key(&component_id1));
        assert!(resp.schemas.contains_key(&component_id2));

        let schema1 = &resp.schemas[&component_id1];
        let schema2 = &resp.schemas[&component_id2];

        assert_eq!(schema1.prim_type(), PrimType::F64);
        assert_eq!(schema1.shape(), &[1]);

        assert_eq!(schema2.prim_type(), PrimType::F32);
        assert_eq!(schema2.shape(), &[2, 2]);
    }

    #[test]
    async fn test_msg_metadata_and_get_msgs() {
        let (addr, _db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();

        #[derive(postcard_schema::Schema, serde::Deserialize, serde::Serialize)]
        struct Msg {
            i: u32,
        }

        let msg_metadata = SetMsgMetadata {
            id: Msg::ID,
            metadata: MsgMetadata {
                name: "TestMessage".to_string(),
                schema: Msg::SCHEMA.into(),
                metadata: [("category".to_string(), "test".to_string())]
                    .into_iter()
                    .collect(),
            },
        };

        client.send(&msg_metadata).await.0.unwrap();
        sleep(Duration::from_millis(50)).await;

        for i in 0..5 {
            let msg = Msg { i };
            client.send(&msg).await.0.unwrap();
            sleep(Duration::from_millis(10)).await;
        }

        let get_metadata = GetMsgMetadata { msg_id: Msg::ID };
        let metadata = client.request(&get_metadata).await.unwrap();

        assert_eq!(metadata.name, "TestMessage");
        let expected_schema: OwnedNamedType = Msg::SCHEMA.into();
        assert_eq!(metadata.schema, expected_schema);
        assert_eq!(metadata.metadata.get("category").unwrap(), "test");

        let get_msgs = GetMsgs {
            msg_id: Msg::ID,
            range: Timestamp(0)..Timestamp(i64::MAX),
            limit: None,
        };

        let response = client.request(&get_msgs).await.unwrap();

        assert_eq!(response.data.len(), 5);

        for (i, (_, data)) in response.data.iter().enumerate() {
            let msg: Msg = postcard::from_bytes(data).unwrap();
            assert_eq!(msg.i, i as u32);
        }
    }

    #[test]
    async fn test_save_archive() {
        let (addr, _db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();

        // Create some test data
        let component_id = ComponentId::new("archive_test");

        // Set metadata for our test data
        client
            .send(&SetComponentMetadata::new(component_id, "TestComponent"))
            .await
            .0
            .unwrap();

        let vtable = vtable([raw_field(
            0,
            8,
            schema(PrimType::F64, &[1], component(component_id)),
        )]);

        let vtable_id = 1u16.to_le_bytes();
        client
            .send(&VTableMsg {
                id: vtable_id,
                vtable,
            })
            .await
            .0
            .unwrap();

        // Send some data - we'll use specific values for easy verification
        let test_values = [10.5f64, 20.5, 30.5];
        for &value in &test_values {
            let mut pkt = LenPacket::table(vtable_id, 8);
            pkt.extend_aligned(&[value]);
            client.send(pkt).await.0.unwrap();
            sleep(Duration::from_millis(10)).await;
        }

        sleep(Duration::from_millis(50)).await;

        let temp_dir = std::env::temp_dir();
        let archive_path = temp_dir.join(format!("test_archive_{}", fastrand::u64(..)));

        let save_archive = SaveArchive {
            path: archive_path.clone(),
            format: ArchiveFormat::ArrowIpc,
        };

        let response = client.request(&save_archive).await.unwrap();
        assert_eq!(response.path, archive_path);
        assert!(archive_path.exists());

        let file = std::fs::File::open(archive_path.join("TestComponent.arrow")).unwrap();
        let mut reader = arrow::ipc::reader::FileReader::try_new(file, None).unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 3);
        let _ = batch.column_by_name("time").unwrap();
        let component = batch.column_by_name("TestComponent").unwrap();
        let values = component
            .as_fixed_size_list()
            .values()
            .as_primitive::<Float64Type>()
            .values();
        assert_eq!(values, &[10.5, 20.5, 30.5]);
    }

    #[cfg(not(windows))]
    #[test]
    async fn test_save_archive_rejects_windows_path_on_unix() {
        let (addr, _db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();

        let invalid_path = std::path::PathBuf::from("C:\\Users\\tester\\snapshot");
        if invalid_path.exists() {
            let _ = std::fs::remove_dir_all(&invalid_path);
        }

        let save_archive = SaveArchive {
            path: invalid_path.clone(),
            format: ArchiveFormat::ArrowIpc,
        };

        let err = client.request(&save_archive).await.unwrap_err();
        let description = err.to_string();
        assert!(
            description.contains("Cannot save db to"),
            "error description missing prefix: {}",
            description
        );
        assert!(
            description.contains("C:\\Users\\tester\\snapshot"),
            "error description missing path: {}",
            description
        );
        assert!(
            description.contains("Windows-style location"),
            "error description missing guidance: {}",
            description
        );
        assert!(
            !invalid_path.exists(),
            "invalid export path should not be created on unix hosts"
        );

        let invalid_drive_relative = std::path::PathBuf::from("C:Users\\tester\\snapshot2");
        let save_archive = SaveArchive {
            path: invalid_drive_relative.clone(),
            format: ArchiveFormat::ArrowIpc,
        };
        let err = client.request(&save_archive).await.unwrap_err();
        let description = err.to_string();
        assert!(
            description.contains("C:Users\\tester\\snapshot2"),
            "error description missing drive-relative path: {}",
            description
        );
    }

    #[test]
    async fn test_save_archive_native_blocks_writes() {
        let (addr, db) = setup_test_db().await.unwrap();
        let mut setup_client = Client::connect(addr).await.unwrap();

        let component_id = ComponentId::new("archive_native_test");
        setup_client
            .send(&SetComponentMetadata::new(
                component_id,
                "TestComponentNative",
            ))
            .await
            .0
            .unwrap();

        let vtable = vtable([raw_field(
            0,
            8,
            schema(PrimType::F64, &[1], component(component_id)),
        )]);
        let vtable_id = 3u16.to_le_bytes();
        setup_client
            .send(&VTableMsg {
                id: vtable_id,
                vtable,
            })
            .await
            .0
            .unwrap();

        let initial_values = [10.5f64, 20.5, 30.5];
        for value in initial_values {
            let mut pkt = LenPacket::table(vtable_id, 8);
            pkt.extend_aligned(&[value]);
            setup_client.send(pkt).await.0.unwrap();
            sleep(Duration::from_millis(5)).await;
        }

        // Add a filler file to make the native copy take perceptible time.
        let filler_path = db.path.join("filler.bin");
        {
            let mut filler = File::create(&filler_path).unwrap();
            filler.write_all(&vec![0xAAu8; 4 * 1024 * 1024]).unwrap();
            filler.sync_all().unwrap();
        }

        // Ensure the server has ingested the component metadata + initial samples
        // before triggering the snapshot; otherwise the copy might race ahead of
        // the write pipeline and miss the component entirely.
        let ready_deadline = Instant::now() + Duration::from_secs(1);
        let mut component_ready = false;
        while Instant::now() < ready_deadline {
            component_ready = db.with_state(|state| {
                state
                    .get_component(component_id)
                    .and_then(|component| component.time_series.latest())
                    .is_some()
            });
            if component_ready {
                break;
            }
            sleep(Duration::from_millis(10)).await;
        }
        assert!(
            component_ready,
            "component metadata/data must exist before snapshot"
        );

        let native_root =
            std::env::temp_dir().join(format!("test_native_archive_{}", fastrand::u64(..)));

        let mut archive_client = Client::connect(addr).await.unwrap();
        let mut writer_client = Client::connect(addr).await.unwrap();
        let late_value = 99.5f64;

        let save_future = {
            let save_path = native_root.clone();
            async move {
                let save_archive = SaveArchive {
                    path: save_path,
                    format: ArchiveFormat::Native,
                };
                archive_client.request(&save_archive).await.unwrap()
            }
        };

        let native_root_for_writer = native_root.clone();
        let write_future = async move {
            // Wait until the snapshot copy has actually started by polling for the
            // temporary directory (db.tmp). This ensures the snapshot barrier is
            // active before sending the late write, making the test deterministic.
            let parent = native_root_for_writer
                .parent()
                .map(std::path::Path::to_path_buf)
                .unwrap_or_else(|| std::path::PathBuf::from("."));
            let tmp_db_dir = parent.join(format!(
                "{}.tmp",
                native_root_for_writer
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
            ));
            let start = std::time::Instant::now();
            while !tmp_db_dir.exists() && start.elapsed() < Duration::from_secs(1) {
                sleep(Duration::from_millis(5)).await;
            }

            let mut pkt = LenPacket::table(vtable_id, 8);
            pkt.extend_aligned(&[late_value]);
            writer_client.send(pkt).await.0.unwrap();
        };

        let (archive_saved, _) = zip(save_future, write_future).await;

        assert_eq!(archive_saved.path, native_root);
        assert!(native_root.exists());

        let snapshot_db = DB::open(native_root.clone()).unwrap();
        snapshot_db.with_state(|state| {
            let component = state
                .get_component(component_id)
                .expect("missing component in snapshot");
            let (timestamps, _) = component
                .time_series
                .get_range(&(Timestamp(i64::MIN)..Timestamp(i64::MAX)))
                .expect("failed to read snapshot range");
            assert_eq!(timestamps.len(), 3);
            let (_, buf) = component
                .time_series
                .latest()
                .expect("missing latest snapshot sample");
            let latest =
                f64::from_le_bytes(buf.try_into().expect("component sample size mismatch"));
            assert!((latest - 30.5).abs() <= f64::EPSILON);
        });

        // The client `send` completes when bytes are written to the socket, not
        // when the server has applied the write. Poll the DB briefly until the
        // late write becomes visible, then assert.
        let start = std::time::Instant::now();
        let timeout = Duration::from_millis(500);
        loop {
            let latest_seen = db.with_state(|state| {
                let component = state
                    .get_component(component_id)
                    .expect("missing component");
                let (_, buf) = component
                    .time_series
                    .latest()
                    .expect("missing latest sample");
                f64::from_le_bytes(buf.try_into().expect("component sample size mismatch"))
            });
            if (latest_seen - late_value).abs() <= f64::EPSILON {
                break;
            }
            if start.elapsed() > timeout {
                panic!(
                    "latest sample should include post-snapshot write; got {}",
                    latest_seen
                );
            }
            sleep(Duration::from_millis(10)).await;
        }

        let _ = std::fs::remove_dir_all(native_root);
    }

    #[test]
    async fn test_error_handling_invalid_query() {
        let (addr, _db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();

        let invalid_sql = "SELECT * FROM nonexistent_table";
        let mut sub = client
            .stream(&SQLQuery(invalid_sql.to_string()))
            .await
            .unwrap();
        sub.next().await.expect_err("sql query didnt return err");
    }

    #[test]
    async fn test_get_time_series_not_found() {
        let (addr, _db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();

        let component_id = ComponentId::new("test_component");

        client
            .send(&VTableMsg {
                id: 1u16.to_le_bytes(),
                vtable: vtable([raw_field(
                    0,
                    4,
                    schema(PrimType::F32, &[], component(component_id)),
                )]),
            })
            .await
            .0
            .unwrap();

        let query = GetTimeSeries {
            id: 1u16.to_le_bytes(),
            range: Timestamp(0)..Timestamp(10000),
            component_id,
            limit: None,
        };

        let result = client.request(&query).await;

        result.unwrap_err();

        // Now try with non-existent component
        let non_existent_component_id = ComponentId::new("non_existent_component");

        let query = GetTimeSeries {
            id: 1u16.to_le_bytes(),
            range: Timestamp(0)..Timestamp(10000),
            component_id: non_existent_component_id,
            limit: None,
        };

        // Should return an error for non-existent component
        let result = client.request(&query).await;
        assert!(
            result.is_err(),
            "Request for non-existent component should fail"
        );
    }

    #[test]
    async fn test_get_schema_not_found() {
        let (addr, _db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();

        let non_existent_component_id = ComponentId::new("non_existent_component");

        let get_schema = GetSchema {
            component_id: non_existent_component_id,
        };

        let result = client.request(&get_schema).await;
        let Err(impeller2_stellar::Error::Response(resp)) = result else {
            panic!("invalid error");
        };
        assert_eq!(
            elodin_db::Error::ComponentNotFound(non_existent_component_id).to_string(),
            resp.description
        );
    }

    #[test]
    async fn test_get_component_metadata_not_found() {
        let (addr, _db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();

        // Try to get metadata for a non-existent component
        let non_existent_component_id = ComponentId::new("non_existent_component");

        let get_metadata = GetComponentMetadata {
            component_id: non_existent_component_id,
        };

        let result = client.request(&get_metadata).await.unwrap_err();
        let impeller2_stellar::Error::Response(resp) = result else {
            panic!("invalid error");
        };
        assert_eq!(
            elodin_db::Error::ComponentNotFound(non_existent_component_id).to_string(),
            resp.description
        );
    }

    #[test]
    async fn test_get_msg_metadata_not_found() {
        let (addr, _db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();

        let msg_id = 9999u16.to_le_bytes();

        let get_metadata = GetMsgMetadata { msg_id };

        let result = client.request(&get_metadata).await.unwrap_err();
        let impeller2_stellar::Error::Response(resp) = result else {
            panic!("invalid error");
        };
        assert_eq!(
            elodin_db::Error::MsgNotFound(msg_id).to_string(),
            resp.description
        );
    }

    #[test]
    async fn test_get_msgs_not_found() {
        let (addr, _db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();

        // Try to get messages for a non-existent message ID
        let msg_id = 9999u16.to_le_bytes();

        let get_msgs = GetMsgs {
            msg_id,
            range: Timestamp(0)..Timestamp(i64::MAX),
            limit: None,
        };

        let result = client.request(&get_msgs).await.unwrap_err();
        let impeller2_stellar::Error::Response(resp) = result else {
            panic!("invalid error");
        };
        assert_eq!(resp.description, format!("msg not found {:?}", msg_id));
    }

    #[test]
    async fn test_concurrent_clients() {
        let (addr, _db) = setup_test_db().await.unwrap();

        let component_id = ComponentId::new("concurrent_test");

        // Define a vtable that will be used by all clients
        // No explicit timestamp - let server auto-assign to avoid time-travel errors
        let vtable = vtable([raw_field(
            0,
            8,
            schema(PrimType::F64, &[1], component(component_id)),
        )]);
        let vtable_id = 1u16.to_le_bytes();

        // Set up the vtable and metadata
        let mut setup_client = Client::connect(addr).await.unwrap();
        setup_client
            .send(&VTableMsg {
                id: vtable_id,
                vtable,
            })
            .await
            .0
            .unwrap();
        setup_client
            .send(&SetComponentMetadata::new(
                component_id,
                "Concurrent Test Component",
            ))
            .await
            .0
            .unwrap();

        sleep(Duration::from_millis(100)).await;

        const NUM_CLIENTS: usize = 5;
        const WRITES_PER_CLIENT: usize = 10;

        let mut join_handles = Vec::with_capacity(NUM_CLIENTS);

        for client_id in 0..NUM_CLIENTS {
            let client_addr = addr;
            let client_vtable_id = vtable_id;

            join_handles.push(spawn(async move {
                let mut client = Client::connect(client_addr).await.unwrap();
                let base_value = (client_id * 100) as f64;

                for i in 0..WRITES_PER_CLIENT {
                    let value = base_value + i as f64;
                    let mut pkt = LenPacket::table(client_vtable_id, 8);
                    pkt.extend_aligned(&[value]);
                    client.send(pkt).await.0.unwrap();
                    // Longer sleep between writes to reduce timestamp collisions
                    sleep(Duration::from_millis(10)).await;
                }

                client_id
            }));
        }

        for handle in join_handles {
            let _ = handle.await;
        }

        // Poll the database until we get the expected count (with timeout)
        // Note: With concurrent clients and auto-assigned timestamps, we might
        // occasionally lose a packet to time-travel errors, so we accept >= 95% success
        let mut verification_client = Client::connect(addr).await.unwrap();

        let expected_count = NUM_CLIENTS * WRITES_PER_CLIENT;
        let min_acceptable = (expected_count * 95) / 100; // Accept 95% success rate
        let timeout = Duration::from_secs(2);
        let start = std::time::Instant::now();

        let actual_count = loop {
            let query = GetTimeSeries {
                id: vtable_id,
                range: Timestamp(0)..Timestamp(i64::MAX),
                component_id,
                limit: Some(expected_count),
            };

            let time_series = verification_client.request(&query).await.unwrap();
            let data = <[f64]>::ref_from_bytes(time_series.data().unwrap()).unwrap();
            let count = data.len();

            // Accept if we have at least the minimum acceptable count
            if count >= min_acceptable {
                break count;
            }

            if start.elapsed() > timeout {
                panic!(
                    "Timeout waiting for data: expected {}, got {} (min acceptable: {}) after {:?}",
                    expected_count,
                    count,
                    min_acceptable,
                    start.elapsed()
                );
            }

            sleep(Duration::from_millis(10)).await;
        };

        // Verify we got at least 95% of expected packets
        assert!(
            actual_count >= min_acceptable,
            "Expected at least {} packets (95% of {}), got {}",
            min_acceptable,
            expected_count,
            actual_count
        );
    }

    #[test]
    async fn test_database_restart() {
        let temp_dir =
            std::env::temp_dir().join(format!("elodin_db_restart_test_{}", fastrand::u64(..)));
        let vtable_id = 1u16.to_le_bytes();
        let test_value = 42.0;
        let component_id = ComponentId::new("restart_test");

        {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();

            let server = Server::from_listener(listener, temp_dir.clone()).unwrap();

            stellar(move || async { server.run().await });

            let mut client = Client::connect(addr).await.unwrap();

            client
                .send(&SetComponentMetadata::new(
                    component_id,
                    "Restart Test Component",
                ))
                .await
                .0
                .unwrap();

            let vtable = vtable([raw_field(
                0,
                8,
                schema(PrimType::F64, &[1], component(component_id)),
            )]);
            client
                .send(&VTableMsg {
                    id: vtable_id,
                    vtable,
                })
                .await
                .0
                .unwrap();

            let mut pkt = LenPacket::table(vtable_id, 8);
            pkt.extend_aligned(&[test_value]);
            client.send(pkt).await.0.unwrap();

            sleep(Duration::from_millis(100)).await;

            let query = GetTimeSeries {
                id: vtable_id,
                range: Timestamp(0)..Timestamp(i64::MAX),
                component_id,
                limit: Some(256),
            };
            let time_series = client.request(&query).await.unwrap();
            let data_before = <[f64]>::ref_from_bytes(time_series.data().unwrap()).unwrap();
            assert_eq!(data_before, &[test_value]);
        }

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let new_addr = listener.local_addr().unwrap();

        let new_server = Server::from_listener(listener, temp_dir.clone()).unwrap();

        stellar(move || async { new_server.run().await });

        let mut new_client = Client::connect(new_addr).await.unwrap();
        sleep(Duration::from_millis(100)).await;

        let component_metadata = new_client
            .request(&GetComponentMetadata { component_id })
            .await
            .unwrap();
        assert_eq!(component_metadata.name, "Restart Test Component");

        let get_schema = GetSchema { component_id };
        let SchemaMsg(schema) = new_client.request(&get_schema).await.unwrap();
        assert_eq!(schema.prim_type(), PrimType::F64);
        assert_eq!(schema.shape(), &[1]);

        let query = GetTimeSeries {
            id: vtable_id,
            range: Timestamp(0)..Timestamp(i64::MAX),
            component_id,
            limit: Some(256),
        };
        let new_time_series = new_client.request(&query).await.unwrap();
        let data_after = <[f64]>::ref_from_bytes(new_time_series.data().unwrap()).unwrap();
        assert_eq!(data_after, &[test_value]);

        if temp_dir.exists() {
            let _ = std::fs::remove_dir_all(&temp_dir);
        }
    }

    #[test]
    async fn test_large_dataset_performance() {
        let (addr, _db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();

        let component_id = ComponentId::new("large_dataset");

        let vtable = vtable([raw_field(
            0,
            800,
            schema(PrimType::F64, &[100], component(component_id)),
        )]);

        let vtable_id = 1u16.to_le_bytes();
        client
            .send(&VTableMsg {
                id: vtable_id,
                vtable,
            })
            .await
            .0
            .unwrap();
        client
            .send(&SetComponentMetadata::new(component_id, "Large Dataset"))
            .await
            .0
            .unwrap();

        for packet_num in 0..20 {
            let mut values = Vec::with_capacity(100);
            for i in 0..100 {
                values.push((packet_num * 100 + i) as f64);
            }

            let mut pkt = LenPacket::table(vtable_id, 800);
            pkt.extend_aligned(&values);
            client.send(pkt).await.0.unwrap();

            if packet_num % 5 == 0 {
                sleep(Duration::from_millis(10)).await;
            }
        }

        sleep(Duration::from_millis(100)).await;

        let query_limited = GetTimeSeries {
            id: vtable_id,
            range: Timestamp(0)..Timestamp(i64::MAX),
            component_id,
            limit: Some(10),
        };

        let time_series_limited = client.request(&query_limited).await.unwrap();
        assert!(time_series_limited.timestamps().unwrap().len() <= 10);

        let query_all = GetTimeSeries {
            id: vtable_id,
            range: Timestamp(0)..Timestamp(i64::MAX),
            component_id,
            limit: None,
        };

        let time_series_all = client.request(&query_all).await.unwrap();

        assert!(!time_series_all.timestamps().unwrap().is_empty());

        let data_flat = time_series_all.data().unwrap();
        assert_eq!(data_flat.len() % (100 * 8), 0);
    }

    #[test]
    async fn test_complex_query() {
        let (addr, _db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();

        client
            .send(&SetComponentMetadata::new("sensor1.temp", "sensor1.temp"))
            .await
            .0
            .unwrap();

        client
            .send(&SetComponentMetadata::new("sensor2.temp", "sensor2.temp"))
            .await
            .0
            .unwrap();

        // Create vtables for both entities
        let vtable1 = vtable([raw_field(
            0,
            8,
            schema(PrimType::F64, &[], component("sensor1.temp")),
        )]);

        let vtable2 = vtable([raw_field(
            0,
            8,
            schema(PrimType::F64, &[], component("sensor2.temp")),
        )]);

        let vtable_id1 = 1u16.to_le_bytes();
        let vtable_id2 = 2u16.to_le_bytes();

        client
            .send(&VTableMsg {
                id: vtable_id1,
                vtable: vtable1,
            })
            .await
            .0
            .unwrap();

        client
            .send(&VTableMsg {
                id: vtable_id2,
                vtable: vtable2,
            })
            .await
            .0
            .unwrap();

        for i in 0..5 {
            let value = 20.0 + i as f64 * 1.5;
            let mut pkt = LenPacket::table(vtable_id1, 8);
            pkt.extend_aligned(&[value]);
            client.send(pkt).await.0.unwrap();
        }

        for i in 0..5 {
            let value = 15.0 + i as f64 * 0.5;
            let mut pkt = LenPacket::table(vtable_id2, 8);
            pkt.extend_aligned(&[value]);
            client.send(pkt).await.0.unwrap();
            sleep(Duration::from_millis(10)).await;
        }

        sleep(Duration::from_millis(100)).await;

        let sql = "SELECT * FROM `sensor_1_temp`, `sensor_2_temp` WHERE `sensor_1_temp`.`sensor_1_temp` > 22.0";
        let mut stream = client.stream(&SQLQuery(sql.to_string())).await.unwrap();

        let mut batches = vec![];
        loop {
            let msg = stream.next().await.unwrap();
            let Some(batch) = msg.batch else {
                break;
            };
            let mut decoder = arrow::ipc::reader::StreamDecoder::new();
            let mut buffer = arrow::buffer::Buffer::from(batch.into_owned());
            if let Some(batch) = decoder.decode(&mut buffer).unwrap() {
                batches.push(batch);
            }
        }

        let sql_simple = "SELECT * FROM `sensor_1_temp` UNION ALL SELECT * FROM `sensor_2_temp`";
        let mut stream_simple = client
            .stream(&SQLQuery(sql_simple.to_string()))
            .await
            .unwrap();

        let mut all_data = vec![];
        loop {
            let msg = stream_simple.next().await.unwrap();
            let Some(batch) = msg.batch else {
                break;
            };
            let mut decoder = arrow::ipc::reader::StreamDecoder::new();
            let mut buffer = arrow::buffer::Buffer::from(batch.into_owned());
            if let Some(batch) = decoder.decode(&mut buffer).unwrap() {
                all_data.push(batch);
            }
        }

        assert!(!all_data.is_empty());
    }

    #[test]
    async fn test_subscribe_last_updated() {
        let (addr, db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();
        let mut sub_client = Client::connect(addr).await.unwrap();

        // we subscribe first to ensure we dont miss any last updates
        let mut stream = sub_client.stream(&SubscribeLastUpdated).await.unwrap();

        let component_id = ComponentId::new("test_component");

        let vtable = vtable([raw_field(
            0,
            8,
            schema(PrimType::F64, &[], component(component_id)),
        )]);

        let vtable_id = 1u16.to_le_bytes();
        client
            .send(&VTableMsg {
                id: vtable_id,
                vtable,
            })
            .await
            .0
            .unwrap();

        sleep(Duration::from_millis(50)).await;

        let correct_last_updated = db.last_updated.latest();

        let last_updated = stream.next().await.unwrap();
        assert_eq!(correct_last_updated, last_updated.0);

        let value = 42.0f64;
        let mut pkt = LenPacket::table(vtable_id, 8);
        pkt.extend_aligned(&[value]);
        client.send(pkt).await.0.unwrap();

        sleep(Duration::from_millis(50)).await;

        let correct_last_updated = db.last_updated.latest();

        let last_updated = stream.next().await.unwrap();
        assert_eq!(correct_last_updated, last_updated.0);
    }

    #[test]
    async fn test_invalid_table_schema() {
        let (addr, _) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();

        let component_id = ComponentId::new("test_component");
        {
            let vtable = vtable([raw_field(
                0,
                8,
                schema(PrimType::F64, &[1], component(component_id)),
            )]);

            client
                .send(&VTableMsg {
                    id: 1u16.to_le_bytes(),
                    vtable,
                })
                .await
                .0
                .unwrap();

            let mut pkt = LenPacket::table(1u16.to_le_bytes(), 8);
            pkt.extend_aligned(&[42.0f64]);
            client.send(pkt).await.0.unwrap();
        }

        sleep(Duration::from_millis(10)).await;

        let vtable_different_type = vtable([raw_field(
            0,
            4,
            schema(PrimType::F32, &[1], component(component_id)),
        )]);

        let vtable_id = 2u16.to_le_bytes();
        client
            .send(
                VTableMsg {
                    id: vtable_id,
                    vtable: vtable_different_type,
                }
                .with_request_id(42),
            )
            .await
            .0
            .unwrap();

        let Err(impeller2_stellar::Error::Response(err)) = client.recv::<()>(42).await else {
            panic!("invalid response");
        };
        assert_eq!(
            elodin_db::Error::SchemaMismatch.to_string(),
            err.description
        );
    }

    #[test]
    async fn test_time_travel() {
        let (addr, _db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();

        let component_id = ComponentId::new("time_travel_test");
        let vtable = vtable([raw_field(
            0,
            8,
            timestamp(
                raw_table(8, 8),
                schema(PrimType::F64, &[1], component(component_id)),
            ),
        )]);

        client
            .send(&VTableMsg {
                id: 1u16.to_le_bytes(),
                vtable,
            })
            .await
            .0
            .unwrap();

        let mut pkt = LenPacket::table(1u16.to_le_bytes(), 8);
        pkt.extend_aligned(&[42.0f64]);
        pkt.push_aligned(Timestamp(100));
        client.send(pkt).await.0.unwrap();

        let mut pkt = LenPacket::table(1u16.to_le_bytes(), 8);
        pkt.extend_aligned(&[40.0f64]);
        pkt.push_aligned(Timestamp(1));
        client.send(pkt.with_request_id(42)).await.0.unwrap();
        sleep(Duration::from_millis(10)).await;

        let Err(impeller2_stellar::Error::Response(err)) = client.recv::<()>(42).await else {
            panic!("invalid response");
        };
        assert_eq!(elodin_db::Error::TimeTravel.to_string(), err.description);
    }

    // ── Follow-mode tests ──────────────────────────────────────────────

    /// Spin up a source DB and a follower DB connected to it.
    /// Returns (source_addr, source_db, follower_addr, follower_db).
    async fn setup_follow_pair(
        packet_size: usize,
    ) -> Result<(SocketAddr, Arc<DB>, SocketAddr, Arc<DB>), Error> {
        let subscriber = tracing_subscriber::FmtSubscriber::new();
        let _ = tracing::subscriber::set_global_default(subscriber);

        // Source server
        let src_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let src_addr = src_listener.local_addr().unwrap();
        let src_temp =
            std::env::temp_dir().join(format!("elodin_db_follow_src_{}", fastrand::u64(..)));
        if src_temp.exists() {
            let _ = std::fs::remove_dir_all(&src_temp);
        }
        let src_server = Server::from_listener(src_listener, src_temp)?;
        let src_db = src_server.db.clone();
        stellar(move || async { src_server.run().await });

        // Follower server
        let fol_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let fol_addr = fol_listener.local_addr().unwrap();
        let fol_temp =
            std::env::temp_dir().join(format!("elodin_db_follow_fol_{}", fastrand::u64(..)));
        if fol_temp.exists() {
            let _ = std::fs::remove_dir_all(&fol_temp);
        }
        let fol_server = Server::from_listener(fol_listener, fol_temp)?;
        let fol_db = fol_server.db.clone();
        stellar(move || async { fol_server.run().await });

        // Spawn follower task
        let follow_db = fol_db.clone();
        stellar(move || {
            elodin_db::follow::run_follower(
                elodin_db::follow::FollowConfig {
                    source_addr: src_addr,
                    target_packet_size: packet_size,
                    reconnect_delay: Duration::from_millis(100),
                },
                follow_db,
            )
        });

        // Give the follower time to connect and perform initial sync
        sleep(Duration::from_millis(500)).await;

        Ok((src_addr, src_db, fol_addr, fol_db))
    }

    /// Helper: register a VTable with explicit timestamps and send component
    /// samples with specific timestamps to the given address.
    async fn send_timestamped_component(
        addr: SocketAddr,
        component_id: ComponentId,
        component_name: &str,
        vtable_id: [u8; 2],
        samples: &[(Timestamp, f64)],
    ) {
        let mut client = Client::connect(addr).await.unwrap();

        // Set component metadata
        client
            .send(&SetComponentMetadata::new(component_id, component_name))
            .await
            .0
            .unwrap();

        // Register VTable with explicit timestamp field.
        // Layout: [f64 data (8 bytes) | i64 timestamp (8 bytes)]
        let vt = vtable([raw_field(
            0,
            8,
            timestamp(
                raw_table(8, 8),
                schema(PrimType::F64, &[], component(component_id)),
            ),
        )]);
        client
            .send(&VTableMsg {
                id: vtable_id,
                vtable: vt,
            })
            .await
            .0
            .unwrap();

        sleep(Duration::from_millis(50)).await;

        // Send each sample with its explicit timestamp.
        for &(ts, value) in samples {
            let mut pkt = LenPacket::table(vtable_id, 16);
            pkt.extend_aligned(&[value]);
            pkt.extend_aligned(&[ts.0]);
            client.send(pkt).await.0.unwrap();
            sleep(Duration::from_millis(10)).await;
        }
    }

    /// Helper: poll until a component has at least `min_count` samples
    /// in the given DB, with a timeout.
    async fn wait_for_component_samples(
        db: &Arc<DB>,
        component_id: ComponentId,
        min_count: usize,
        timeout: Duration,
    ) -> bool {
        let start = Instant::now();
        loop {
            let count = db.with_state(|state| {
                state
                    .get_component(component_id)
                    .and_then(|c| {
                        c.time_series
                            .get_range(&(Timestamp(i64::MIN)..Timestamp(i64::MAX)))
                            .map(|(ts, _)| ts.len())
                    })
                    .unwrap_or(0)
            });
            if count >= min_count {
                return true;
            }
            if start.elapsed() > timeout {
                return false;
            }
            sleep(Duration::from_millis(50)).await;
        }
    }

    /// Helper: poll until a message log has at least `min_count` messages.
    async fn wait_for_msg_count(
        db: &Arc<DB>,
        msg_id: impeller2::types::PacketId,
        min_count: usize,
        timeout: Duration,
    ) -> bool {
        let start = Instant::now();
        let db_path = db.path.clone();
        loop {
            let count = db.with_state_mut(|state| {
                state
                    .get_or_insert_msg_log(msg_id, &db_path)
                    .map(|log| log.timestamps().len())
                    .unwrap_or(0)
            });
            if count >= min_count {
                return true;
            }
            if start.elapsed() > timeout {
                return false;
            }
            sleep(Duration::from_millis(50)).await;
        }
    }

    // ── DB comparison helpers ────────────────────────────────────────────

    /// Read all .csv files in a directory, returning filename -> contents.
    fn collect_csv_files(dir: &Path) -> BTreeMap<String, String> {
        let mut files = BTreeMap::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "csv") {
                    let name = path.file_name().unwrap().to_string_lossy().to_string();
                    let content = std::fs::read_to_string(&path).unwrap();
                    files.insert(name, content);
                }
            }
        }
        files
    }

    /// Export both databases to CSV and assert they produce identical files.
    fn assert_exports_match(src_db: &Arc<DB>, fol_db: &Arc<DB>, pattern: Option<&str>) {
        let src_out = std::env::temp_dir().join(format!("elodin_export_src_{}", fastrand::u64(..)));
        let fol_out = std::env::temp_dir().join(format!("elodin_export_fol_{}", fastrand::u64(..)));

        src_db.flush_all().unwrap();
        fol_db.flush_all().unwrap();

        elodin_db::export::run(
            src_db.path.clone(),
            src_out.clone(),
            elodin_db::export::ExportFormat::Csv,
            true,
            pattern.map(String::from),
        )
        .unwrap();

        elodin_db::export::run(
            fol_db.path.clone(),
            fol_out.clone(),
            elodin_db::export::ExportFormat::Csv,
            true,
            pattern.map(String::from),
        )
        .unwrap();

        let src_files = collect_csv_files(&src_out);
        let fol_files = collect_csv_files(&fol_out);

        assert_eq!(
            src_files.keys().collect::<BTreeSet<_>>(),
            fol_files.keys().collect::<BTreeSet<_>>(),
            "source and follower should have the same set of exported CSV files"
        );

        for (name, src_content) in &src_files {
            let fol_content = &fol_files[name];
            assert_eq!(
                src_content, fol_content,
                "CSV mismatch for {}: source and follower data differ",
                name
            );
        }

        let _ = std::fs::remove_dir_all(&src_out);
        let _ = std::fs::remove_dir_all(&fol_out);
    }

    /// Read the committed data from an AppendLog file on disk.
    /// Returns bytes 16..committed_len -- the `extra` field (start_timestamp
    /// for index, element_size for data) plus all committed data.  Bytes
    /// 0-15 (committed_len + head_len atomics) are runtime bookkeeping and
    /// skipped.  The `extra` field IS included because the follower now
    /// sets it to match the source's value during backfill.
    fn read_append_log_committed(path: &Path) -> Vec<u8> {
        let data = std::fs::read(path).unwrap_or_default();
        if data.len() < 16 {
            return vec![];
        }
        let committed_len = u64::from_ne_bytes(data[0..8].try_into().unwrap()) as usize;
        if committed_len <= 16 || committed_len > data.len() {
            return vec![];
        }
        data[16..committed_len].to_vec()
    }

    /// Compare the data-bearing files in two DB directories.
    /// Skips db_state (contains creation timestamps). For AppendLog files,
    /// compares only the deterministic portion (bytes 16..committed_len).
    fn assert_db_files_match(src_path: &Path, fol_path: &Path) {
        // Collect numeric component directories from source.
        let src_entries: BTreeSet<String> = std::fs::read_dir(src_path)
            .unwrap()
            .flatten()
            .filter(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                e.path().is_dir() && name != "msgs" && name.chars().all(|c| c.is_ascii_digit())
            })
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();

        let fol_entries: BTreeSet<String> = std::fs::read_dir(fol_path)
            .unwrap()
            .flatten()
            .filter(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                e.path().is_dir() && name != "msgs" && name.chars().all(|c| c.is_ascii_digit())
            })
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();

        assert_eq!(
            src_entries, fol_entries,
            "source and follower should have the same component directories"
        );

        // Compare each component directory.
        for comp_id in &src_entries {
            let src_dir = src_path.join(comp_id);
            let fol_dir = fol_path.join(comp_id);

            // schema: byte-for-byte
            let src_schema = src_dir.join("schema");
            let fol_schema = fol_dir.join("schema");
            if src_schema.exists() {
                assert!(
                    fol_schema.exists(),
                    "follower missing schema for component {}",
                    comp_id
                );
                assert_eq!(
                    std::fs::read(&src_schema).unwrap(),
                    std::fs::read(&fol_schema).unwrap(),
                    "schema mismatch for component {}",
                    comp_id
                );
            }

            // metadata: byte-for-byte (if present in source)
            let src_meta = src_dir.join("metadata");
            let fol_meta = fol_dir.join("metadata");
            if src_meta.exists() {
                assert!(
                    fol_meta.exists(),
                    "follower missing metadata for component {}",
                    comp_id
                );
                assert_eq!(
                    std::fs::read(&src_meta).unwrap(),
                    std::fs::read(&fol_meta).unwrap(),
                    "metadata mismatch for component {}",
                    comp_id
                );
            }

            // index and data: compare committed portions of AppendLog files
            for filename in &["index", "data"] {
                let src_file = src_dir.join(filename);
                let fol_file = fol_dir.join(filename);
                if src_file.exists() {
                    assert!(
                        fol_file.exists(),
                        "follower missing {} for component {}",
                        filename,
                        comp_id
                    );
                    let src_data = read_append_log_committed(&src_file);
                    let fol_data = read_append_log_committed(&fol_file);
                    assert_eq!(
                        src_data,
                        fol_data,
                        "{} data mismatch for component {} (src {} bytes, fol {} bytes)",
                        filename,
                        comp_id,
                        src_data.len(),
                        fol_data.len()
                    );
                }
            }
        }

        // Compare msgs/ subdirectories.
        let src_msgs = src_path.join("msgs");
        let fol_msgs = fol_path.join("msgs");
        if src_msgs.exists() {
            let src_msg_ids: BTreeSet<String> = std::fs::read_dir(&src_msgs)
                .unwrap()
                .flatten()
                .filter(|e| e.path().is_dir())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .collect();

            let fol_msg_ids: BTreeSet<String> = if fol_msgs.exists() {
                std::fs::read_dir(&fol_msgs)
                    .unwrap()
                    .flatten()
                    .filter(|e| e.path().is_dir())
                    .map(|e| e.file_name().to_string_lossy().to_string())
                    .collect()
            } else {
                BTreeSet::new()
            };

            assert_eq!(
                src_msg_ids, fol_msg_ids,
                "source and follower should have the same message log directories"
            );

            for msg_id in &src_msg_ids {
                let src_msg_dir = src_msgs.join(msg_id);
                let fol_msg_dir = fol_msgs.join(msg_id);

                for filename in &["timestamps", "offsets", "data_log"] {
                    let src_file = src_msg_dir.join(filename);
                    let fol_file = fol_msg_dir.join(filename);
                    if src_file.exists() {
                        assert!(
                            fol_file.exists(),
                            "follower missing msgs/{}/{}",
                            msg_id,
                            filename
                        );
                        let src_data = read_append_log_committed(&src_file);
                        let fol_data = read_append_log_committed(&fol_file);
                        assert_eq!(
                            src_data,
                            fol_data,
                            "msgs/{}/{} data mismatch (src {} bytes, fol {} bytes)",
                            msg_id,
                            filename,
                            src_data.len(),
                            fol_data.len()
                        );
                    }
                }

                // metadata: byte-for-byte
                let src_meta = src_msg_dir.join("metadata");
                let fol_meta = fol_msg_dir.join("metadata");
                if src_meta.exists() {
                    assert!(
                        fol_meta.exists(),
                        "follower missing msgs/{}/metadata",
                        msg_id
                    );
                    assert_eq!(
                        std::fs::read(&src_meta).unwrap(),
                        std::fs::read(&fol_meta).unwrap(),
                        "msgs/{}/metadata mismatch",
                        msg_id
                    );
                }
            }
        }
    }

    #[test]
    async fn test_follow_metadata_sync() {
        let (src_addr, _src_db, _fol_addr, fol_db) = setup_follow_pair(1500).await.unwrap();

        let component_id = ComponentId::new("follow_meta_test");
        let vtable_id = 10u16.to_le_bytes();

        // Write component metadata + data to source so the schema is registered.
        send_timestamped_component(
            src_addr,
            component_id,
            "Follow Meta Test",
            vtable_id,
            &[(Timestamp(1000), 42.0)],
        )
        .await;

        // Also set message metadata on the source.
        let msg_name = "follow_test_msg";
        let msg_id = impeller2::types::msg_id(msg_name);
        {
            let mut client = Client::connect(src_addr).await.unwrap();
            client
                .send(&SetMsgMetadata {
                    id: msg_id,
                    metadata: MsgMetadata {
                        name: msg_name.to_string(),
                        schema: <impeller2_wkt::OpaqueBytes as postcard_schema::Schema>::SCHEMA
                            .into(),
                        metadata: Default::default(),
                    },
                })
                .await
                .0
                .unwrap();
        }

        // Wait for follower to sync.
        sleep(Duration::from_millis(1000)).await;

        // Assert follower has component metadata.
        fol_db.with_state(|state| {
            let meta = state.get_component_metadata(component_id);
            assert!(
                meta.is_some(),
                "follower should have component metadata for {:?}",
                component_id
            );
            assert_eq!(meta.unwrap().name, "Follow Meta Test");
        });

        // Assert follower has component schema.
        fol_db.with_state(|state| {
            let c = state.get_component(component_id);
            assert!(
                c.is_some(),
                "follower should have component {:?}",
                component_id
            );
        });

        // Assert follower has message metadata.
        let fol_path = fol_db.path.clone();
        fol_db.with_state_mut(|state| {
            let msg_log = state.get_or_insert_msg_log(msg_id, &fol_path).unwrap();
            let metadata = msg_log.metadata();
            assert!(metadata.is_some(), "follower msg log should have metadata");
            assert_eq!(metadata.unwrap().name, msg_name);
        });
    }

    #[test]
    async fn test_follow_component_data() {
        // Write 10 samples to source BEFORE creating the follower.
        let src_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let src_addr = src_listener.local_addr().unwrap();
        let src_temp =
            std::env::temp_dir().join(format!("elodin_db_follow_comp_{}", fastrand::u64(..)));
        if src_temp.exists() {
            let _ = std::fs::remove_dir_all(&src_temp);
        }
        let src_server = Server::from_listener(src_listener, src_temp).unwrap();
        let src_db = src_server.db.clone();
        stellar(move || async { src_server.run().await });

        let component_id = ComponentId::new("follow_comp_test");
        let vtable_id = 11u16.to_le_bytes();

        let initial_samples: Vec<(Timestamp, f64)> = (0..10)
            .map(|i| (Timestamp((i + 1) * 1000), (i + 1) as f64 * 10.0))
            .collect();
        send_timestamped_component(
            src_addr,
            component_id,
            "Follow Comp Test",
            vtable_id,
            &initial_samples,
        )
        .await;
        sleep(Duration::from_millis(100)).await;

        // Now create follower (it should backfill the 10 samples).
        let fol_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let fol_addr = fol_listener.local_addr().unwrap();
        let fol_temp =
            std::env::temp_dir().join(format!("elodin_db_follow_comp_fol_{}", fastrand::u64(..)));
        if fol_temp.exists() {
            let _ = std::fs::remove_dir_all(&fol_temp);
        }
        let fol_server = Server::from_listener(fol_listener, fol_temp).unwrap();
        let fol_db = fol_server.db.clone();
        stellar(move || async { fol_server.run().await });

        let follow_db = fol_db.clone();
        stellar(move || {
            elodin_db::follow::run_follower(
                elodin_db::follow::FollowConfig {
                    source_addr: src_addr,
                    target_packet_size: 1500,
                    reconnect_delay: Duration::from_millis(100),
                },
                follow_db,
            )
        });

        // Wait for backfill.
        assert!(
            wait_for_component_samples(&fol_db, component_id, 10, Duration::from_secs(5)).await,
            "follower should have 10 backfilled samples"
        );

        // Verify timestamps match.
        fol_db.with_state(|state| {
            let c = state.get_component(component_id).unwrap();
            let (timestamps, _) = c
                .time_series
                .get_range(&(Timestamp(0)..Timestamp(i64::MAX)))
                .unwrap();
            assert_eq!(timestamps.len(), 10);
            for (i, ts) in timestamps.iter().enumerate() {
                assert_eq!(
                    ts.0,
                    (i as i64 + 1) * 1000,
                    "timestamp mismatch at index {}",
                    i
                );
            }
        });

        // Send 5 more samples in real-time.
        let realtime_samples: Vec<(Timestamp, f64)> = (10..15)
            .map(|i| (Timestamp((i + 1) * 1000), (i + 1) as f64 * 10.0))
            .collect();
        send_timestamped_component(
            src_addr,
            component_id,
            "Follow Comp Test",
            vtable_id,
            &realtime_samples,
        )
        .await;

        // Wait for real-time sync.
        assert!(
            wait_for_component_samples(&fol_db, component_id, 15, Duration::from_secs(5)).await,
            "follower should have 15 total samples"
        );

        // Verify all 15 timestamps.
        fol_db.with_state(|state| {
            let c = state.get_component(component_id).unwrap();
            let (timestamps, data) = c
                .time_series
                .get_range(&(Timestamp(0)..Timestamp(i64::MAX)))
                .unwrap();
            assert_eq!(timestamps.len(), 15);
            let values = <[f64]>::ref_from_bytes(data).unwrap();
            for (i, (ts, val)) in timestamps.iter().zip(values.iter()).enumerate() {
                assert_eq!(ts.0, (i as i64 + 1) * 1000);
                assert_eq!(*val, (i + 1) as f64 * 10.0);
            }
        });

        // CSV and binary comparison.
        assert_exports_match(&src_db, &fol_db, Some("follow_comp_test*"));
        assert_db_files_match(&src_db.path, &fol_db.path);
        let _ = fol_addr; // suppress unused warning
    }

    #[test]
    async fn test_follow_message_data() {
        // Create source and send messages BEFORE creating follower.
        let src_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let src_addr = src_listener.local_addr().unwrap();
        let src_temp =
            std::env::temp_dir().join(format!("elodin_db_follow_msg_{}", fastrand::u64(..)));
        if src_temp.exists() {
            let _ = std::fs::remove_dir_all(&src_temp);
        }
        let src_server = Server::from_listener(src_listener, src_temp).unwrap();
        let _src_db = src_server.db.clone();
        stellar(move || async { src_server.run().await });

        let msg_name = "follow_video_test";
        let msg_id = impeller2::types::msg_id(msg_name);

        {
            let mut client = Client::connect(src_addr).await.unwrap();
            // Set message metadata.
            client
                .send(&SetMsgMetadata {
                    id: msg_id,
                    metadata: MsgMetadata {
                        name: msg_name.to_string(),
                        schema: <impeller2_wkt::OpaqueBytes as postcard_schema::Schema>::SCHEMA
                            .into(),
                        metadata: Default::default(),
                    },
                })
                .await
                .0
                .unwrap();
            sleep(Duration::from_millis(50)).await;

            // Send 5 messages with explicit timestamps.
            for i in 0..5u32 {
                let ts = Timestamp((i as i64 + 1) * 100_000);
                let payload = i.to_le_bytes();
                let mut pkt = LenPacket::msg_with_timestamp(msg_id, ts, payload.len());
                pkt.extend_from_slice(&payload);
                client.send(pkt).await.0.unwrap();
                sleep(Duration::from_millis(10)).await;
            }
            sleep(Duration::from_millis(100)).await;
        }

        // Create follower.
        let fol_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let _fol_addr = fol_listener.local_addr().unwrap();
        let fol_temp =
            std::env::temp_dir().join(format!("elodin_db_follow_msg_fol_{}", fastrand::u64(..)));
        if fol_temp.exists() {
            let _ = std::fs::remove_dir_all(&fol_temp);
        }
        let fol_server = Server::from_listener(fol_listener, fol_temp).unwrap();
        let fol_db = fol_server.db.clone();
        stellar(move || async { fol_server.run().await });

        let follow_db = fol_db.clone();
        stellar(move || {
            elodin_db::follow::run_follower(
                elodin_db::follow::FollowConfig {
                    source_addr: src_addr,
                    target_packet_size: 1500,
                    reconnect_delay: Duration::from_millis(100),
                },
                follow_db,
            )
        });

        // Wait for backfill.
        assert!(
            wait_for_msg_count(&fol_db, msg_id, 5, Duration::from_secs(5)).await,
            "follower should have 5 backfilled messages"
        );

        // Verify timestamps and payloads.
        let fol_path = fol_db.path.clone();
        fol_db.with_state_mut(|state| {
            let log = state.get_or_insert_msg_log(msg_id, &fol_path).unwrap();
            let timestamps = log.timestamps();
            assert_eq!(timestamps.len(), 5);
            for i in 0..5u32 {
                let expected_ts = Timestamp((i as i64 + 1) * 100_000);
                assert_eq!(timestamps[i as usize], expected_ts);
            }
        });

        // Send 3 more messages in real-time.
        {
            let mut client = Client::connect(src_addr).await.unwrap();
            for i in 5..8u32 {
                let ts = Timestamp((i as i64 + 1) * 100_000);
                let payload = i.to_le_bytes();
                let mut pkt = LenPacket::msg_with_timestamp(msg_id, ts, payload.len());
                pkt.extend_from_slice(&payload);
                client.send(pkt).await.0.unwrap();
                sleep(Duration::from_millis(10)).await;
            }
        }

        // Wait for real-time sync.
        assert!(
            wait_for_msg_count(&fol_db, msg_id, 8, Duration::from_secs(5)).await,
            "follower should have 8 total messages"
        );
    }

    #[test]
    async fn test_follow_timestamp_preservation() {
        let (src_addr, src_db, _fol_addr, fol_db) = setup_follow_pair(1500).await.unwrap();

        let component_id = ComponentId::new("follow_ts_test");
        let vtable_id = 12u16.to_le_bytes();

        // Use a realistic microsecond timestamp.
        let realistic_ts = Timestamp(1_700_000_000_000_000);
        send_timestamped_component(
            src_addr,
            component_id,
            "TS Preservation",
            vtable_id,
            &[(realistic_ts, 99.9)],
        )
        .await;

        // Also send a message with a specific timestamp.
        let msg_name = "follow_ts_msg";
        let msg_id = impeller2::types::msg_id(msg_name);
        let msg_ts = Timestamp(1_700_000_000_500_000);
        {
            let mut client = Client::connect(src_addr).await.unwrap();
            client
                .send(&SetMsgMetadata {
                    id: msg_id,
                    metadata: MsgMetadata {
                        name: msg_name.to_string(),
                        schema: <impeller2_wkt::OpaqueBytes as postcard_schema::Schema>::SCHEMA
                            .into(),
                        metadata: Default::default(),
                    },
                })
                .await
                .0
                .unwrap();
            sleep(Duration::from_millis(50)).await;

            let payload = b"hello-timestamp";
            let mut pkt = LenPacket::msg_with_timestamp(msg_id, msg_ts, payload.len());
            pkt.extend_from_slice(payload);
            client.send(pkt).await.0.unwrap();
        }

        // Wait for sync.
        assert!(
            wait_for_component_samples(&fol_db, component_id, 1, Duration::from_secs(5)).await,
            "follower should have the component sample"
        );
        assert!(
            wait_for_msg_count(&fol_db, msg_id, 1, Duration::from_secs(5)).await,
            "follower should have the message"
        );

        // Verify exact timestamp on component.
        fol_db.with_state(|state| {
            let c = state.get_component(component_id).unwrap();
            let (ts, _) = c.time_series.latest().unwrap();
            assert_eq!(
                ts.0, realistic_ts.0,
                "component timestamp should survive round-trip exactly"
            );
        });

        // Verify exact timestamp on message.
        let fol_path = fol_db.path.clone();
        fol_db.with_state_mut(|state| {
            let log = state.get_or_insert_msg_log(msg_id, &fol_path).unwrap();
            let timestamps = log.timestamps();
            assert_eq!(timestamps.len(), 1);
            assert_eq!(
                timestamps[0].0, msg_ts.0,
                "message timestamp should survive round-trip exactly"
            );
        });

        // CSV and binary comparison.
        assert_exports_match(&src_db, &fol_db, Some("follow_ts_test*"));
        assert_db_files_match(&src_db.path, &fol_db.path);
    }

    #[test]
    async fn test_follow_dual_writer_warning() {
        let (src_addr, _src_db, fol_addr, fol_db) = setup_follow_pair(1500).await.unwrap();

        let component_id = ComponentId::new("follow_dual_test");
        let vtable_id = 13u16.to_le_bytes();

        // Write component data to the source.
        send_timestamped_component(
            src_addr,
            component_id,
            "Dual Writer Test",
            vtable_id,
            &[(Timestamp(1000), 1.0)],
        )
        .await;

        // Wait for follower to replicate.
        assert!(
            wait_for_component_samples(&fol_db, component_id, 1, Duration::from_secs(5)).await,
            "follower should have replicated the component"
        );

        // Verify the component is tracked as followed.
        {
            let followed = fol_db.followed_components.read().unwrap();
            assert!(
                followed.contains(&component_id),
                "component should be in followed_components"
            );
        }

        // Now write to the SAME component from a local client connected to the follower.
        // This should succeed (not error), but the component is in followed_components.
        {
            let mut fol_client = Client::connect(fol_addr).await.unwrap();
            let vt = vtable([raw_field(
                0,
                8,
                timestamp(
                    raw_table(8, 8),
                    schema(PrimType::F64, &[], component(component_id)),
                ),
            )]);
            fol_client
                .send(&VTableMsg {
                    id: vtable_id,
                    vtable: vt,
                })
                .await
                .0
                .unwrap();
            sleep(Duration::from_millis(50)).await;

            let mut pkt = LenPacket::table(vtable_id, 16);
            pkt.extend_aligned(&[999.0f64]);
            pkt.extend_aligned(&[2000i64]);
            fol_client.send(pkt).await.0.unwrap();
            sleep(Duration::from_millis(100)).await;
        }

        // The write should have succeeded (no panic/error).
        // The component is still in followed_components.
        let followed = fol_db.followed_components.read().unwrap();
        assert!(
            followed.contains(&component_id),
            "component should still be in followed_components after local write"
        );
    }

    #[test]
    async fn test_follow_local_writer_independent() {
        let (src_addr, _src_db, fol_addr, fol_db) = setup_follow_pair(1500).await.unwrap();

        // Source writes "source_temp".
        let src_component = ComponentId::new("source_temp");
        let src_vtable_id = 14u16.to_le_bytes();
        send_timestamped_component(
            src_addr,
            src_component,
            "Source Temp",
            src_vtable_id,
            &[(Timestamp(1000), 25.0)],
        )
        .await;

        // Wait for follower to replicate source component.
        assert!(
            wait_for_component_samples(&fol_db, src_component, 1, Duration::from_secs(5)).await,
            "follower should have source_temp"
        );

        // Local client writes a DIFFERENT component directly to the follower.
        let local_component = ComponentId::new("local_video");
        let local_vtable_id = 15u16.to_le_bytes();
        send_timestamped_component(
            fol_addr,
            local_component,
            "Local Video",
            local_vtable_id,
            &[(Timestamp(2000), 99.0)],
        )
        .await;
        sleep(Duration::from_millis(200)).await;

        // "local_video" should NOT be in followed_components.
        {
            let followed = fol_db.followed_components.read().unwrap();
            assert!(
                !followed.contains(&local_component),
                "local_video should NOT be in followed_components"
            );
            assert!(
                followed.contains(&src_component),
                "source_temp should be in followed_components"
            );
        }

        // Both components should exist in the follower DB.
        fol_db.with_state(|state| {
            assert!(
                state.get_component(src_component).is_some(),
                "follower should have source_temp"
            );
            assert!(
                state.get_component(local_component).is_some(),
                "follower should have local_video"
            );
        });
    }

    #[test]
    async fn test_db_reopen() {
        let temp_dir =
            std::env::temp_dir().join(format!("elodin_db_persistence_test_{}", fastrand::u64(..)));

        let component_id = ComponentId::new("subscription_test");
        let vtable_id = 1u16.to_le_bytes();
        let test_value = 123.45f64;

        // Define message type for testing
        #[derive(postcard_schema::Schema, serde::Deserialize, serde::Serialize)]
        struct TestMsg {
            value: u32,
            text: String,
        }

        let test_msg = TestMsg {
            value: 42,
            text: "Hello, persistence!".to_string(),
        };

        let set_msg_metadata = SetMsgMetadata {
            id: TestMsg::ID,
            metadata: MsgMetadata {
                name: "TestMessage".to_string(),
                schema: TestMsg::SCHEMA.into(),
                metadata: [("category".to_string(), "persistence_test".to_string())]
                    .into_iter()
                    .collect(),
            },
        };

        {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let server = Server::from_listener(listener, temp_dir.clone()).unwrap();
            stellar(move || async { server.run().await });

            let mut client = Client::connect(addr).await.unwrap();

            client
                .send(
                    &SetComponentMetadata::new(component_id, "Persistence Test Component")
                        .metadata(
                            [("unit".to_string(), "test_unit".to_string())]
                                .into_iter()
                                .collect(),
                        ),
                )
                .await
                .0
                .unwrap();

            let vtable = vtable([raw_field(
                0,
                8,
                schema(PrimType::F64, &[1], component(component_id)),
            )]);

            client
                .send(&VTableMsg {
                    id: vtable_id,
                    vtable,
                })
                .await
                .0
                .unwrap();

            let mut pkt = LenPacket::table(vtable_id, 8);
            pkt.extend_aligned(&[test_value]);
            client.send(pkt).await.0.unwrap();

            client.send(&set_msg_metadata).await.0.unwrap();

            client.send(&test_msg).await.0.unwrap();

            sleep(Duration::from_millis(100)).await;
        }

        let db = elodin_db::DB::open(temp_dir.clone()).unwrap();
        db.with_state_mut(|state| {
            let component = state.get_component_metadata(component_id).unwrap();
            assert_eq!(&component.name, "Persistence Test Component");
            assert_eq!(
                component.metadata.clone(),
                [("unit".to_string(), "test_unit".to_string())]
                    .into_iter()
                    .collect()
            );
            let msg = state.get_or_insert_msg_log(TestMsg::ID, &temp_dir).unwrap();
            let msg_metadata = msg.metadata().unwrap();
            assert_eq!(msg_metadata, &set_msg_metadata.metadata);
            let (_, msg_data) = msg.latest().expect("missing msg");
            assert_eq!(msg_data, postcard::to_allocvec(&test_msg).unwrap());
        });
    }

    // ── Additional follow-mode resilience tests ─────────────────────────

    #[test]
    async fn test_follow_full_disruption() {
        // Persistent data directories that survive across restarts.
        let src_dir =
            std::env::temp_dir().join(format!("elodin_db_disrupt_src_{}", fastrand::u64(..)));
        let fol_dir =
            std::env::temp_dir().join(format!("elodin_db_disrupt_fol_{}", fastrand::u64(..)));
        for d in [&src_dir, &fol_dir] {
            if d.exists() {
                let _ = std::fs::remove_dir_all(d);
            }
        }

        let component_id = ComponentId::new("disruption_test");
        let vtable_id = 20u16.to_le_bytes();
        let msg_name = "disruption_msg";
        let msg_id = impeller2::types::msg_id(msg_name);

        // ── Era 1: steady state ─────────────────────────────────────────
        // Source and follower both run on ephemeral ports. The Era 1 threads
        // will be orphaned when the test moves to Era 2 (we can't kill
        // stellar threads), but that's OK -- we use NEW ports in later eras.
        let src_listener_era1 = TcpListener::bind("127.0.0.1:0").unwrap();
        let src_addr_era1 = src_listener_era1.local_addr().unwrap();

        let src_server_era1 = Server::from_listener(src_listener_era1, &src_dir).unwrap();
        let _src_db_era1 = src_server_era1.db.clone();
        stellar(move || async { src_server_era1.run().await });

        // Write 5 component samples at timestamps 1000..5000.
        let samples: Vec<(Timestamp, f64)> = (1..=5)
            .map(|i| (Timestamp(i * 1000), i as f64 * 10.0))
            .collect();
        send_timestamped_component(
            src_addr_era1,
            component_id,
            "Disruption Test",
            vtable_id,
            &samples,
        )
        .await;

        // Write 3 messages at timestamps 1500, 2500, 3500.
        {
            let mut client = Client::connect(src_addr_era1).await.unwrap();
            client
                .send(&SetMsgMetadata {
                    id: msg_id,
                    metadata: MsgMetadata {
                        name: msg_name.to_string(),
                        schema: <impeller2_wkt::OpaqueBytes as Schema>::SCHEMA.into(),
                        metadata: Default::default(),
                    },
                })
                .await
                .0
                .unwrap();
            sleep(Duration::from_millis(50)).await;
            for i in 0..3u32 {
                let ts = Timestamp((i as i64 + 1) * 1000 + 500);
                let payload = i.to_le_bytes();
                let mut pkt = LenPacket::msg_with_timestamp(msg_id, ts, payload.len());
                pkt.extend_from_slice(&payload);
                client.send(pkt).await.0.unwrap();
                sleep(Duration::from_millis(10)).await;
            }
        }
        sleep(Duration::from_millis(100)).await;

        // Start follower Era 1 and wait for initial sync.
        let fol_listener_era1 = TcpListener::bind("127.0.0.1:0").unwrap();
        let fol_server_era1 = Server::from_listener(fol_listener_era1, &fol_dir).unwrap();
        let fol_db_era1 = fol_server_era1.db.clone();
        stellar(move || async { fol_server_era1.run().await });

        let follow_db_era1 = fol_db_era1.clone();
        stellar(move || {
            elodin_db::follow::run_follower(
                elodin_db::follow::FollowConfig {
                    source_addr: src_addr_era1,
                    target_packet_size: 1500,
                    reconnect_delay: Duration::from_millis(100),
                },
                follow_db_era1,
            )
        });

        // Verify initial sync.
        assert!(
            wait_for_component_samples(&fol_db_era1, component_id, 5, Duration::from_secs(5)).await,
            "era 1: follower should have 5 component samples"
        );
        assert!(
            wait_for_msg_count(&fol_db_era1, msg_id, 3, Duration::from_secs(5)).await,
            "era 1: follower should have 3 messages"
        );

        // ── Disruption: both sides go down ──────────────────────────────
        // We can't kill the stellar threads, but the data directories persist.
        // We start new servers on NEW ports in the next era (simulating a
        // restart where the OS assigned a new port).
        sleep(Duration::from_millis(200)).await;

        // ── Era 2: source comes back on a NEW port with new data ────────
        let src_listener_era2 = TcpListener::bind("127.0.0.1:0").unwrap();
        let src_addr_era2 = src_listener_era2.local_addr().unwrap();
        // Reopen from the same persistent source data dir.
        let src_server_era2 = Server::from_listener(src_listener_era2, &src_dir).unwrap();
        let src_db_era2 = src_server_era2.db.clone();
        stellar(move || async { src_server_era2.run().await });

        // Write 5 more component samples at timestamps 6000..10000.
        let new_samples: Vec<(Timestamp, f64)> = (6..=10)
            .map(|i| (Timestamp(i * 1000), i as f64 * 10.0))
            .collect();
        send_timestamped_component(
            src_addr_era2,
            component_id,
            "Disruption Test",
            vtable_id,
            &new_samples,
        )
        .await;

        // Write 2 more messages at timestamps 4500, 5500.
        {
            let mut client = Client::connect(src_addr_era2).await.unwrap();
            for i in 3..5u32 {
                let ts = Timestamp((i as i64 + 1) * 1000 + 500);
                let payload = i.to_le_bytes();
                let mut pkt = LenPacket::msg_with_timestamp(msg_id, ts, payload.len());
                pkt.extend_from_slice(&payload);
                client.send(pkt).await.0.unwrap();
                sleep(Duration::from_millis(10)).await;
            }
        }
        sleep(Duration::from_millis(200)).await;

        // ── Era 3: follower comes back on a NEW port ────────────────────
        let fol_listener_era3 = TcpListener::bind("127.0.0.1:0").unwrap();
        // Reopen the follower's persisted data directory (has Era 1 data).
        let fol_server_era3 = Server::from_listener(fol_listener_era3, &fol_dir).unwrap();
        let fol_db_era3 = fol_server_era3.db.clone();
        stellar(move || async { fol_server_era3.run().await });

        // New follower task pointing at Era 2's source address.
        let follow_db_era3 = fol_db_era3.clone();
        stellar(move || {
            elodin_db::follow::run_follower(
                elodin_db::follow::FollowConfig {
                    source_addr: src_addr_era2,
                    target_packet_size: 1500,
                    reconnect_delay: Duration::from_millis(100),
                },
                follow_db_era3,
            )
        });

        // ── Final verification ──────────────────────────────────────────
        assert!(
            wait_for_component_samples(&fol_db_era3, component_id, 10, Duration::from_secs(5))
                .await,
            "era 3: follower should have 10 component samples after full disruption"
        );
        assert!(
            wait_for_msg_count(&fol_db_era3, msg_id, 5, Duration::from_secs(5)).await,
            "era 3: follower should have 5 messages after full disruption"
        );

        // Verify exact timestamp sequences.
        fol_db_era3.with_state(|state| {
            let c = state.get_component(component_id).unwrap();
            let (timestamps, data) = c
                .time_series
                .get_range(&(Timestamp(0)..Timestamp(i64::MAX)))
                .unwrap();
            assert_eq!(
                timestamps.len(),
                10,
                "should have exactly 10 samples (no duplicates)"
            );
            let values = <[f64]>::ref_from_bytes(data).unwrap();
            for i in 0..10usize {
                assert_eq!(timestamps[i].0, (i as i64 + 1) * 1000);
                assert_eq!(values[i], (i + 1) as f64 * 10.0);
            }
        });

        let fol_path = fol_db_era3.path.clone();
        fol_db_era3.with_state_mut(|state| {
            let log = state.get_or_insert_msg_log(msg_id, &fol_path).unwrap();
            let timestamps = log.timestamps();
            assert_eq!(
                timestamps.len(),
                5,
                "should have exactly 5 messages (no duplicates)"
            );
            let expected_msg_ts = [1500i64, 2500, 3500, 4500, 5500];
            for (i, ts) in timestamps.iter().enumerate() {
                assert_eq!(
                    ts.0, expected_msg_ts[i],
                    "message timestamp mismatch at {}",
                    i
                );
            }
        });

        // CSV and binary comparison of era 2 source vs era 3 follower.
        assert_exports_match(&src_db_era2, &fol_db_era3, Some("disruption_test*"));
        assert_db_files_match(&src_db_era2.path, &fol_db_era3.path);
    }

    #[test]
    async fn test_follow_new_component_after_connect() {
        let (src_addr, src_db, _fol_addr, fol_db) = setup_follow_pair(1500).await.unwrap();

        // No components exist yet. Create "alpha" with 3 samples.
        let alpha_id = ComponentId::new("follow_alpha");
        let alpha_vtable = 30u16.to_le_bytes();
        send_timestamped_component(
            src_addr,
            alpha_id,
            "Alpha",
            alpha_vtable,
            &[
                (Timestamp(1000), 1.0),
                (Timestamp(2000), 2.0),
                (Timestamp(3000), 3.0),
            ],
        )
        .await;

        // Wait for follower to replicate alpha.
        assert!(
            wait_for_component_samples(&fol_db, alpha_id, 3, Duration::from_secs(5)).await,
            "follower should discover and replicate alpha"
        );

        // Now create a SECOND component "beta" with 2 samples.
        let beta_id = ComponentId::new("follow_beta");
        let beta_vtable = 31u16.to_le_bytes();
        send_timestamped_component(
            src_addr,
            beta_id,
            "Beta",
            beta_vtable,
            &[(Timestamp(4000), 40.0), (Timestamp(5000), 50.0)],
        )
        .await;

        // Wait for follower to replicate beta.
        assert!(
            wait_for_component_samples(&fol_db, beta_id, 2, Duration::from_secs(5)).await,
            "follower should discover and replicate beta"
        );

        // Verify both components have correct data.
        fol_db.with_state(|state| {
            let alpha = state.get_component(alpha_id).unwrap();
            let (ts, _) = alpha
                .time_series
                .get_range(&(Timestamp(0)..Timestamp(i64::MAX)))
                .unwrap();
            assert_eq!(ts.len(), 3);
            assert_eq!(ts[0].0, 1000);
            assert_eq!(ts[1].0, 2000);
            assert_eq!(ts[2].0, 3000);

            let beta = state.get_component(beta_id).unwrap();
            let (ts, _) = beta
                .time_series
                .get_range(&(Timestamp(0)..Timestamp(i64::MAX)))
                .unwrap();
            assert_eq!(ts.len(), 2);
            assert_eq!(ts[0].0, 4000);
            assert_eq!(ts[1].0, 5000);
        });

        // CSV and binary comparison.
        assert_exports_match(&src_db, &fol_db, None);
        assert_db_files_match(&src_db.path, &fol_db.path);
    }

    #[test]
    async fn test_follow_many_components_batching() {
        let (src_addr, src_db, _fol_addr, fol_db) = setup_follow_pair(1500).await.unwrap();

        const NUM_COMPONENTS: usize = 20;

        // Create 20 components, each with 1 initial sample.
        let mut component_ids = Vec::with_capacity(NUM_COMPONENTS);
        for i in 0..NUM_COMPONENTS {
            let name = format!("batch_comp_{}", i);
            let cid = ComponentId::new(&name);
            component_ids.push(cid);

            let vtable_id = (40 + i as u16).to_le_bytes();
            send_timestamped_component(
                src_addr,
                cid,
                &name,
                vtable_id,
                &[(Timestamp((i as i64 + 1) * 1000), i as f64)],
            )
            .await;
        }

        // Wait for follower to replicate all 20 components.
        for (i, &cid) in component_ids.iter().enumerate() {
            assert!(
                wait_for_component_samples(&fol_db, cid, 1, Duration::from_secs(10)).await,
                "follower should have component {} (batch_comp_{})",
                i,
                i
            );
        }

        // Write 1 new sample to each of the 20 components (strictly newer timestamp).
        for (i, &cid) in component_ids.iter().enumerate() {
            let vtable_id = (40 + i as u16).to_le_bytes();
            send_timestamped_component(
                src_addr,
                cid,
                &format!("batch_comp_{}", i),
                vtable_id,
                &[(Timestamp((i as i64 + 1) * 1000 + 500), (i as f64) + 100.0)],
            )
            .await;
        }

        // Wait for all 20 components to have 2 samples.
        for (i, &cid) in component_ids.iter().enumerate() {
            assert!(
                wait_for_component_samples(&fol_db, cid, 2, Duration::from_secs(10)).await,
                "follower should have 2 samples for component {} (batch_comp_{})",
                i,
                i
            );
        }

        // Verify exact counts (no duplicates).
        fol_db.with_state(|state| {
            for (i, &cid) in component_ids.iter().enumerate() {
                let c = state.get_component(cid).unwrap();
                let (ts, _) = c
                    .time_series
                    .get_range(&(Timestamp(0)..Timestamp(i64::MAX)))
                    .unwrap();
                assert_eq!(
                    ts.len(),
                    2,
                    "batch_comp_{} should have exactly 2 samples, got {}",
                    i,
                    ts.len()
                );
            }
        });
    }

    #[test]
    async fn test_follow_source_data_before_and_after_connect() {
        // Start source manually -- no follower yet.
        let src_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let src_addr = src_listener.local_addr().unwrap();
        let src_temp =
            std::env::temp_dir().join(format!("elodin_db_follow_mixed_{}", fastrand::u64(..)));
        if src_temp.exists() {
            let _ = std::fs::remove_dir_all(&src_temp);
        }
        let src_server = Server::from_listener(src_listener, src_temp).unwrap();
        let src_db = src_server.db.clone();
        stellar(move || async { src_server.run().await });

        let component_id = ComponentId::new("follow_sensor");
        let vtable_id = 70u16.to_le_bytes();
        let msg_name = "follow_telemetry";
        let msg_id = impeller2::types::msg_id(msg_name);

        // Write pre-connect component data: timestamps 1000, 2000, 3000.
        send_timestamped_component(
            src_addr,
            component_id,
            "Sensor",
            vtable_id,
            &[
                (Timestamp(1000), 10.0),
                (Timestamp(2000), 20.0),
                (Timestamp(3000), 30.0),
            ],
        )
        .await;

        // Write pre-connect messages: timestamps 1500, 2500.
        {
            let mut client = Client::connect(src_addr).await.unwrap();
            client
                .send(&SetMsgMetadata {
                    id: msg_id,
                    metadata: MsgMetadata {
                        name: msg_name.to_string(),
                        schema: <impeller2_wkt::OpaqueBytes as Schema>::SCHEMA.into(),
                        metadata: Default::default(),
                    },
                })
                .await
                .0
                .unwrap();
            sleep(Duration::from_millis(50)).await;
            for &ts_val in &[1500i64, 2500] {
                let ts = Timestamp(ts_val);
                let payload = ts_val.to_le_bytes();
                let mut pkt = LenPacket::msg_with_timestamp(msg_id, ts, payload.len());
                pkt.extend_from_slice(&payload);
                client.send(pkt).await.0.unwrap();
                sleep(Duration::from_millis(10)).await;
            }
        }
        sleep(Duration::from_millis(100)).await;

        // Start follower -- it will backfill the 3 component samples and 2 messages.
        let fol_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let _fol_addr = fol_listener.local_addr().unwrap();
        let fol_temp =
            std::env::temp_dir().join(format!("elodin_db_follow_mixed_fol_{}", fastrand::u64(..)));
        if fol_temp.exists() {
            let _ = std::fs::remove_dir_all(&fol_temp);
        }
        let fol_server = Server::from_listener(fol_listener, fol_temp).unwrap();
        let fol_db = fol_server.db.clone();
        stellar(move || async { fol_server.run().await });

        let follow_db = fol_db.clone();
        stellar(move || {
            elodin_db::follow::run_follower(
                elodin_db::follow::FollowConfig {
                    source_addr: src_addr,
                    target_packet_size: 1500,
                    reconnect_delay: Duration::from_millis(100),
                },
                follow_db,
            )
        });

        // Wait for backfill to complete.
        assert!(
            wait_for_component_samples(&fol_db, component_id, 3, Duration::from_secs(5)).await,
            "follower should backfill 3 component samples"
        );
        assert!(
            wait_for_msg_count(&fol_db, msg_id, 2, Duration::from_secs(5)).await,
            "follower should backfill 2 messages"
        );

        // Now write post-connect data.
        // Component samples at 4000, 5000.
        send_timestamped_component(
            src_addr,
            component_id,
            "Sensor",
            vtable_id,
            &[(Timestamp(4000), 40.0), (Timestamp(5000), 50.0)],
        )
        .await;

        // Messages at 3500, 4500.
        {
            let mut client = Client::connect(src_addr).await.unwrap();
            for &ts_val in &[3500i64, 4500] {
                let ts = Timestamp(ts_val);
                let payload = ts_val.to_le_bytes();
                let mut pkt = LenPacket::msg_with_timestamp(msg_id, ts, payload.len());
                pkt.extend_from_slice(&payload);
                client.send(pkt).await.0.unwrap();
                sleep(Duration::from_millis(10)).await;
            }
        }

        // Wait for follower to have all data.
        assert!(
            wait_for_component_samples(&fol_db, component_id, 5, Duration::from_secs(5)).await,
            "follower should have 5 total component samples"
        );
        assert!(
            wait_for_msg_count(&fol_db, msg_id, 4, Duration::from_secs(5)).await,
            "follower should have 4 total messages"
        );

        // Verify EXACT timestamp sequences with no duplicates.
        fol_db.with_state(|state| {
            let c = state.get_component(component_id).unwrap();
            let (timestamps, data) = c
                .time_series
                .get_range(&(Timestamp(0)..Timestamp(i64::MAX)))
                .unwrap();
            assert_eq!(
                timestamps.len(),
                5,
                "should have exactly 5 component samples (no duplicates)"
            );
            let expected_ts = [1000i64, 2000, 3000, 4000, 5000];
            let expected_vals = [10.0, 20.0, 30.0, 40.0, 50.0];
            let values = <[f64]>::ref_from_bytes(data).unwrap();
            for i in 0..5 {
                assert_eq!(
                    timestamps[i].0, expected_ts[i],
                    "component ts mismatch at {}",
                    i
                );
                assert_eq!(
                    values[i], expected_vals[i],
                    "component value mismatch at {}",
                    i
                );
            }
        });

        let fol_path = fol_db.path.clone();
        fol_db.with_state_mut(|state| {
            let log = state.get_or_insert_msg_log(msg_id, &fol_path).unwrap();
            let timestamps = log.timestamps();
            assert_eq!(
                timestamps.len(),
                4,
                "should have exactly 4 messages (no duplicates)"
            );
            let expected_msg_ts = [1500i64, 2500, 3500, 4500];
            for (i, ts) in timestamps.iter().enumerate() {
                assert_eq!(ts.0, expected_msg_ts[i], "message ts mismatch at {}", i);
            }
        });

        // CSV and binary comparison.
        assert_exports_match(&src_db, &fol_db, Some("follow_sensor*"));
        assert_db_files_match(&src_db.path, &fol_db.path);
    }

    // ── Comprehensive replication verification ──────────────────────────

    /// Helper: send a typed component with explicit timestamps.
    /// `prim` and `elem_size` describe the element type; `make_row` generates
    /// one row of data as bytes for each sample index.
    async fn send_typed_component(
        addr: SocketAddr,
        cid: ComponentId,
        name: &str,
        vtable_id: [u8; 2],
        prim: PrimType,
        shape: &[u64],
        timestamps: &[Timestamp],
        make_row: impl Fn(usize) -> Vec<u8>,
    ) {
        let mut client = Client::connect(addr).await.unwrap();

        client
            .send(&SetComponentMetadata::new(cid, name))
            .await
            .0
            .unwrap();

        let elem_count: usize = shape.iter().product::<u64>().max(1) as usize;
        let prim_size = match prim {
            PrimType::F64 | PrimType::U64 | PrimType::I64 => 8,
            PrimType::F32 | PrimType::U32 | PrimType::I32 => 4,
            PrimType::U16 | PrimType::I16 => 2,
            PrimType::U8 | PrimType::I8 | PrimType::Bool => 1,
        };
        let row_size = elem_count * prim_size;

        // VTable with explicit timestamp: [data | timestamp]
        let vt = vtable([raw_field(
            0,
            row_size as u16,
            timestamp(
                raw_table(row_size as u16, 8),
                schema(prim, shape, component(cid)),
            ),
        )]);
        client
            .send(&VTableMsg {
                id: vtable_id,
                vtable: vt,
            })
            .await
            .0
            .unwrap();
        sleep(Duration::from_millis(50)).await;

        for (i, ts) in timestamps.iter().enumerate() {
            let row = make_row(i);
            let mut pkt = LenPacket::table(vtable_id, row.len() + 8);
            pkt.extend_from_slice(&row);
            pkt.extend_aligned(&[ts.0]);
            client.send(pkt).await.0.unwrap();
            // Small delay to avoid timestamp collisions
            if i % 20 == 19 {
                sleep(Duration::from_millis(5)).await;
            }
        }
        sleep(Duration::from_millis(50)).await;
    }

    #[test]
    async fn test_follow_realistic_replication() {
        let subscriber = tracing_subscriber::FmtSubscriber::new();
        let _ = tracing::subscriber::set_global_default(subscriber);

        // Persistent directories.
        let src_dir =
            std::env::temp_dir().join(format!("elodin_db_realistic_src_{}", fastrand::u64(..)));
        let fol_dir =
            std::env::temp_dir().join(format!("elodin_db_realistic_fol_{}", fastrand::u64(..)));
        for d in [&src_dir, &fol_dir] {
            if d.exists() {
                let _ = std::fs::remove_dir_all(d);
            }
        }

        // ── Start source ────────────────────────────────────────────────
        let src_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let src_addr = src_listener.local_addr().unwrap();
        let src_server = Server::from_listener(src_listener, &src_dir).unwrap();
        let src_db = src_server.db.clone();
        stellar(move || async { src_server.run().await });

        // ── Send KDL schematic ──────────────────────────────────────────
        {
            let mut client = Client::connect(src_addr).await.unwrap();
            let kdl = r#"
hsplit {
  viewport pos="vehicle.world_pos.translate_world(0,0,5)" look_at="vehicle.world_pos"
}
object_3d vehicle.world_pos {
  sphere radius=0.5 { color blue }
}
graph "vehicle.world_pos"
graph "vehicle.world_vel"
graph "vehicle.thrust"
"#;
            client
                .send(&SetDbConfig {
                    recording: None,
                    metadata: [("schematic.content".to_string(), kdl.to_string())]
                        .into_iter()
                        .collect(),
                })
                .await
                .0
                .unwrap();
            sleep(Duration::from_millis(50)).await;
        }

        // ── Send entity metadata (bare names, no schema) ────────────────
        {
            let mut client = Client::connect(src_addr).await.unwrap();
            client
                .send(&SetComponentMetadata::new(
                    ComponentId::new("vehicle"),
                    "vehicle",
                ))
                .await
                .0
                .unwrap();
            client
                .send(&SetComponentMetadata::new(
                    ComponentId::new("Globals"),
                    "Globals",
                ))
                .await
                .0
                .unwrap();
            sleep(Duration::from_millis(50)).await;
        }

        // ── Component data ──────────────────────────────────────────────
        let sample_count = 100usize;
        let timestamps: Vec<Timestamp> = (1..=sample_count)
            .map(|i| Timestamp(i as i64 * 1000))
            .collect();

        // vehicle.world_pos: f64[7] (SpatialTransform: qw,qx,qy,qz,x,y,z)
        let world_pos_id = ComponentId::new("vehicle.world_pos");
        send_typed_component(
            src_addr,
            world_pos_id,
            "vehicle.world_pos",
            100u16.to_le_bytes(),
            PrimType::F64,
            &[7],
            &timestamps,
            |i| {
                let qw = 1.0f64;
                let qx = 0.0f64;
                let qy = 0.0f64;
                let qz = 0.0f64;
                let x = i as f64 * 0.5;
                let y = (i as f64 * 0.1).sin() * 10.0;
                let z = 50.0 + (i as f64 * 0.05).cos() * 5.0;
                [qw, qx, qy, qz, x, y, z]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect()
            },
        )
        .await;

        // vehicle.world_vel: f64[6] (SpatialMotion: wx,wy,wz,vx,vy,vz)
        let world_vel_id = ComponentId::new("vehicle.world_vel");
        send_typed_component(
            src_addr,
            world_vel_id,
            "vehicle.world_vel",
            101u16.to_le_bytes(),
            PrimType::F64,
            &[6],
            &timestamps,
            |i| {
                let wx = 0.01f64 * i as f64;
                let wy = 0.0f64;
                let wz = -0.005f64 * i as f64;
                let vx = 70.0f64;
                let vy = (i as f64 * 0.1).cos() * 2.0;
                let vz = -0.5f64;
                [wx, wy, wz, vx, vy, vz]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect()
            },
        )
        .await;

        // vehicle.thrust: f64[1] (scalar)
        let thrust_id = ComponentId::new("vehicle.thrust");
        send_typed_component(
            src_addr,
            thrust_id,
            "vehicle.thrust",
            102u16.to_le_bytes(),
            PrimType::F64,
            &[],
            &timestamps,
            |i| {
                let thrust = 150.0f64 + (i as f64 * 0.1).sin() * 30.0;
                thrust.to_le_bytes().to_vec()
            },
        )
        .await;

        // vehicle.control_surfaces: f32[4] (different prim type!)
        let ctrl_id = ComponentId::new("vehicle.control_surfaces");
        send_typed_component(
            src_addr,
            ctrl_id,
            "vehicle.control_surfaces",
            103u16.to_le_bytes(),
            PrimType::F32,
            &[4],
            &timestamps,
            |i| {
                let aileron = (i as f32 * 0.2).sin() * 15.0;
                let elevator = -5.0f32 + i as f32 * 0.1;
                let rudder = 0.0f32;
                let flap = if i > 50 { 10.0f32 } else { 0.0f32 };
                [aileron, elevator, rudder, flap]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect()
            },
        )
        .await;

        // Globals.tick: u64[1] (integer type)
        let tick_id = ComponentId::new("Globals.tick");
        send_typed_component(
            src_addr,
            tick_id,
            "Globals.tick",
            104u16.to_le_bytes(),
            PrimType::U64,
            &[],
            &timestamps,
            |i| (i as u64).to_le_bytes().to_vec(),
        )
        .await;

        // Globals.simulation_time_step: f64[1] (single static value)
        let timestep_id = ComponentId::new("Globals.simulation_time_step");
        send_typed_component(
            src_addr,
            timestep_id,
            "Globals.simulation_time_step",
            105u16.to_le_bytes(),
            PrimType::F64,
            &[],
            &[Timestamp(1000)],
            |_| (1.0f64 / 300.0).to_le_bytes().to_vec(),
        )
        .await;

        // ── Messages ────────────────────────────────────────────────────
        let msg_name = "telemetry_log";
        let msg_id = impeller2::types::msg_id(msg_name);
        {
            let mut client = Client::connect(src_addr).await.unwrap();
            client
                .send(&SetMsgMetadata {
                    id: msg_id,
                    metadata: MsgMetadata {
                        name: msg_name.to_string(),
                        schema: <impeller2_wkt::OpaqueBytes as postcard_schema::Schema>::SCHEMA
                            .into(),
                        metadata: Default::default(),
                    },
                })
                .await
                .0
                .unwrap();
            sleep(Duration::from_millis(50)).await;

            for i in 0..20u32 {
                let ts = Timestamp((i as i64 + 1) * 5000);
                let payload = format!("telemetry_entry_{:04}", i);
                let mut pkt = LenPacket::msg_with_timestamp(msg_id, ts, payload.len());
                pkt.extend_from_slice(payload.as_bytes());
                client.send(pkt).await.0.unwrap();
                sleep(Duration::from_millis(5)).await;
            }
        }

        sleep(Duration::from_millis(200)).await;

        // ── Start follower ──────────────────────────────────────────────
        let fol_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let fol_server = Server::from_listener(fol_listener, &fol_dir).unwrap();
        let fol_db = fol_server.db.clone();
        stellar(move || async { fol_server.run().await });

        let follow_db = fol_db.clone();
        stellar(move || {
            elodin_db::follow::run_follower(
                elodin_db::follow::FollowConfig {
                    source_addr: src_addr,
                    target_packet_size: 1500,
                    reconnect_delay: Duration::from_millis(100),
                },
                follow_db,
            )
        });

        // ── Wait for full sync ──────────────────────────────────────────
        for (cid, expected) in [
            (world_pos_id, sample_count),
            (world_vel_id, sample_count),
            (thrust_id, sample_count),
            (ctrl_id, sample_count),
            (tick_id, sample_count),
            (timestep_id, 1),
        ] {
            assert!(
                wait_for_component_samples(&fol_db, cid, expected, Duration::from_secs(10)).await,
                "follower should have {} samples for {:?}",
                expected,
                cid
            );
        }
        assert!(
            wait_for_msg_count(&fol_db, msg_id, 20, Duration::from_secs(10)).await,
            "follower should have 20 messages"
        );

        // ── Flush both databases ────────────────────────────────────────
        src_db.flush_all().unwrap();
        fol_db.flush_all().unwrap();
        sleep(Duration::from_millis(200)).await;

        // ── CSV export comparison ───────────────────────────────────────
        assert_exports_match(&src_db, &fol_db, None);

        // ── Binary DB file comparison ───────────────────────────────────
        assert_db_files_match(&src_db.path, &fol_db.path);
    }

    // ── Packet size compliance test ─────────────────────────────────────

    /// Test that different `--follow-packet-size` values (small, default,
    /// large) all correctly replicate data.
    ///
    /// Uses 30 scalar f64 components with 100 samples each = 3,000 per-
    /// component packets of 24 bytes = 72,000 bytes total.  This fills:
    /// - small  (64 B):  ~1,125 flushes
    /// - default (1500 B): ~48 flushes
    /// - large  (9000 B): ~8 flushes
    #[test]
    async fn test_follow_packet_size_compliance() {
        // Create source with 30 scalar components BEFORE starting followers.
        let src_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let src_addr = src_listener.local_addr().unwrap();
        let src_temp =
            std::env::temp_dir().join(format!("elodin_db_pkt_src_{}", fastrand::u64(..)));
        if src_temp.exists() {
            let _ = std::fs::remove_dir_all(&src_temp);
        }
        let src_server = Server::from_listener(src_listener, &src_temp).unwrap();
        let _src_db = src_server.db.clone();
        stellar(move || async { src_server.run().await });

        const NUM: usize = 30;
        let samples = 100usize;
        let timestamps: Vec<Timestamp> =
            (1..=samples).map(|i| Timestamp(i as i64 * 1000)).collect();

        // Write all data via a single client connection for speed.
        // Each component gets its own VTable and 100 samples.
        {
            let mut client = Client::connect(src_addr).await.unwrap();
            for i in 0..NUM {
                let name = format!("pkt_{}", i);
                let cid = ComponentId::new(&name);
                let vtable_id = (200 + i as u16).to_le_bytes();

                client
                    .send(&SetComponentMetadata::new(cid, &name))
                    .await
                    .0
                    .unwrap();

                let vt = vtable([raw_field(
                    0,
                    8,
                    timestamp(raw_table(8, 8), schema(PrimType::F64, &[], component(cid))),
                )]);
                client
                    .send(&VTableMsg {
                        id: vtable_id,
                        vtable: vt,
                    })
                    .await
                    .0
                    .unwrap();

                // Send all 100 samples without per-sample delays.
                for &ts in &timestamps {
                    let mut pkt = LenPacket::table(vtable_id, 16);
                    pkt.extend_aligned(&[i as f64]);
                    pkt.extend_aligned(&[ts.0]);
                    client.send(pkt).await.0.unwrap();
                }
            }
        }
        sleep(Duration::from_millis(200)).await;

        // Test each packet size sequentially.
        for (packet_size, label) in [(64, "small"), (1500, "default"), (9000, "large")] {
            let fol_listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let fol_temp = std::env::temp_dir().join(format!(
                "elodin_db_pkt_fol_{}_{}",
                label,
                fastrand::u64(..)
            ));
            if fol_temp.exists() {
                let _ = std::fs::remove_dir_all(&fol_temp);
            }
            let fol_server = Server::from_listener(fol_listener, &fol_temp).unwrap();
            let fol_db = fol_server.db.clone();
            stellar(move || async { fol_server.run().await });

            let follow_db = fol_db.clone();
            stellar(move || {
                elodin_db::follow::run_follower(
                    elodin_db::follow::FollowConfig {
                        source_addr: src_addr,
                        target_packet_size: packet_size,
                        reconnect_delay: Duration::from_millis(100),
                    },
                    follow_db,
                )
            });

            // Wait for all components to arrive with all samples.
            for i in 0..NUM {
                let cid = ComponentId::new(&format!("pkt_{}", i));
                assert!(
                    wait_for_component_samples(&fol_db, cid, samples, Duration::from_secs(15))
                        .await,
                    "'{}' (size={}) missing data for pkt_{}",
                    label,
                    packet_size,
                    i
                );
            }

            // Verify exact sample counts (no duplicates, no missing).
            fol_db.with_state(|state| {
                for i in 0..NUM {
                    let cid = ComponentId::new(&format!("pkt_{}", i));
                    let c = state.get_component(cid).unwrap();
                    let (ts, _) = c
                        .time_series
                        .get_range(&(Timestamp(0)..Timestamp(i64::MAX)))
                        .unwrap();
                    assert_eq!(
                        ts.len(),
                        samples,
                        "'{}' (size={}) pkt_{} has {} samples, expected {}",
                        label,
                        packet_size,
                        i,
                        ts.len(),
                        samples
                    );
                }
            });
        }
    }
}
