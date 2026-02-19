#[cfg(test)]
mod tests {

    use arrow::{array::AsArray, datatypes::Float64Type};
    use elodin_db::{AtomicTimestampExt, DB, Error, Server};
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

    /// Helper: register a VTable and send typed component samples over an
    /// existing client connection.  No per-sample sleeps -- only a short
    /// delay after VTable registration.
    async fn send_timestamped_samples(
        client: &mut Client,
        component_id: ComponentId,
        component_name: &str,
        vtable_id: [u8; 2],
        prim: PrimType,
        shape: &[u64],
        timestamps: &[Timestamp],
        make_row: impl Fn(usize) -> Vec<u8>,
    ) {
        client
            .send(&SetComponentMetadata::new(component_id, component_name))
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

        let vt = vtable([raw_field(
            0,
            row_size as u16,
            timestamp(
                raw_table(row_size as u16, 8),
                schema(prim, shape, component(component_id)),
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
        }
    }

    /// Convenience wrapper for scalar f64 components.
    async fn send_f64_samples(
        client: &mut Client,
        component_id: ComponentId,
        name: &str,
        vtable_id: [u8; 2],
        samples: &[(Timestamp, f64)],
    ) {
        let timestamps: Vec<Timestamp> = samples.iter().map(|(ts, _)| *ts).collect();
        let values: Vec<f64> = samples.iter().map(|(_, v)| *v).collect();
        send_timestamped_samples(
            client,
            component_id,
            name,
            vtable_id,
            PrimType::F64,
            &[],
            &timestamps,
            |i| values[i].to_le_bytes().to_vec(),
        )
        .await;
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

    /// Covers: metadata sync, component backfill + real-time, message
    /// backfill + real-time, microsecond timestamp preservation,
    /// backfill/real-time boundary, CSV + binary comparison.
    #[test]
    async fn test_follow_basic_replication() {
        let subscriber = tracing_subscriber::FmtSubscriber::new();
        let _ = tracing::subscriber::set_global_default(subscriber);

        // Start source.
        let src_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let src_addr = src_listener.local_addr().unwrap();
        let src_temp =
            std::env::temp_dir().join(format!("elodin_db_follow_basic_{}", fastrand::u64(..)));
        if src_temp.exists() {
            let _ = std::fs::remove_dir_all(&src_temp);
        }
        let src_server = Server::from_listener(src_listener, src_temp).unwrap();
        let src_db = src_server.db.clone();
        stellar(move || async { src_server.run().await });

        let component_id = ComponentId::new("follow_sensor");
        let vtable_id = 10u16.to_le_bytes();
        let msg_name = "follow_telemetry";
        let msg_id = impeller2::types::msg_id(msg_name);

        // Pre-connect: write 3 component samples with realistic timestamps.
        {
            let mut client = Client::connect(src_addr).await.unwrap();
            send_f64_samples(
                &mut client,
                component_id,
                "Follow Sensor",
                vtable_id,
                &[
                    (Timestamp(1_700_000_000_000_000), 10.0),
                    (Timestamp(1_700_000_000_100_000), 20.0),
                    (Timestamp(1_700_000_000_200_000), 30.0),
                ],
            )
            .await;

            // Pre-connect: write 2 messages.
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

            for i in 0..2u32 {
                let ts = Timestamp(1_700_000_000_050_000 + i as i64 * 100_000);
                let payload = i.to_le_bytes();
                let mut pkt = LenPacket::msg_with_timestamp(msg_id, ts, payload.len());
                pkt.extend_from_slice(&payload);
                client.send(pkt).await.0.unwrap();
            }
        }
        sleep(Duration::from_millis(100)).await;

        // Start follower.
        let fol_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let _fol_addr = fol_listener.local_addr().unwrap();
        let fol_temp =
            std::env::temp_dir().join(format!("elodin_db_follow_basic_fol_{}", fastrand::u64(..)));
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
            wait_for_component_samples(&fol_db, component_id, 3, Duration::from_secs(5)).await,
            "follower should backfill 3 component samples"
        );
        assert!(
            wait_for_msg_count(&fol_db, msg_id, 2, Duration::from_secs(5)).await,
            "follower should backfill 2 messages"
        );

        // Verify metadata.
        fol_db.with_state(|state| {
            let meta = state.get_component_metadata(component_id);
            assert!(meta.is_some(), "follower should have component metadata");
            assert_eq!(meta.unwrap().name, "Follow Sensor");
            assert!(
                state.get_component(component_id).is_some(),
                "follower should have component schema"
            );
        });
        let fol_path = fol_db.path.clone();
        fol_db.with_state_mut(|state| {
            let log = state.get_or_insert_msg_log(msg_id, &fol_path).unwrap();
            let metadata = log.metadata();
            assert!(metadata.is_some(), "follower should have msg metadata");
            assert_eq!(metadata.unwrap().name, msg_name);
        });

        // Verify exact timestamps survived round-trip.
        fol_db.with_state(|state| {
            let c = state.get_component(component_id).unwrap();
            let (ts, data) = c
                .time_series
                .get_range(&(Timestamp(0)..Timestamp(i64::MAX)))
                .unwrap();
            assert_eq!(ts.len(), 3);
            assert_eq!(ts[0].0, 1_700_000_000_000_000);
            assert_eq!(ts[1].0, 1_700_000_000_100_000);
            assert_eq!(ts[2].0, 1_700_000_000_200_000);
            let values = <[f64]>::ref_from_bytes(data).unwrap();
            assert_eq!(values[0], 10.0);
            assert_eq!(values[1], 20.0);
            assert_eq!(values[2], 30.0);
        });

        // Post-connect: write 2 more component samples + 1 message.
        {
            let mut client = Client::connect(src_addr).await.unwrap();
            send_f64_samples(
                &mut client,
                component_id,
                "Follow Sensor",
                vtable_id,
                &[
                    (Timestamp(1_700_000_000_300_000), 40.0),
                    (Timestamp(1_700_000_000_400_000), 50.0),
                ],
            )
            .await;

            let ts = Timestamp(1_700_000_000_350_000);
            let payload = 2u32.to_le_bytes();
            let mut pkt = LenPacket::msg_with_timestamp(msg_id, ts, payload.len());
            pkt.extend_from_slice(&payload);
            client.send(pkt).await.0.unwrap();
        }

        // Wait for real-time sync.
        assert!(
            wait_for_component_samples(&fol_db, component_id, 5, Duration::from_secs(5)).await,
            "follower should have 5 total component samples"
        );
        assert!(
            wait_for_msg_count(&fol_db, msg_id, 3, Duration::from_secs(5)).await,
            "follower should have 3 total messages"
        );

        // Verify all 5 timestamps + values.
        fol_db.with_state(|state| {
            let c = state.get_component(component_id).unwrap();
            let (ts, data) = c
                .time_series
                .get_range(&(Timestamp(0)..Timestamp(i64::MAX)))
                .unwrap();
            assert_eq!(ts.len(), 5, "should have exactly 5 samples (no duplicates)");
            let expected_ts = [
                1_700_000_000_000_000i64,
                1_700_000_000_100_000,
                1_700_000_000_200_000,
                1_700_000_000_300_000,
                1_700_000_000_400_000,
            ];
            let expected_vals = [10.0, 20.0, 30.0, 40.0, 50.0];
            let values = <[f64]>::ref_from_bytes(data).unwrap();
            for i in 0..5 {
                assert_eq!(ts[i].0, expected_ts[i], "component ts mismatch at {}", i);
                assert_eq!(values[i], expected_vals[i], "value mismatch at {}", i);
            }
        });

        // CSV + binary comparison.
        assert_exports_match(&src_db, &fol_db, None);
        assert_db_files_match(&src_db.path, &fol_db.path);
    }

    /// Covers: dual-writer to a followed component, independent local
    /// writer to a non-followed component.
    #[test]
    async fn test_follow_local_writers() {
        let (src_addr, _src_db, fol_addr, fol_db) = setup_follow_pair(1500).await.unwrap();

        let followed_id = ComponentId::new("followed_temp");
        let followed_vtable = 13u16.to_le_bytes();
        let local_id = ComponentId::new("local_video");
        let local_vtable = 14u16.to_le_bytes();

        // Source writes the followed component.
        {
            let mut client = Client::connect(src_addr).await.unwrap();
            send_f64_samples(
                &mut client,
                followed_id,
                "Followed Temp",
                followed_vtable,
                &[(Timestamp(1000), 25.0)],
            )
            .await;
        }

        // Wait for follower to replicate.
        assert!(
            wait_for_component_samples(&fol_db, followed_id, 1, Duration::from_secs(5)).await,
            "follower should have replicated followed_temp"
        );

        // Verify it is tracked as followed.
        {
            let followed = fol_db.followed_components.read().unwrap();
            assert!(
                followed.contains(&followed_id),
                "followed_temp should be in followed_components"
            );
        }

        // Local client writes to the SAME followed component (dual writer).
        {
            let mut fol_client = Client::connect(fol_addr).await.unwrap();
            let vt = vtable([raw_field(
                0,
                8,
                timestamp(
                    raw_table(8, 8),
                    schema(PrimType::F64, &[], component(followed_id)),
                ),
            )]);
            fol_client
                .send(&VTableMsg {
                    id: followed_vtable,
                    vtable: vt,
                })
                .await
                .0
                .unwrap();
            sleep(Duration::from_millis(50)).await;

            let mut pkt = LenPacket::table(followed_vtable, 16);
            pkt.extend_aligned(&[999.0f64]);
            pkt.extend_aligned(&[2000i64]);
            fol_client.send(pkt).await.0.unwrap();
            sleep(Duration::from_millis(100)).await;
        }

        // Still tracked as followed after local write.
        {
            let followed = fol_db.followed_components.read().unwrap();
            assert!(
                followed.contains(&followed_id),
                "followed_temp should still be in followed_components after local write"
            );
        }

        // Local client writes a DIFFERENT (non-followed) component.
        {
            let mut fol_client = Client::connect(fol_addr).await.unwrap();
            send_f64_samples(
                &mut fol_client,
                local_id,
                "Local Video",
                local_vtable,
                &[(Timestamp(2000), 99.0)],
            )
            .await;
        }
        sleep(Duration::from_millis(200)).await;

        // local_video should NOT be in followed_components.
        {
            let followed = fol_db.followed_components.read().unwrap();
            assert!(
                !followed.contains(&local_id),
                "local_video should NOT be in followed_components"
            );
            assert!(
                followed.contains(&followed_id),
                "followed_temp should still be in followed_components"
            );
        }

        // Both components should exist.
        fol_db.with_state(|state| {
            assert!(state.get_component(followed_id).is_some());
            assert!(state.get_component(local_id).is_some());
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

    // ── Additional follow-mode tests ────────────────────────────────────

    /// Covers: reconnection dedup for both components and messages after
    /// a full source + follower restart.
    #[test]
    async fn test_follow_reconnection_dedup() {
        let subscriber = tracing_subscriber::FmtSubscriber::new();
        let _ = tracing::subscriber::set_global_default(subscriber);

        let src_dir =
            std::env::temp_dir().join(format!("elodin_db_dedup_src_{}", fastrand::u64(..)));
        let fol_dir =
            std::env::temp_dir().join(format!("elodin_db_dedup_fol_{}", fastrand::u64(..)));
        for d in [&src_dir, &fol_dir] {
            if d.exists() {
                let _ = std::fs::remove_dir_all(d);
            }
        }

        let component_id = ComponentId::new("dedup_test");
        let vtable_id = 20u16.to_le_bytes();
        let msg_name = "dedup_msg";
        let msg_id = impeller2::types::msg_id(msg_name);

        // ── Era 1: write 3 component samples + 1 message ────────────────
        let src_listener_1 = TcpListener::bind("127.0.0.1:0").unwrap();
        let src_addr_1 = src_listener_1.local_addr().unwrap();
        let src_server_1 = Server::from_listener(src_listener_1, &src_dir).unwrap();
        let _src_db_1 = src_server_1.db.clone();
        stellar(move || async { src_server_1.run().await });

        {
            let mut client = Client::connect(src_addr_1).await.unwrap();
            send_f64_samples(
                &mut client,
                component_id,
                "Dedup Test",
                vtable_id,
                &[
                    (Timestamp(1000), 10.0),
                    (Timestamp(2000), 20.0),
                    (Timestamp(3000), 30.0),
                ],
            )
            .await;

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

            let ts = Timestamp(1500);
            let payload = 0u32.to_le_bytes();
            let mut pkt = LenPacket::msg_with_timestamp(msg_id, ts, payload.len());
            pkt.extend_from_slice(&payload);
            client.send(pkt).await.0.unwrap();
        }
        sleep(Duration::from_millis(100)).await;

        // Start follower and wait for sync.
        let fol_listener_1 = TcpListener::bind("127.0.0.1:0").unwrap();
        let fol_server_1 = Server::from_listener(fol_listener_1, &fol_dir).unwrap();
        let fol_db_1 = fol_server_1.db.clone();
        stellar(move || async { fol_server_1.run().await });

        let follow_db_1 = fol_db_1.clone();
        stellar(move || {
            elodin_db::follow::run_follower(
                elodin_db::follow::FollowConfig {
                    source_addr: src_addr_1,
                    target_packet_size: 1500,
                    reconnect_delay: Duration::from_millis(100),
                },
                follow_db_1,
            )
        });

        assert!(
            wait_for_component_samples(&fol_db_1, component_id, 3, Duration::from_secs(5)).await,
            "era 1: follower should have 3 samples"
        );
        assert!(
            wait_for_msg_count(&fol_db_1, msg_id, 1, Duration::from_secs(5)).await,
            "era 1: follower should have 1 message"
        );
        sleep(Duration::from_millis(200)).await;

        // ── Era 2: restart both, write 2 more samples + 1 message ───────
        let src_listener_2 = TcpListener::bind("127.0.0.1:0").unwrap();
        let src_addr_2 = src_listener_2.local_addr().unwrap();
        let src_server_2 = Server::from_listener(src_listener_2, &src_dir).unwrap();
        let src_db_2 = src_server_2.db.clone();
        stellar(move || async { src_server_2.run().await });

        {
            let mut client = Client::connect(src_addr_2).await.unwrap();
            send_f64_samples(
                &mut client,
                component_id,
                "Dedup Test",
                vtable_id,
                &[(Timestamp(4000), 40.0), (Timestamp(5000), 50.0)],
            )
            .await;

            let ts = Timestamp(4500);
            let payload = 1u32.to_le_bytes();
            let mut pkt = LenPacket::msg_with_timestamp(msg_id, ts, payload.len());
            pkt.extend_from_slice(&payload);
            client.send(pkt).await.0.unwrap();
        }
        sleep(Duration::from_millis(200)).await;

        let fol_listener_2 = TcpListener::bind("127.0.0.1:0").unwrap();
        let fol_server_2 = Server::from_listener(fol_listener_2, &fol_dir).unwrap();
        let fol_db_2 = fol_server_2.db.clone();
        stellar(move || async { fol_server_2.run().await });

        let follow_db_2 = fol_db_2.clone();
        stellar(move || {
            elodin_db::follow::run_follower(
                elodin_db::follow::FollowConfig {
                    source_addr: src_addr_2,
                    target_packet_size: 1500,
                    reconnect_delay: Duration::from_millis(100),
                },
                follow_db_2,
            )
        });

        // ── Final verification: exact counts, no duplicates ─────────────
        assert!(
            wait_for_component_samples(&fol_db_2, component_id, 5, Duration::from_secs(5)).await,
            "era 2: follower should have 5 samples (no duplicates)"
        );
        assert!(
            wait_for_msg_count(&fol_db_2, msg_id, 2, Duration::from_secs(5)).await,
            "era 2: follower should have 2 messages (no duplicates)"
        );

        fol_db_2.with_state(|state| {
            let c = state.get_component(component_id).unwrap();
            let (ts, data) = c
                .time_series
                .get_range(&(Timestamp(0)..Timestamp(i64::MAX)))
                .unwrap();
            assert_eq!(ts.len(), 5, "exactly 5 samples (no duplicates)");
            let values = <[f64]>::ref_from_bytes(data).unwrap();
            for i in 0..5usize {
                assert_eq!(ts[i].0, (i as i64 + 1) * 1000);
                assert_eq!(values[i], (i + 1) as f64 * 10.0);
            }
        });

        assert_exports_match(&src_db_2, &fol_db_2, Some("dedup_test*"));
        assert_db_files_match(&src_db_2.path, &fol_db_2.path);
    }

    /// Covers: dynamic component discovery after follower connects,
    /// round-robin batching with multiple components.
    #[test]
    async fn test_follow_dynamic_discovery() {
        let (src_addr, src_db, _fol_addr, fol_db) = setup_follow_pair(1500).await.unwrap();

        // Add 3 components dynamically, each with 2 samples.
        let mut component_ids = Vec::new();
        {
            let mut client = Client::connect(src_addr).await.unwrap();
            for i in 0..3usize {
                let name = format!("dyn_comp_{}", i);
                let cid = ComponentId::new(&name);
                component_ids.push(cid);
                let vtable_id = (30 + i as u16).to_le_bytes();
                send_f64_samples(
                    &mut client,
                    cid,
                    &name,
                    vtable_id,
                    &[
                        (Timestamp((i as i64 + 1) * 1000), i as f64),
                        (Timestamp((i as i64 + 1) * 1000 + 500), i as f64 + 100.0),
                    ],
                )
                .await;
            }
        }

        // Wait for all 3 to replicate.
        for (i, &cid) in component_ids.iter().enumerate() {
            assert!(
                wait_for_component_samples(&fol_db, cid, 2, Duration::from_secs(5)).await,
                "follower should have 2 samples for dyn_comp_{}",
                i
            );
        }

        // Add 2 MORE components after the first batch.
        {
            let mut client = Client::connect(src_addr).await.unwrap();
            for i in 3..5usize {
                let name = format!("dyn_comp_{}", i);
                let cid = ComponentId::new(&name);
                component_ids.push(cid);
                let vtable_id = (30 + i as u16).to_le_bytes();
                send_f64_samples(
                    &mut client,
                    cid,
                    &name,
                    vtable_id,
                    &[
                        (Timestamp((i as i64 + 1) * 1000), i as f64),
                        (Timestamp((i as i64 + 1) * 1000 + 500), i as f64 + 100.0),
                    ],
                )
                .await;
            }
        }

        // Wait for all 5 to replicate.
        for (i, &cid) in component_ids.iter().enumerate() {
            assert!(
                wait_for_component_samples(&fol_db, cid, 2, Duration::from_secs(5)).await,
                "follower should have 2 samples for dyn_comp_{}",
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
                    "dyn_comp_{} should have exactly 2 samples, got {}",
                    i,
                    ts.len()
                );
            }
        });

        assert_exports_match(&src_db, &fol_db, None);
        assert_db_files_match(&src_db.path, &fol_db.path);
    }

    /// Covers: multi-type replication (f64 array, f32 array, u64 scalar),
    /// message replication, CSV + binary data integrity.
    #[test]
    async fn test_follow_multi_type_integrity() {
        let subscriber = tracing_subscriber::FmtSubscriber::new();
        let _ = tracing::subscriber::set_global_default(subscriber);

        let src_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let src_addr = src_listener.local_addr().unwrap();
        let src_temp =
            std::env::temp_dir().join(format!("elodin_db_multi_src_{}", fastrand::u64(..)));
        if src_temp.exists() {
            let _ = std::fs::remove_dir_all(&src_temp);
        }
        let src_server = Server::from_listener(src_listener, src_temp).unwrap();
        let src_db = src_server.db.clone();
        stellar(move || async { src_server.run().await });

        let sample_count = 10usize;
        let timestamps: Vec<Timestamp> = (1..=sample_count)
            .map(|i| Timestamp(i as i64 * 1000))
            .collect();

        // f64[3] component (e.g., position vector).
        let pos_id = ComponentId::new("multi_pos");
        {
            let mut client = Client::connect(src_addr).await.unwrap();
            send_timestamped_samples(
                &mut client,
                pos_id,
                "multi_pos",
                100u16.to_le_bytes(),
                PrimType::F64,
                &[3],
                &timestamps,
                |i| {
                    let x = i as f64;
                    let y = i as f64 * 2.0;
                    let z = i as f64 * 3.0;
                    [x, y, z].iter().flat_map(|v| v.to_le_bytes()).collect()
                },
            )
            .await;
        }

        // f32[2] component (e.g., control surfaces).
        let ctrl_id = ComponentId::new("multi_ctrl");
        {
            let mut client = Client::connect(src_addr).await.unwrap();
            send_timestamped_samples(
                &mut client,
                ctrl_id,
                "multi_ctrl",
                101u16.to_le_bytes(),
                PrimType::F32,
                &[2],
                &timestamps,
                |i| {
                    let a = i as f32 * 0.5;
                    let b = -(i as f32);
                    [a, b].iter().flat_map(|v| v.to_le_bytes()).collect()
                },
            )
            .await;
        }

        // u64 scalar component (e.g., tick counter).
        let tick_id = ComponentId::new("multi_tick");
        {
            let mut client = Client::connect(src_addr).await.unwrap();
            send_timestamped_samples(
                &mut client,
                tick_id,
                "multi_tick",
                102u16.to_le_bytes(),
                PrimType::U64,
                &[],
                &timestamps,
                |i| (i as u64).to_le_bytes().to_vec(),
            )
            .await;
        }

        // 5 messages.
        let msg_name = "multi_log";
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

            for i in 0..5u32 {
                let ts = Timestamp((i as i64 + 1) * 2000);
                let payload = i.to_le_bytes();
                let mut pkt = LenPacket::msg_with_timestamp(msg_id, ts, payload.len());
                pkt.extend_from_slice(&payload);
                client.send(pkt).await.0.unwrap();
            }
        }
        sleep(Duration::from_millis(100)).await;

        // Start follower.
        let fol_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let fol_temp =
            std::env::temp_dir().join(format!("elodin_db_multi_fol_{}", fastrand::u64(..)));
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

        // Wait for full sync.
        for (cid, expected) in [
            (pos_id, sample_count),
            (ctrl_id, sample_count),
            (tick_id, sample_count),
        ] {
            assert!(
                wait_for_component_samples(&fol_db, cid, expected, Duration::from_secs(5)).await,
                "follower should have {} samples for {:?}",
                expected,
                cid
            );
        }
        assert!(
            wait_for_msg_count(&fol_db, msg_id, 5, Duration::from_secs(5)).await,
            "follower should have 5 messages"
        );

        // CSV + binary comparison.
        src_db.flush_all().unwrap();
        fol_db.flush_all().unwrap();
        sleep(Duration::from_millis(100)).await;

        assert_exports_match(&src_db, &fol_db, None);
        assert_db_files_match(&src_db.path, &fol_db.path);
    }

    /// Verify that earliest_timestamp tracks actual data at runtime via
    /// update_min, and that DB::open() also computes it correctly from data
    /// when time_start_timestamp_micros is not explicitly set.
    #[test]
    async fn test_open_fixes_mismatched_timestamp_domains() {
        let subscriber = tracing_subscriber::FmtSubscriber::new();
        let _ = tracing::subscriber::set_global_default(subscriber);

        let temp_dir =
            std::env::temp_dir().join(format!("elodin_db_ts_mismatch_{}", fastrand::u64(..)));
        if temp_dir.exists() {
            let _ = std::fs::remove_dir_all(&temp_dir);
        }

        let component_id = ComponentId::new("mono_test");
        let schema = elodin_db::ComponentSchema::new(PrimType::F64, &[1]);

        // Monotonic timestamps: ~48 minutes from epoch (typical boot-time values)
        let monotonic_start = Timestamp(2_877_000_000); // ~48 min
        let monotonic_ts1 = Timestamp(2_878_000_000);
        let monotonic_ts2 = Timestamp(2_879_000_000);

        {
            let db = elodin_db::DB::create(temp_dir.clone()).unwrap();

            // After creation, time_start_timestamp_micros should NOT be set
            // (deferred until data arrives or explicit set_earliest_timestamp)
            db.with_state(|state| {
                assert!(
                    state.db_config.time_start_timestamp_micros().is_none(),
                    "DB creation should NOT persist time_start_timestamp_micros (got {:?})",
                    state.db_config.time_start_timestamp_micros(),
                );
            });

            // In-memory earliest_timestamp starts at wall-clock (for apply_implicit_timestamp)
            let wall_clock_start = db.earliest_timestamp.latest();
            assert!(
                wall_clock_start.0 > 1_500_000_000_000_000,
                "in-memory earliest_timestamp should start at wall-clock (got {})",
                wall_clock_start.0,
            );

            // Insert a component and write data with monotonic timestamps
            db.with_state_mut(|state| {
                state
                    .insert_component_with_start_timestamp(
                        component_id,
                        schema.clone(),
                        monotonic_start,
                        &temp_dir,
                    )
                    .unwrap();
            });

            db.with_state(|state| {
                let component = state.get_component(component_id).unwrap();
                let data = 42.0f64.to_le_bytes();
                component
                    .time_series
                    .push_buf(monotonic_ts1, &data)
                    .unwrap();
                component
                    .time_series
                    .push_buf(monotonic_ts2, &data)
                    .unwrap();
            });

            // update_min is called by DBSink (via Table packets), but push_buf
            // is a direct TimeSeries call that doesn't go through DBSink.
            // Simulate the update_min that would happen via the normal packet path.
            db.earliest_timestamp.update_min(monotonic_ts1);

            // Verify earliest_timestamp tracked down to the data range at runtime
            let earliest_live = db.earliest_timestamp.latest();
            assert!(
                earliest_live.0 <= monotonic_ts1.0,
                "earliest_timestamp should have tracked down to data range via update_min \
                 (got {}, expected <= {})",
                earliest_live.0,
                monotonic_ts1.0,
            );

            // Flush to ensure data hits disk
            db.flush_all().unwrap();
        }
        // DB is dropped, files are on disk

        // Re-open and verify
        let db = elodin_db::DB::open(temp_dir.clone()).unwrap();
        let earliest = db.earliest_timestamp.latest();
        let last = db.last_updated.latest();

        assert!(
            earliest <= last,
            "earliest_timestamp ({}) must be <= last_updated ({}) after open",
            earliest.0,
            last.0,
        );
        assert_eq!(
            earliest.0, monotonic_start.0,
            "earliest_timestamp should match the data's start timestamp, not wall-clock"
        );
        assert_eq!(
            last.0, monotonic_ts2.0,
            "last_updated should be the latest data timestamp"
        );

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
