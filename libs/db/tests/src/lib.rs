#[cfg(test)]
mod tests {

    use arrow::{array::AsArray, datatypes::Float64Type};
    use elodin_db::{DB, Error, Server};
    use impeller2::{
        types::{ComponentId, EntityId, IntoLenPacket, LenPacket, Msg, PrimType, Timestamp},
        vtable::builder::{pair, raw_field, raw_table, schema, timestamp, vtable},
    };
    use impeller2_stellar::Client;
    use postcard_schema::{Schema, schema::owned::OwnedNamedType};
    use std::{borrow::Cow, collections::HashMap, net::SocketAddr, sync::Arc, time::Duration};
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
            std::fs::remove_dir_all(&temp_dir).unwrap();
        }
        std::fs::create_dir_all(&temp_dir).unwrap();

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
            schema(PrimType::F64, &[2], pair(1, "test")),
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
                .get_component(EntityId(1), ComponentId::new("test"))
                .expect("missing component");
            let (_, data) = c.time_series.latest().expect("missing latest value");
            assert_eq!(data, floats.as_bytes());
        })
    }

    #[test]
    async fn test_vtable_stream() {
        let (addr, _) = setup_test_db().await.unwrap();
        let mut tx_client = Client::connect(addr).await.unwrap();
        let mut rx_client = Client::connect(addr).await.unwrap();

        let entity_id = EntityId(42);
        let component_id = ComponentId::new("temperature");

        let vtable = vtable([raw_field(
            0,
            8,
            schema(PrimType::F64, &[1], pair(entity_id, component_id)),
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
        let entity_id = EntityId(42);
        let entity_metadata = SetEntityMetadata::new(entity_id, "test_entity").metadata(
            [("foo".to_string(), "bar".to_string())]
                .into_iter()
                .collect(),
        );
        let component_id = ComponentId::new("test_component");
        let component_metadata = SetComponentMetadata::new(component_id.clone(), "Test Component")
            .metadata(
                [("baz".to_string(), "bang".to_string())]
                    .into_iter()
                    .collect(),
            );

        client.send(&entity_metadata).await.0.unwrap();
        client.send(&component_metadata).await.0.unwrap();
        sleep(Duration::from_millis(100)).await;
        let mut response = client.request(&DumpMetadata).await.unwrap();
        response.entity_metadata.sort_by_key(|e| e.entity_id);
        assert_eq!(
            response.entity_metadata,
            &[EntityMetadata {
                entity_id,
                name: "test_entity".to_string(),
                metadata: [("foo".to_string(), "bar".to_string())]
                    .into_iter()
                    .collect(),
            },]
        );
        response.component_metadata.sort_by_key(|c| c.component_id);
        assert_eq!(
            response.component_metadata,
            &[ComponentMetadata {
                component_id,
                name: "Test Component".to_string(),
                metadata: [("baz".to_string(), "bang".to_string())]
                    .into_iter()
                    .collect(),
                asset: false,
            },]
        )
    }

    #[test]
    async fn test_sql_query() {
        let (addr, _db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();

        let entity_id = EntityId(42);
        let component_id = ComponentId::new("temperature");

        let vtable = vtable([raw_field(
            0,
            8,
            schema(PrimType::F64, &[1], pair(entity_id, component_id)),
        )]);

        client
            .send(&SetComponentMetadata::new(component_id, "temperature"))
            .await
            .0
            .unwrap();
        client
            .send(&SetEntityMetadata::new(entity_id, "cpu"))
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
            .column_by_name("temperature")
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

        let entity_id = EntityId(42);
        let component_id = ComponentId::new("sensor_data");

        let vtable = vtable([raw_field(
            0,
            8,
            timestamp(
                raw_table(8, 8),
                schema(PrimType::F64, &[], pair(entity_id, component_id)),
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
            entity_id,
            component_id,
            limit: Some(10),
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
            schema(PrimType::F64, &[1], pair(1, component_id)),
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
        let metadata = SetComponentMetadata::new(component_id, "Temperature Sensor")
            .metadata(
                [("unit".to_string(), "celsius".to_string())]
                    .into_iter()
                    .collect(),
            )
            .asset(false);

        client.send(&metadata).await.0.unwrap();
        sleep(Duration::from_millis(50)).await;

        let get_metadata = GetComponentMetadata { component_id };
        let component_metadata = client.request(&get_metadata).await.unwrap();

        assert_eq!(component_metadata.component_id, component_id);
        assert_eq!(component_metadata.name, "Temperature Sensor");
        assert_eq!(component_metadata.metadata.get("unit").unwrap(), "celsius");
        assert_eq!(component_metadata.asset, false);
    }

    #[test]
    async fn test_get_entity_metadata() {
        let (addr, _db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();

        let entity_id = EntityId(42);
        let metadata = SetEntityMetadata::new(entity_id, "Test Robot").metadata(
            [("type".to_string(), "mobile".to_string())]
                .into_iter()
                .collect(),
        );

        client.send(&metadata).await.0.unwrap();
        sleep(Duration::from_millis(50)).await;

        let get_metadata = GetEntityMetadata { entity_id };
        let entity_metadata = client.request(&get_metadata).await.unwrap();

        assert_eq!(entity_metadata.entity_id, entity_id);
        assert_eq!(entity_metadata.name, "Test Robot");
        assert_eq!(entity_metadata.metadata.get("type").unwrap(), "mobile");
    }

    #[test]
    async fn test_get_asset() {
        let (addr, _db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();

        let asset_id = 0;
        let test_data = vec![1, 2, 3, 4, 5];

        let set_asset = SetAsset {
            id: asset_id,
            buf: Cow::Owned(test_data.clone()),
        };

        client.send(&set_asset).await.0.unwrap();
        sleep(Duration::from_millis(50)).await;

        let get_asset = GetAsset { id: asset_id };
        let asset = client.request(&get_asset).await.unwrap();

        assert_eq!(asset.id, asset_id);
        assert_eq!(asset.buf.as_ref(), test_data.as_slice());
    }

    #[test]
    async fn test_update_asset() {
        let (addr, _db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();

        let asset_id = 0;
        let test_data = vec![1, 2, 3, 4, 5];

        let set_asset = SetAsset {
            id: asset_id,
            buf: Cow::Owned(test_data.clone()),
        };

        client.send(&set_asset).await.0.unwrap();
        let asset_id = 0;
        let test_data = vec![0xFF, 0xFF];

        let set_asset = SetAsset {
            id: asset_id,
            buf: Cow::Owned(test_data.clone()),
        };

        client.send(&set_asset).await.0.unwrap();
        sleep(Duration::from_millis(50)).await;

        let get_asset = GetAsset { id: asset_id };
        let asset = client.request(&get_asset).await.unwrap();

        assert_eq!(asset.id, asset_id);
        assert_eq!(asset.buf.as_ref(), test_data.as_slice());
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
            schema(PrimType::F64, &[1], pair(1, component_id1)),
        )]);

        let vtable2 = vtable([raw_field(
            0,
            4,
            schema(PrimType::F32, &[2, 2], pair(2, component_id2)),
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
        let entity_id = EntityId(42);
        let component_id = ComponentId::new("archive_test");

        // Set metadata for our test data
        client
            .send(&SetEntityMetadata::new(entity_id, "TestEntity"))
            .await
            .0
            .unwrap();

        client
            .send(&SetComponentMetadata::new(component_id, "TestComponent"))
            .await
            .0
            .unwrap();

        let vtable = vtable([raw_field(
            0,
            8,
            schema(PrimType::F64, &[1], pair(entity_id, component_id)),
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

        let file =
            std::fs::File::open(&archive_path.join("TestEntity_TestComponent.arrow")).unwrap();
        let mut reader = arrow::ipc::reader::FileReader::try_new(file, None).unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 3);
        let _ = batch.column_by_name("time").unwrap();
        let component = batch.column_by_name("TestEntity_TestComponent").unwrap();
        let values = component
            .as_fixed_size_list()
            .values()
            .as_primitive::<Float64Type>()
            .values();
        assert_eq!(values, &[10.5, 20.5, 30.5]);
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

        let non_existent_entity_id = EntityId(9999);
        let component_id = ComponentId::new("test_component");

        client
            .send(&VTableMsg {
                id: 1u16.to_le_bytes(),
                vtable: vtable([raw_field(
                    0,
                    4,
                    schema(PrimType::F32, &[], pair(2, component_id)),
                )]),
            })
            .await
            .0
            .unwrap();

        let query = GetTimeSeries {
            id: 1u16.to_le_bytes(),
            range: Timestamp(0)..Timestamp(10000),
            entity_id: non_existent_entity_id,
            component_id,
            limit: Some(10),
        };

        let result = client.request(&query).await;

        let Err(impeller2_stellar::Error::Response(resp)) = result else {
            panic!("invalid error");
        };
        assert_eq!(
            elodin_db::Error::EntityNotFound(non_existent_entity_id).to_string(),
            resp.description
        );

        // Try with existing entity but non-existent component
        let existing_entity_id = EntityId(0); // Global entity always exists
        let non_existent_component_id = ComponentId::new("non_existent_component");

        let query = GetTimeSeries {
            id: 1u16.to_le_bytes(),
            range: Timestamp(0)..Timestamp(10000),
            entity_id: existing_entity_id,
            component_id: non_existent_component_id,
            limit: Some(10),
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
    async fn test_get_entity_metadata_not_found() {
        let (addr, _db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();

        let non_existent_entity_id = EntityId(9999);

        let get_metadata = GetEntityMetadata {
            entity_id: non_existent_entity_id,
        };

        let result = client.request(&get_metadata).await.unwrap_err();
        let impeller2_stellar::Error::Response(resp) = result else {
            panic!("invalid error");
        };
        assert_eq!(
            elodin_db::Error::EntityNotFound(non_existent_entity_id).to_string(),
            resp.description
        );
    }

    #[test]
    async fn test_get_asset_not_found() {
        let (addr, _db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();

        // Try to get a non-existent asset
        let asset_id = 9999;

        let get_asset = GetAsset { id: asset_id };

        let result = client.request(&get_asset).await.unwrap_err();
        let impeller2_stellar::Error::Response(resp) = result else {
            panic!("invalid error");
        };
        assert_eq!(
            elodin_db::Error::AssetNotFound(asset_id).to_string(),
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

        let entity_id = EntityId(99);
        let component_id = ComponentId::new("concurrent_test");

        // Define a vtable that will be used by all clients
        let vtable = vtable([raw_field(
            0,
            8,
            schema(PrimType::F64, &[1], pair(entity_id, component_id)),
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
        setup_client
            .send(&SetEntityMetadata::new(entity_id, "Concurrent Test Entity"))
            .await
            .0
            .unwrap();

        sleep(Duration::from_millis(50)).await;

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
                    sleep(Duration::from_millis(5)).await;
                }

                client_id
            }));
        }

        for handle in join_handles {
            let _ = handle.await;
        }

        sleep(Duration::from_millis(120)).await;

        // Verify that data was written by all clients
        let mut verification_client = Client::connect(addr).await.unwrap();
        let query = GetTimeSeries {
            id: vtable_id,
            range: Timestamp(0)..Timestamp(i64::MAX),
            entity_id,
            component_id,
            limit: Some(NUM_CLIENTS * WRITES_PER_CLIENT),
        };

        let time_series = verification_client.request(&query).await.unwrap();
        let data = <[f64]>::ref_from_bytes(time_series.data().unwrap()).unwrap();
        assert_eq!(data.len(), NUM_CLIENTS * WRITES_PER_CLIENT);
    }

    #[test]
    async fn test_database_restart() {
        let entity_id = EntityId(123);

        let temp_dir =
            std::env::temp_dir().join(format!("elodin_db_restart_test_{}", fastrand::u64(..)));
        std::fs::create_dir_all(&temp_dir).unwrap();
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
                .send(&SetEntityMetadata::new(entity_id, "Restart Test Entity"))
                .await
                .0
                .unwrap();
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
                schema(PrimType::F64, &[1], pair(entity_id, component_id)),
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
                entity_id,
                component_id,
                limit: Some(10),
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

        let entity_metadata = new_client
            .request(&GetEntityMetadata { entity_id })
            .await
            .unwrap();
        assert_eq!(entity_metadata.name, "Restart Test Entity");

        let component_metadata = new_client
            .request(&GetComponentMetadata { component_id })
            .await
            .unwrap();
        assert_eq!(component_metadata.name, "Restart Test Component");

        let get_schema = GetSchema { component_id };
        let SchemaMsg(schema) = new_client.request(&get_schema).await.unwrap();
        assert_eq!(schema.prim_type(), PrimType::F64);
        assert_eq!(schema.shape(), &[1]);

        let new_query = GetTimeSeries {
            id: vtable_id,
            range: Timestamp(0)..Timestamp(i64::MAX),
            entity_id,
            component_id,
            limit: Some(10),
        };
        let new_time_series = new_client.request(&new_query).await.unwrap();
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

        let entity_id = EntityId(789);
        let component_id = ComponentId::new("large_data");

        let vtable = vtable([raw_field(
            0,
            800,
            schema(PrimType::F64, &[100], pair(entity_id, component_id)),
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
            .send(&SetEntityMetadata::new(entity_id, "Performance Test"))
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
            entity_id,
            component_id,
            limit: Some(10),
        };

        let time_series_limited = client.request(&query_limited).await.unwrap();
        assert!(time_series_limited.timestamps().unwrap().len() <= 10);

        let query_all = GetTimeSeries {
            id: vtable_id,
            range: Timestamp(0)..Timestamp(i64::MAX),
            entity_id,
            component_id,
            limit: None,
        };

        let time_series_all = client.request(&query_all).await.unwrap();

        assert!(time_series_all.timestamps().unwrap().len() > 0);

        let data_flat = time_series_all.data().unwrap();
        assert_eq!(data_flat.len() % (100 * 8), 0);
    }

    #[test]
    async fn test_complex_query() {
        let (addr, _db) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();

        let entity_id1 = EntityId(601);
        let entity_id2 = EntityId(602);
        let component_id = ComponentId::new("temperature");

        client
            .send(&SetEntityMetadata::new(entity_id1, "Sensor1"))
            .await
            .0
            .unwrap();
        client
            .send(&SetEntityMetadata::new(entity_id2, "Sensor2"))
            .await
            .0
            .unwrap();
        client
            .send(&SetComponentMetadata::new(component_id, "Temperature"))
            .await
            .0
            .unwrap();

        // Create vtables for both entities
        let vtable1 = vtable([raw_field(
            0,
            8,
            schema(PrimType::F64, &[], pair(entity_id1, component_id)),
        )]);

        let vtable2 = vtable([raw_field(
            0,
            8,
            schema(PrimType::F64, &[], pair(entity_id2, component_id)),
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

        let sql = "SELECT * FROM sensor_1_temperature, sensor_2_temperature WHERE sensor_1_temperature.temperature > 22.0";
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

        let sql_simple =
            "SELECT * FROM sensor_1_temperature UNION ALL SELECT * FROM sensor_2_temperature";
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

        let entity_id = EntityId(42);
        let component_id = ComponentId::new("test_component");

        let vtable = vtable([raw_field(
            0,
            8,
            schema(PrimType::F64, &[], pair(entity_id, component_id)),
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

        let entity_id = EntityId(42);
        let component_id = ComponentId::new("test_component");
        {
            let vtable = vtable([raw_field(
                0,
                8,
                schema(PrimType::F64, &[1], pair(entity_id, component_id)),
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
            schema(PrimType::F32, &[1], pair(entity_id, component_id)),
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
        let (addr, _) = setup_test_db().await.unwrap();
        let mut client = Client::connect(addr).await.unwrap();

        let entity_id = EntityId(42);
        let component_id = ComponentId::new("test_component");
        let vtable = vtable([raw_field(
            0,
            8,
            timestamp(
                raw_table(8, 8),
                schema(PrimType::F64, &[1], pair(entity_id, component_id)),
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

    #[test]
    async fn test_db_reopen() {
        let temp_dir =
            std::env::temp_dir().join(format!("elodin_db_persistence_test_{}", fastrand::u64(..)));
        std::fs::create_dir_all(&temp_dir).unwrap();

        // Create entity and component identifiers
        let entity_id = EntityId(55);
        let component_id = ComponentId::new("persistence_test");
        let vtable_id = 1u16.to_le_bytes();
        let test_value = 123.45f64;
        let asset_id = 77;
        let asset_data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

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
                    &SetEntityMetadata::new(entity_id, "Persistence Test Entity").metadata(
                        [("type".to_string(), "test".to_string())]
                            .into_iter()
                            .collect(),
                    ),
                )
                .await
                .0
                .unwrap();

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
                schema(PrimType::F64, &[1], pair(entity_id, component_id)),
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

            let set_asset = SetAsset {
                id: asset_id,
                buf: Cow::Owned(asset_data.clone()),
            };

            client.send(&set_asset).await.0.unwrap();

            sleep(Duration::from_millis(100)).await;
        }

        let db = elodin_db::DB::open(temp_dir.clone()).unwrap();
        db.with_state_mut(|state| {
            let entity = state.get_entity_metadata(entity_id).unwrap();
            assert_eq!(
                entity.metadata.clone(),
                [("type".to_string(), "test".to_string())]
                    .into_iter()
                    .collect::<HashMap<_, _>>(),
            );
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
}
